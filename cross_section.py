from __future__ import annotations

import heapq
from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    from .models import DetectorConfig
except ImportError:
    from models import DetectorConfig


def make_sampling_grid(radius_mm: float, pixel_mm: float):
    count = int(np.ceil(radius_mm / pixel_mm))
    offsets = np.arange(-count, count + 1, dtype=float) * pixel_mm
    grid_u, grid_v = np.meshgrid(offsets, offsets, indexing="ij")
    radial_distance = np.hypot(grid_u, grid_v)
    return grid_u, grid_v, radial_distance


def sample_plane(
    volume: np.ndarray,
    center_voxel_zyx: np.ndarray,
    frame_u_mm: np.ndarray,
    frame_v_mm: np.ndarray,
    spacing_zyx: Sequence[float],
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    order: int,
    cval: float,
) -> np.ndarray:
    spacing = np.asarray(spacing_zyx, dtype=float)
    center_mm = np.asarray(center_voxel_zyx, dtype=float) * spacing
    points_mm = (
        center_mm[:, None, None]
        + frame_u_mm[:, None, None] * grid_u[None]
        + frame_v_mm[:, None, None] * grid_v[None]
    )
    coordinates = points_mm / spacing[:, None, None]
    return ndi.map_coordinates(
        volume,
        coordinates,
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=order > 1,
    )


def _connected_component_at_seed(mask: np.ndarray, seed: Tuple[int, int]) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    seed_label = int(labels[seed])
    if seed_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == seed_label


def _priority_region_grow(
    image: np.ndarray,
    gradient: np.ndarray,
    allowed: np.ndarray,
    seed_mask: np.ndarray,
    config: DetectorConfig,
) -> Tuple[np.ndarray, float, float]:
    """ 区域生长算法实现 """
    
    seed_values = image[seed_mask & allowed]
    if seed_values.size == 0:
        return np.zeros_like(allowed, dtype=bool), 0.0, 0.0

    seed_intensity = float(np.median(seed_values))
    grad_values = gradient[allowed]
    positive_gradients = grad_values[grad_values > 1e-8]
    if positive_gradients.size:
        gradient_scale = float(
            np.percentile(positive_gradients, config.gradient_percentile)
        )
    else:
        gradient_scale = 1.0
    gradient_scale = max(gradient_scale, 1e-6)

    intensity_cost = np.abs(image - seed_intensity) / max(config.intensity_tolerance, 1e-6)
    gradient_cost = gradient / gradient_scale
    # A strong edge should stop growth mainly when intensity is also departing
    # from the seed distribution. This keeps the inner half of a sharp lumen
    # boundary while rejecting the background side of the same edge.
    gradient_gate = np.minimum(1.0, intensity_cost + 0.10)
    total_cost = (
        config.intensity_weight * intensity_cost
        + config.gradient_weight * gradient_cost * gradient_gate
    )

    accepted = np.zeros_like(allowed, dtype=bool)
    visited = np.zeros_like(allowed, dtype=bool)
    queue = []
    for y, x in np.argwhere(seed_mask & allowed):
        heapq.heappush(queue, (float(total_cost[y, x]), int(y), int(x)))

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    height, width = allowed.shape

    # Lazy Dijkstra
    while queue:
        cost, y, x = heapq.heappop(queue)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if not allowed[y, x] or cost > config.region_cost_threshold:
            continue
        accepted[y, x] = True
        for dy, dx in neighbor_offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                if allowed[ny, nx]:
                    heapq.heappush(queue, (float(total_cost[ny, nx]), ny, nx))

    accepted = ndi.binary_closing(accepted, structure=np.ones((3, 3), dtype=bool))
    accepted = ndi.binary_fill_holes(accepted)
    center = (height // 2, width // 2)
    accepted = _connected_component_at_seed(accepted, center)
    return accepted, seed_intensity, gradient_scale


def segment_cross_section(
    image_plane: np.ndarray,
    gradient_plane: np.ndarray,
    prior_plane: np.ndarray,
    allowed_plane: np.ndarray,
    radial_distance: np.ndarray, # 截面采样网格上每个像素到中心的平面距离
    config: DetectorConfig,
):
    center = (image_plane.shape[0] // 2, image_plane.shape[1] // 2)
    seed_mask = radial_distance <= config.seed_radius_mm
    seed_mask &= allowed_plane
    if not seed_mask[center]:
        seed_mask[center] = bool(allowed_plane[center])

    region, _, _ = _priority_region_grow(
        image_plane, gradient_plane, allowed_plane, seed_mask, config
    )
    prior_component = _connected_component_at_seed(prior_plane, center)
    pixel_area = config.cross_section_pixel_mm ** 2
    region_area = float(np.count_nonzero(region) * pixel_area)
    prior_area = float(np.count_nonzero(prior_component) * pixel_area)
    flags = []

    area_ratio = region_area / max(prior_area, pixel_area)
    invalid = (
        region_area < config.min_lumen_area_mm2
        or area_ratio < config.min_area_ratio_to_prior
        or area_ratio > config.max_area_ratio_to_prior
        or not region[center]
    )
    if invalid:
        flags.append("region_grow_invalid")
        # 强度分割不可靠时，改用 label 截面当管腔
        if config.use_label_fallback and prior_area >= config.min_lumen_area_mm2:
            region = prior_component
            region_area = prior_area
            flags.append("used_label_fallback")

    # 判断是否碰到边界
    if np.any(region[[0, -1], :]) or np.any(region[:, [0, -1]]):
        flags.append("touches_patch_boundary")
    if prior_area <= 0:
        flags.append("empty_prior_section")

    if region_area > 0 and prior_area > 0:
        intersection = np.count_nonzero(region & prior_component)
        prior_containment = intersection / max(np.count_nonzero(region), 1)
    else:
        prior_containment = 0.0
    quality = float(np.clip(0.25 + 0.75 * prior_containment, 0.0, 1.0))
    if "used_label_fallback" in flags:
        quality = min(quality, 0.45)
    if "touches_patch_boundary" in flags:
        quality *= 0.6

    return region, region_area, prior_area, quality, flags
