from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    from .models import BranchResult, CPRResult
except ImportError:
    from models import BranchResult, CPRResult


def global_vector_of_interest_zyx(
    angle_degrees: float,
    image_direction_xyz: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a transverse LPS direction and its image-axis z-y-x form.

    Zero degrees is patient LPS +X. Positive angles rotate toward LPS +Y
    around patient LPS +Z according to the right-hand rule.
    """
    if not np.isfinite(angle_degrees):
        raise ValueError("angle_degrees must be finite.")
    if image_direction_xyz is None:
        direction = np.eye(3, dtype=float)
    else:
        direction = np.asarray(image_direction_xyz, dtype=float)
        if direction.size != 9:
            raise ValueError("image_direction_xyz must contain nine values.")
        direction = direction.reshape(3, 3)
    if np.any(~np.isfinite(direction)):
        raise ValueError("image_direction_xyz must contain finite values.")
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("image_direction_xyz must be orthonormal.")

    angle_radians = np.deg2rad(float(angle_degrees))
    direction_lps_xyz = np.asarray(
        [np.cos(angle_radians), np.sin(angle_radians), 0.0],
        dtype=float,
    )
    direction_image_xyz = direction.T @ direction_lps_xyz
    direction_image_zyx = direction_image_xyz[::-1].copy()
    direction_image_zyx /= np.linalg.norm(direction_image_zyx)
    return direction_image_zyx, direction_lps_xyz


def sample_cpr(
    image: np.ndarray,
    centers_voxel_zyx: np.ndarray,
    frames_u_zyx: np.ndarray,
    frames_v_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    angle_degrees: float,
    radius_mm: float = 5.0,
    pixel_mm: float = 0.15,
    interpolation_order: int = 1,
    cval: float | None = None,
):
    """Sample one line per transported frame and stack lines into a CPR."""
    volume = np.asarray(image)
    centers = np.asarray(centers_voxel_zyx, dtype=float)
    frames_u = np.asarray(frames_u_zyx, dtype=float)
    frames_v = np.asarray(frames_v_zyx, dtype=float)
    spacing = np.asarray(spacing_zyx, dtype=float)

    if volume.ndim != 3:
        raise ValueError("image must be a 3-D array in z-y-x order.")
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("centers_voxel_zyx must have shape (N, 3).")
    if frames_u.shape != centers.shape or frames_v.shape != centers.shape:
        raise ValueError("CPR frame arrays must have the same shape as centers.")
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive values.")
    if radius_mm <= 0 or pixel_mm <= 0:
        raise ValueError("CPR radius and pixel size must be positive.")
    if not np.isfinite(angle_degrees):
        raise ValueError("angle_degrees must be finite.")
    if interpolation_order < 0 or interpolation_order > 5:
        raise ValueError("interpolation_order must be between 0 and 5.")

    angle_radians = np.deg2rad(float(angle_degrees))
    directions = (
        np.cos(angle_radians) * frames_u
        + np.sin(angle_radians) * frames_v
    )
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(direction_norms < 1e-8):
        raise ValueError("A CPR sampling direction has near-zero length.")
    directions /= direction_norms

    half_count = max(1, int(np.ceil(radius_mm / pixel_mm)))
    offsets = np.linspace(-float(radius_mm), float(radius_mm), 2 * half_count + 1, dtype=float)
    centers_mm = centers * spacing
    sample_points_mm = (centers_mm[:, None, :] + offsets[None, :, None] * directions[:, None, :])
    coordinates = np.moveaxis(sample_points_mm / spacing[None, None, :], -1,0)

    if cval is None:
        finite = volume[np.isfinite(volume)]
        cval = float(np.min(finite)) if finite.size else 0.0
    sampled = ndi.map_coordinates(
        volume,
        coordinates,
        order=int(interpolation_order),
        mode="constant",
        cval=float(cval),
        prefilter=interpolation_order > 1,
    )
    return sampled, offsets


def _prepare_centerline_geometry(
    centerline_voxel_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    frames_u_zyx: np.ndarray | None,
    frames_v_zyx: np.ndarray | None,
    centerline_distances_mm: np.ndarray | None,
    curve_resolution_mm: float | None,
    centerline_smoothing_sigma_mm: float,
):
    centerline = np.asarray(centerline_voxel_zyx, dtype=float)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 3:
        raise ValueError("centerline_voxel_zyx must have shape (N, 3).")
    if len(centerline) < 2:
        raise ValueError("At least two ordered centerline points are required.")
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive values.")

    if curve_resolution_mm is not None:
        if frames_u_zyx is not None or frames_v_zyx is not None:
            raise ValueError(
                "Explicit frames cannot be combined with centerline resampling."
            )
        if centerline_distances_mm is not None:
            raise ValueError(
                "Explicit distances cannot be combined with centerline resampling."
            )
        try:
            from .centerline import smooth_and_resample_centerline
        except ImportError:
            from centerline import smooth_and_resample_centerline

        centerline, centerline_distances_mm = smooth_and_resample_centerline(
            centerline,
            spacing,
            step_mm=float(curve_resolution_mm),
            smoothing_sigma_mm=centerline_smoothing_sigma_mm,
        )

    if (frames_u_zyx is None) != (frames_v_zyx is None):
        raise ValueError("frames_u_zyx and frames_v_zyx must be supplied together.")
    if frames_u_zyx is None:
        try:
            from .centerline import tangents_and_frames
        except ImportError:
            from centerline import tangents_and_frames

        _, frames_u, frames_v = tangents_and_frames(centerline, spacing)
    else:
        frames_u = np.asarray(frames_u_zyx, dtype=float)
        frames_v = np.asarray(frames_v_zyx, dtype=float)
        if frames_u.shape != centerline.shape or frames_v.shape != centerline.shape:
            raise ValueError("Explicit frame arrays must match the centerline shape.")

    if centerline_distances_mm is None:
        segment_lengths = np.linalg.norm(
            np.diff(centerline * spacing, axis=0),
            axis=1,
        )
        distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    else:
        distances = np.asarray(centerline_distances_mm, dtype=float)
        if distances.shape != (len(centerline),):
            raise ValueError(
                "centerline_distances_mm must have one value per centerline point."
            )
        if np.any(~np.isfinite(distances)) or np.any(np.diff(distances) < 0):
            raise ValueError(
                "centerline_distances_mm must be finite and nondecreasing."
            )
    return centerline, spacing, frames_u, frames_v, distances


def generate_centerline_cpr(
    image: np.ndarray,
    centerline_voxel_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    angle_degrees: float,
    frames_u_zyx: np.ndarray | None = None,
    frames_v_zyx: np.ndarray | None = None,
    centerline_distances_mm: np.ndarray | None = None,
    distance_from_root_mm: float = 0.0,
    radius_mm: float = 5.0,
    pixel_mm: float = 0.15,
    interpolation_order: int = 1,
    name: str = "centerline",
    curve_resolution_mm: float | None = None,
    centerline_smoothing_sigma_mm: float = 0.8,
) -> CPRResult:
    """Generate CPR directly from an ordered centerline, without detection."""
    if not np.isfinite(distance_from_root_mm):
        raise ValueError("distance_from_root_mm must be finite.")
    centerline, spacing, frames_u, frames_v, distances = (
        _prepare_centerline_geometry(
            centerline_voxel_zyx,
            spacing_zyx,
            frames_u_zyx,
            frames_v_zyx,
            centerline_distances_mm,
            curve_resolution_mm,
            centerline_smoothing_sigma_mm,
        )
    )

    cpr_image, offsets = sample_cpr(
        image=image,
        centers_voxel_zyx=centerline,
        frames_u_zyx=frames_u,
        frames_v_zyx=frames_v,
        spacing_zyx=spacing,
        angle_degrees=angle_degrees,
        radius_mm=radius_mm,
        pixel_mm=pixel_mm,
        interpolation_order=interpolation_order,
    )
    label = str(name) or "centerline"
    return CPRResult(
        image=cpr_image,
        centerline_distances_mm=distances,
        distances_from_root_mm=distances + float(distance_from_root_mm),
        offsets_mm=offsets,
        angle_degrees=float(angle_degrees),
        branch_id=label,
        anatomical_label=label,
        point_anatomical_labels=tuple(label for _ in centerline),
        sampling_line_length_mm=2.0 * float(radius_mm),
    )


def generate_stretched_cpr(
    image: np.ndarray,
    centerline_voxel_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    angle_degrees: float,
    sampling_line_length_mm: float = 40.0,
    slice_resolution_mm: float = 1.0,
    frames_u_zyx: np.ndarray | None = None,
    frames_v_zyx: np.ndarray | None = None,
    centerline_distances_mm: np.ndarray | None = None,
    distance_from_root_mm: float = 0.0,
    interpolation_order: int = 1,
    name: str = "centerline",
    curve_resolution_mm: float | None = None,
    centerline_smoothing_sigma_mm: float = 0.8,
    vector_of_interest_zyx: np.ndarray | None = None,
    image_direction_xyz: Sequence[float] | np.ndarray | None = None,
) -> CPRResult:
    """Generate stretched CPR using one global patient-LPS direction."""
    if not np.isfinite(distance_from_root_mm):
        raise ValueError("distance_from_root_mm must be finite.")
    if slice_resolution_mm <= 0:
        raise ValueError("slice_resolution_mm must be positive.")
    if (
        not np.isfinite(sampling_line_length_mm)
        or sampling_line_length_mm <= 0
    ):
        raise ValueError("sampling_line_length_mm must be positive and finite.")
    if interpolation_order < 0 or interpolation_order > 5:
        raise ValueError("interpolation_order must be between 0 and 5.")

    centerline, spacing, frames_u, frames_v, distances = (
        _prepare_centerline_geometry(
            centerline_voxel_zyx,
            spacing_zyx,
            frames_u_zyx,
            frames_v_zyx,
            centerline_distances_mm,
            curve_resolution_mm,
            centerline_smoothing_sigma_mm,
        )
    )
    if vector_of_interest_zyx is None:
        line_direction, line_direction_lps = global_vector_of_interest_zyx(
            angle_degrees,
            image_direction_xyz=image_direction_xyz,
        )
    else:
        line_direction = np.asarray(vector_of_interest_zyx, dtype=float)
        if line_direction.shape != (3,):
            raise ValueError("vector_of_interest_zyx must have shape (3,).")
    direction_norm = float(np.linalg.norm(line_direction))
    if direction_norm < 1e-8 or not np.isfinite(direction_norm):
        raise ValueError("vector_of_interest_zyx must be a finite nonzero vector.")
    line_direction = line_direction / direction_norm
    if vector_of_interest_zyx is not None:
        if image_direction_xyz is None:
            image_direction = np.eye(3, dtype=float)
        else:
            image_direction = np.asarray(
                image_direction_xyz,
                dtype=float,
            ).reshape(3, 3)
        line_direction_lps = image_direction @ line_direction[::-1]
        line_direction_lps /= np.linalg.norm(line_direction_lps)

    centerline_mm = centerline * spacing
    segments_mm = np.diff(centerline_mm, axis=0)
    parallel_steps = segments_mm @ line_direction
    segment_lengths_sq = np.sum(segments_mm * segments_mm, axis=1)
    perpendicular_steps = np.sqrt(
        np.maximum(segment_lengths_sq - parallel_steps * parallel_steps, 0.0)
    )
    centerline_x = np.concatenate(([0.0], np.cumsum(parallel_steps)))
    centerline_y = np.concatenate(([0.0], np.cumsum(perpendicular_steps)))
    if centerline_y[-1] <= 1e-8:
        raise ValueError(
            "Stretched CPR is degenerate because the centerline is parallel "
            "to the vector of interest."
        )

    radius_mm = 0.5 * float(sampling_line_length_mm)
    x_min = float(np.min(centerline_x) - radius_mm)
    x_max = float(np.max(centerline_x) + radius_mm)
    x_count = max(1, int(np.ceil((x_max - x_min) / slice_resolution_mm)))
    y_count = max(
        1,
        int(np.ceil(centerline_y[-1] / slice_resolution_mm)),
    )
    display_x = np.linspace(x_min, x_max, x_count + 1, dtype=float)
    display_y = np.linspace(0.0, centerline_y[-1], y_count + 1, dtype=float)

    keep = np.concatenate(
        ([True], np.diff(centerline_y) > 1e-8)
    )
    unique_y = centerline_y[keep]
    unique_centers_mm = centerline_mm[keep]
    unique_center_x = centerline_x[keep]
    surface_centers_mm = np.column_stack(
        [
            np.interp(display_y, unique_y, unique_centers_mm[:, axis])
            for axis in range(3)
        ]
    )
    surface_center_x = np.interp(
        display_y,
        unique_y,
        unique_center_x,
    )
    line_parameters = display_x[None, :] - surface_center_x[:, None]
    sample_points_mm = (
        surface_centers_mm[:, None, :]
        + line_parameters[:, :, None] * line_direction[None, None, :]
    )
    coordinates = np.moveaxis(
        sample_points_mm / spacing[None, None, :],
        -1,
        0,
    )
    volume = np.asarray(image)
    if volume.ndim != 3:
        raise ValueError("image must be a 3-D array in z-y-x order.")
    finite = volume[np.isfinite(volume)]
    cval = float(np.min(finite)) if finite.size else 0.0
    stretched_image = ndi.map_coordinates(
        volume,
        coordinates,
        order=int(interpolation_order),
        mode="constant",
        cval=cval,
        prefilter=interpolation_order > 1,
    )

    label = str(name) or "centerline"
    return CPRResult(
        image=stretched_image,
        centerline_distances_mm=distances,
        distances_from_root_mm=distances + float(distance_from_root_mm),
        offsets_mm=display_x,
        angle_degrees=float(angle_degrees),
        branch_id=label,
        anatomical_label=label,
        point_anatomical_labels=tuple(label for _ in centerline),
        mode="stretched",
        display_x_mm=display_x,
        display_y_mm=display_y,
        centerline_x_mm=centerline_x,
        centerline_y_mm=centerline_y,
        vector_of_interest_zyx=tuple(
            float(value) for value in line_direction
        ),
        vector_of_interest_lps_xyz=tuple(
            float(value) for value in line_direction_lps
        ),
        sampling_line_length_mm=float(sampling_line_length_mm),
    )


def generate_label_cprs(
    label: np.ndarray,
    image: np.ndarray,
    spacing_zyx: Sequence[float],
    angles_degrees: Sequence[float],
    label_values: Dict[str, int],
    aorta_mask: Optional[np.ndarray] = None,
    analyze_side_branches: bool = True,
    centerline_step_mm: float = 0.4,
    centerline_smoothing_sigma_mm: float = 0.8,
    min_branch_length_mm: float = 3.0,
    min_branch_mean_radius_mm: float = 0.4,
    max_branch_order: Optional[int] = None,
    sampling_line_length_mm: float = 40.0,
    pixel_mm: float = 0.15,
    interpolation_order: int = 1,
    mode: str = "straightened",
    image_direction_xyz: Sequence[float] | np.ndarray | None = None,
) -> list[CPRResult]:
    """Extract centerline trees from a label and generate CPR images."""
    try:
        from .anatomy import build_anatomical_plan, point_anatomical_label
        from .centerline import extract_centerline_tree
    except ImportError:
        from anatomy import build_anatomical_plan, point_anatomical_label
        from centerline import extract_centerline_tree

    label_array = np.asarray(label)
    image_array = np.asarray(image)
    if label_array.shape != image_array.shape or image_array.ndim != 3:
        raise ValueError("label and image must be aligned 3-D arrays.")
    if aorta_mask is not None and np.asarray(aorta_mask).shape != label_array.shape:
        raise ValueError("aorta_mask must have the same shape as label.")
    if max_branch_order is not None and max_branch_order < 0:
        raise ValueError("max_branch_order cannot be negative.")
    if (
        not np.isfinite(sampling_line_length_mm)
        or sampling_line_length_mm <= 0
    ):
        raise ValueError("sampling_line_length_mm must be positive and finite.")
    angles = [float(angle) for angle in angles_degrees]
    if not angles or any(not np.isfinite(angle) for angle in angles):
        raise ValueError("angles_degrees must contain finite values.")
    mode_name = str(mode).lower()
    if mode_name not in {"straightened", "stretched"}:
        raise ValueError("mode must be 'straightened' or 'stretched'.")

    results = []
    for vessel, label_value in label_values.items():
        vessel_mask = label_array == int(label_value)
        if not np.any(vessel_mask):
            continue
        tree = extract_centerline_tree(
            vessel_mask,
            spacing_zyx=spacing_zyx,
            step_mm=centerline_step_mm,
            smoothing_sigma_mm=centerline_smoothing_sigma_mm,
            min_branch_length_mm=min_branch_length_mm,
            min_branch_mean_radius_mm=min_branch_mean_radius_mm,
            aorta_mask=aorta_mask,
        )
        branches = [
            branch
            for branch in tree.branches
            if (analyze_side_branches or branch.is_main)
            and (
                max_branch_order is None
                or branch.branch_order <= max_branch_order
            )
        ]
        anatomical_labels, split_distance = build_anatomical_plan(vessel, branches)
        for branch in branches:
            branch_label = anatomical_labels.get(
                branch.branch_id,
                vessel.upper(),
            )
            point_labels = tuple(
                point_anatomical_label(
                    vessel=vessel,
                    branch=branch,
                    branch_label=branch_label,
                    distance_from_root_mm=(
                        branch.distance_from_root_mm + float(distance)
                    ),
                    main_split_distance_mm=split_distance,
                )
                for distance in branch.distances_mm
            )
            for angle in angles:
                if mode_name == "straightened":
                    cpr = generate_centerline_cpr(
                        image=image_array,
                        centerline_voxel_zyx=branch.path_voxel_zyx,
                        spacing_zyx=spacing_zyx,
                        angle_degrees=angle,
                        centerline_distances_mm=branch.distances_mm,
                        distance_from_root_mm=branch.distance_from_root_mm,
                        radius_mm=0.5 * float(sampling_line_length_mm),
                        pixel_mm=pixel_mm,
                        interpolation_order=interpolation_order,
                        name=branch_label,
                    )
                else:
                    cpr = generate_stretched_cpr(
                        image=image_array,
                        centerline_voxel_zyx=branch.path_voxel_zyx,
                        spacing_zyx=spacing_zyx,
                        angle_degrees=angle,
                        sampling_line_length_mm=sampling_line_length_mm,
                        slice_resolution_mm=pixel_mm,
                        centerline_distances_mm=branch.distances_mm,
                        distance_from_root_mm=branch.distance_from_root_mm,
                        interpolation_order=interpolation_order,
                        name=branch_label,
                        image_direction_xyz=image_direction_xyz,
                    )
                results.append(
                    replace(
                        cpr,
                        branch_id=f"{vessel}-B{branch.branch_id:03d}",
                        point_anatomical_labels=point_labels,
                    )
                )
    if not results:
        raise ValueError("No requested vessel labels produced a CPR image.")
    return results


def generate_branch_cpr(
    image: np.ndarray,
    spacing_zyx: Sequence[float],
    branch: BranchResult,
    angle_degrees: float,
    radius_mm: float = 5.0,
    pixel_mm: float = 0.15,
    interpolation_order: int = 1,
) -> CPRResult:
    """Generate a physically calibrated CPR from stored branch frames."""
    if not branch.measurements:
        raise ValueError(f"Branch {branch.branch_id} has no measurements.")

    centers = np.asarray(
        [item.center_voxel_zyx for item in branch.measurements],
        dtype=float,
    )
    if any(
        item.frame_u_zyx is None or item.frame_v_zyx is None
        for item in branch.measurements
    ):
        raise ValueError(
            f"Branch {branch.branch_id} does not contain complete CPR frames."
        )
    frames_u = np.asarray(
        [item.frame_u_zyx for item in branch.measurements],
        dtype=float,
    )
    frames_v = np.asarray(
        [item.frame_v_zyx for item in branch.measurements],
        dtype=float,
    )
    cpr_image, offsets = sample_cpr(
        image=image,
        centers_voxel_zyx=centers,
        frames_u_zyx=frames_u,
        frames_v_zyx=frames_v,
        spacing_zyx=spacing_zyx,
        angle_degrees=angle_degrees,
        radius_mm=radius_mm,
        pixel_mm=pixel_mm,
        interpolation_order=interpolation_order,
    )
    return CPRResult(
        image=cpr_image,
        centerline_distances_mm=np.asarray(
            [item.distance_mm for item in branch.measurements],
            dtype=float,
        ),
        distances_from_root_mm=np.asarray(
            [item.distance_from_root_mm for item in branch.measurements],
            dtype=float,
        ),
        offsets_mm=offsets,
        angle_degrees=float(angle_degrees),
        branch_id=branch.branch_id,
        anatomical_label=branch.anatomical_label,
        point_anatomical_labels=tuple(
            item.anatomical_label for item in branch.measurements
        ),
    )
