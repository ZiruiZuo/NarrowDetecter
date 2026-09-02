from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    from .anatomy import (
        build_anatomical_plan as _build_anatomical_plan,
        point_anatomical_label as _point_anatomical_label,
    )
    from .centerline import (
        CenterlineBranch,
        extract_centerline_tree,
        tangents_and_frames,
    )
    from .cross_section import (
        make_sampling_grid,
        sample_plane,
        segment_cross_section,
    )
    from .models import (
        BranchResult,
        CrossSectionMeasurement,
        DetectionResult,
        DetectorConfig,
        StenosisCandidate,
        VesselResult,
    )
except ImportError:
    from anatomy import (
        build_anatomical_plan as _build_anatomical_plan,
        point_anatomical_label as _point_anatomical_label,
    )
    from centerline import (
        CenterlineBranch,
        extract_centerline_tree,
        tangents_and_frames,
    )
    from cross_section import (
        make_sampling_grid,
        sample_plane,
        segment_cross_section,
    )
    from models import (
        BranchResult,
        CrossSectionMeasurement,
        DetectionResult,
        DetectorConfig,
        StenosisCandidate,
        VesselResult,
    )


def _validate_inputs(
    label: np.ndarray,
    image: np.ndarray,
    spacing_zyx: Sequence[float],
):
    label = np.asarray(label)
    image = np.asarray(image, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if label.ndim != 3 or image.ndim != 3:
        raise ValueError(
            "label and image must both be 3-D arrays in [z, y, x] order."
        )
    if label.shape != image.shape:
        raise ValueError(
            f"Shape mismatch: label={label.shape}, image={image.shape}."
        )
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(
            "spacing_zyx must contain three positive values."
        )
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains NaN or infinite values.")
    return label, image, spacing


def _gradient_magnitude(
    image: np.ndarray,
    spacing: np.ndarray,
    sigma_mm: float,
):
    sigma_voxel = sigma_mm / spacing
    smoothed = ndi.gaussian_filter(
        image, sigma=sigma_voxel, mode="nearest"
    )
    derivatives = np.gradient(smoothed, *spacing)
    gradient = np.sqrt(
        sum(component * component for component in derivatives)
    )
    return smoothed.astype(np.float32), gradient.astype(np.float32)


def _distance_to_points_mm(
    path_voxel: np.ndarray,
    points_voxel: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    if len(points_voxel) == 0:
        return np.full(len(path_voxel), np.inf)
    path_mm = path_voxel * spacing
    points_mm = points_voxel * spacing
    distances = np.linalg.norm(
        path_mm[:, None, :] - points_mm[None, :, :], axis=2
    )
    return np.min(distances, axis=1)


def _smooth_valid_profile(
    values: np.ndarray,
    quality: np.ndarray,
    sigma_samples: float,
):
    valid = np.isfinite(values) & (values > 0) & (quality > 0)
    if np.count_nonzero(valid) < 2:
        return values.copy()
    indices = np.arange(len(values))
    filled = np.interp(indices, indices[valid], values[valid])
    if sigma_samples > 0:
        filled = ndi.gaussian_filter1d(
            filled, sigma=sigma_samples, mode="nearest"
        )
    return filled


def _estimate_reference(
    diameters: np.ndarray,
    distances: np.ndarray,
    usable: np.ndarray,
    config: DetectorConfig,
) -> np.ndarray:
    reference = np.full_like(diameters, np.nan, dtype=float)
    half_window = config.reference_window_mm / 2.0
    for index, distance in enumerate(distances):
        delta = np.abs(distances - distance)
        selection = (
            usable
            & (delta <= half_window)
            & (delta >= config.reference_exclusion_mm)
            & np.isfinite(diameters)
            & (diameters > 0)
        )
        values = diameters[selection]
        if values.size < 3:
            selection = (
                usable
                & (delta <= half_window)
                & np.isfinite(diameters)
                & (diameters > 0)
            )
            values = diameters[selection]
        if values.size:
            reference[index] = float(
                np.percentile(values, config.reference_percentile)
            )

    valid = np.isfinite(reference)
    if np.count_nonzero(valid) >= 2:
        reference = np.interp(
            np.arange(len(reference)),
            np.flatnonzero(valid),
            reference[valid],
        )
        sigma = max(
            config.profile_smoothing_mm
            / config.centerline_step_mm,
            0.0,
        )
        reference = ndi.gaussian_filter1d(
            reference, sigma=sigma, mode="nearest"
        )
    return reference


def _grade(stenosis: float) -> str:
    percent = stenosis * 100.0
    if percent < 25:
        return "minimal"
    if percent < 50:
        return "mild"
    if percent < 70:
        return "moderate"
    if percent < 100:
        return "severe"
    return "occlusion_suspected"


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_short_gaps(
    mask: np.ndarray,
    distances: np.ndarray,
    max_gap_mm: float,
) -> np.ndarray:
    result = mask.copy()
    for start, end in _runs(~mask):
        if start == 0 or end == len(mask) - 1:
            continue
        gap = (
            distances[end] - distances[start]
            if end > start
            else 0.0
        )
        if gap <= max_gap_mm:
            result[start : end + 1] = True
    return result


def _make_candidates(
    vessel: str,
    measurements: List[CrossSectionMeasurement],
    config: DetectorConfig,
    branch_id: str,
    parent_branch_id: Optional[str],
    branch_order: int,
    is_main: bool,
) -> List[StenosisCandidate]:
    distances = np.asarray(
        [measurement.distance_mm for measurement in measurements]
    )
    stenoses = np.asarray(
        [
            np.nan
            if measurement.diameter_stenosis is None
            else measurement.diameter_stenosis
            for measurement in measurements
        ]
    )
    quality = np.asarray(
        [measurement.quality for measurement in measurements]
    )
    candidate_mask = (
        np.isfinite(stenoses)
        & (stenoses >= config.candidate_threshold)
        & (quality > 0.2)
    )
    candidate_mask = _merge_short_gaps(
        candidate_mask, distances, config.merge_gap_mm
    )

    candidates = []
    for start, end in _runs(candidate_mask):
        length = float(
            distances[end]
            - distances[start]
            + config.centerline_step_mm
        )
        if length < config.min_stenosis_length_mm:
            continue
        local_stenoses = stenoses[start : end + 1]
        minimum_index = start + int(np.nanargmax(local_stenoses))
        minimum = measurements[minimum_index]
        segment_quality = quality[start : end + 1]
        flags = sorted(
            {
                flag
                for measurement in measurements[start : end + 1]
                for flag in measurement.flags
            }
        )
        candidates.append(
            StenosisCandidate(
                vessel=vessel,
                start_distance_mm=float(distances[start]),
                end_distance_mm=float(distances[end]),
                length_mm=length,
                min_distance_mm=minimum.distance_mm,
                min_center_voxel_zyx=minimum.center_voxel_zyx,
                min_diameter_mm=minimum.equivalent_diameter_mm,
                reference_diameter_mm=float(
                    minimum.reference_diameter_mm or np.nan
                ),
                max_diameter_stenosis=float(
                    minimum.diameter_stenosis or 0.0
                ),
                max_area_stenosis=minimum.area_stenosis,
                grade=_grade(
                    float(minimum.diameter_stenosis or 0.0)
                ),
                confidence=float(
                    np.clip(np.mean(segment_quality), 0.0, 1.0)
                ),
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                branch_order=branch_order,
                is_main=is_main,
                start_distance_from_root_mm=measurements[
                    start
                ].distance_from_root_mm,
                end_distance_from_root_mm=measurements[
                    end
                ].distance_from_root_mm,
                min_distance_from_root_mm=minimum.distance_from_root_mm,
                anatomical_label=minimum.anatomical_label,
                flags=flags,
            )
        )
    return candidates


def _measure_branch(
    vessel: str,
    tree_branch: CenterlineBranch,
    parent_branch_id: Optional[str],
    mask: np.ndarray,
    image: np.ndarray,
    gradient: np.ndarray,
    allowed_volume: np.ndarray,
    junction_voxels: np.ndarray,
    spacing: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    radial_distance: np.ndarray,
    config: DetectorConfig,
    anatomical_label: str,
    main_split_distance_mm: Optional[float],
) -> BranchResult:
    branch_id = f"{vessel}-B{tree_branch.branch_id:03d}"
    branch_result = BranchResult(
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        branch_order=tree_branch.branch_order,
        is_main=tree_branch.is_main,
        length_mm=tree_branch.length_mm,
        mean_radius_mm=tree_branch.mean_radius_mm,
        distance_from_root_mm=tree_branch.distance_from_root_mm,
        anatomical_label=anatomical_label,
    )

    path = tree_branch.path_voxel_zyx
    distances = tree_branch.distances_mm
    tangents, frames_u, frames_v = tangents_and_frames(path, spacing)
    junction_distance = _distance_to_points_mm(
        path, junction_voxels, spacing
    )
    background_value = float(np.min(image))
    maximum_gradient = float(np.max(gradient))
    measurements = []

    for index, center in enumerate(path):
        image_plane = sample_plane(
            image,
            center,
            frames_u[index],
            frames_v[index],
            spacing,
            grid_u,
            grid_v,
            order=1,
            cval=background_value,
        )
        gradient_plane = sample_plane(
            gradient,
            center,
            frames_u[index],
            frames_v[index],
            spacing,
            grid_u,
            grid_v,
            order=1,
            cval=maximum_gradient,
        )
        prior_plane = (
            sample_plane(
                mask.astype(np.float32),
                center,
                frames_u[index],
                frames_v[index],
                spacing,
                grid_u,
                grid_v,
                order=0,
                cval=0.0,
            )
            > 0.5
        )
        allowed_plane = (
            sample_plane(
                allowed_volume.astype(np.float32),
                center,
                frames_u[index],
                frames_v[index],
                spacing,
                grid_u,
                grid_v,
                order=0,
                cval=0.0,
            )
            > 0.5
        )
        allowed_plane &= (
            radial_distance <= config.cross_section_radius_mm
        )
        _, area, prior_area, quality, flags = segment_cross_section(
            image_plane,
            gradient_plane,
            prior_plane,
            allowed_plane,
            radial_distance,
            config,
        )

        near_start_endpoint = (
            tree_branch.starts_at_endpoint
            and distances[index] < config.endpoint_exclusion_mm
        )
        near_end_endpoint = (
            tree_branch.ends_at_endpoint
            and distances[-1] - distances[index]
            < config.endpoint_exclusion_mm
        )
        if near_start_endpoint or near_end_endpoint:
            flags.append("near_endpoint")
            quality *= 0.5
        if (
            junction_distance[index]
            < config.bifurcation_exclusion_mm
        ):
            flags.append("near_bifurcation")
            quality *= 0.5

        diameter = float(
            2.0 * np.sqrt(max(area, 0.0) / np.pi)
        )
        distance_from_root = float(
            tree_branch.distance_from_root_mm + distances[index]
        )
        measurements.append(
            CrossSectionMeasurement(
                index=index,
                distance_mm=float(distances[index]),
                center_voxel_zyx=tuple(float(value) for value in center),
                tangent_zyx=tuple(
                    float(value) for value in tangents[index]
                ),
                area_mm2=area,
                equivalent_diameter_mm=diameter,
                prior_area_mm2=prior_area,
                branch_id=branch_id,
                branch_index=tree_branch.branch_id,
                distance_from_root_mm=distance_from_root,
                anatomical_label=_point_anatomical_label(
                    vessel=vessel,
                    branch=tree_branch,
                    branch_label=anatomical_label,
                    distance_from_root_mm=distance_from_root,
                    main_split_distance_mm=main_split_distance_mm,
                ),
                frame_u_zyx=tuple(
                    float(value) for value in frames_u[index]
                ),
                frame_v_zyx=tuple(
                    float(value) for value in frames_v[index]
                ),
                quality=float(quality),
                flags=flags,
            )
        )

    diameters = np.asarray(
        [
            measurement.equivalent_diameter_mm
            for measurement in measurements
        ]
    )
    qualities = np.asarray(
        [measurement.quality for measurement in measurements]
    )
    smoothed = _smooth_valid_profile(
        diameters,
        qualities,
        config.profile_smoothing_mm
        / config.centerline_step_mm,
    )
    usable = np.asarray(
        [
            measurement.quality > 0.2
            and "near_endpoint" not in measurement.flags
            and "near_bifurcation" not in measurement.flags
            for measurement in measurements
        ]
    )
    if np.count_nonzero(usable) < 3:
        usable = qualities > 0.2
        branch_result.warnings.append(
            "Reference estimation included endpoint or bifurcation sections "
            "because fewer than three unaffected sections were available."
        )
    reference = _estimate_reference(
        smoothed, distances, usable, config
    )
    for index, measurement in enumerate(measurements):
        measurement.equivalent_diameter_mm = float(smoothed[index])
        if np.isfinite(reference[index]) and reference[index] > 0:
            measurement.reference_diameter_mm = float(reference[index])
            measurement.diameter_stenosis = float(
                np.clip(
                    1.0 - smoothed[index] / reference[index],
                    0.0,
                    1.0,
                )
            )
            measurement.area_stenosis = float(
                np.clip(
                    1.0
                    - (smoothed[index] / reference[index]) ** 2,
                    0.0,
                    1.0,
                )
            )

    branch_result.measurements = measurements
    branch_result.candidates = _make_candidates(
        vessel=vessel,
        measurements=measurements,
        config=config,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        branch_order=tree_branch.branch_order,
        is_main=tree_branch.is_main,
    )
    fallback_fraction = np.mean(
        [
            "used_label_fallback" in measurement.flags
            for measurement in measurements
        ]
    )
    if fallback_fraction > 0.2:
        branch_result.warnings.append(
            "More than 20% of sections used the label fallback."
        )
    return branch_result


def _process_vessel(
    vessel: str,
    label_value: int,
    mask: np.ndarray,
    image: np.ndarray,
    gradient: np.ndarray,
    spacing: np.ndarray,
    config: DetectorConfig,
    aorta_mask: Optional[np.ndarray] = None,
) -> VesselResult:
    result = VesselResult(
        vessel=vessel,
        label_value=label_value,
        status="ok",
    )
    try:
        tree = extract_centerline_tree(
            mask=mask,
            spacing_zyx=spacing,
            step_mm=config.centerline_step_mm,
            smoothing_sigma_mm=(
                config.centerline_smoothing_sigma_mm
            ),
            min_branch_length_mm=config.min_branch_length_mm,
            min_branch_mean_radius_mm=(
                config.min_branch_mean_radius_mm
            ),
            aorta_mask=aorta_mask,
        )
    except ValueError as error:
        result.status = "failed"
        result.warnings.append(str(error))
        return result

    selected_branches = []
    anatomical_labels, main_split_distance = _build_anatomical_plan(
        vessel, tree.branches
    )
    for branch in tree.branches:
        if not config.analyze_side_branches and not branch.is_main:
            continue
        if (
            config.max_branch_order is not None
            and branch.branch_order > config.max_branch_order
        ):
            continue
        selected_branches.append(branch)

    grid_u, grid_v, radial_distance = make_sampling_grid(
        config.cross_section_radius_mm,
        config.cross_section_pixel_mm,
    )
    allowed_volume = (
        ndi.distance_transform_edt(~mask, sampling=spacing)
        <= config.mask_dilation_mm
    )
    available_ids = {
        branch.branch_id: f"{vessel}-B{branch.branch_id:03d}"
        for branch in selected_branches
    }
    branch_results = []
    for branch in selected_branches:
        parent_branch_id = available_ids.get(
            branch.parent_branch_id
        )
        branch_results.append(
            _measure_branch(
                vessel=vessel,
                tree_branch=branch,
                parent_branch_id=parent_branch_id,
                mask=mask,
                image=image,
                gradient=gradient,
                allowed_volume=allowed_volume,
                junction_voxels=tree.junction_voxels_zyx,
                spacing=spacing,
                grid_u=grid_u,
                grid_v=grid_v,
                radial_distance=radial_distance,
                config=config,
                anatomical_label=anatomical_labels.get(
                    branch.branch_id,
                    available_ids[branch.branch_id],
                ),
                main_split_distance_mm=main_split_distance,
            )
        )

    result.centerline_length_mm = float(
        tree.main_distances_mm[-1]
    )
    result.branches = branch_results
    result.candidates = [
        candidate
        for branch_result in branch_results
        for candidate in branch_result.candidates
    ]
    main_result = next(
        (
            branch_result
            for branch_result in branch_results
            if branch_result.is_main
        ),
        None,
    )
    if main_result is not None:
        result.measurements = main_result.measurements
    else:
        result.status = "failed"
        result.warnings.append("No main branch was available for analysis.")
    for branch_result in branch_results:
        for warning in branch_result.warnings:
            result.warnings.append(
                f"{branch_result.branch_id}: {warning}"
            )
    return result


def detect_coronary_stenosis(
    label: np.ndarray,
    image: np.ndarray,
    spacing_zyx: Sequence[float],
    config: Optional[DetectorConfig] = None,
    aorta_mask: Optional[np.ndarray] = None,
) -> DetectionResult:
    """Screen coronary main and side branches for focal lumen narrowing."""
    config = config or DetectorConfig()
    config.validate()
    label, image, spacing = _validate_inputs(
        label, image, spacing_zyx
    )
    if aorta_mask is not None:
        aorta_mask = np.asarray(aorta_mask, dtype=bool)
        if aorta_mask.shape != label.shape:
            raise ValueError(
                "aorta_mask must have the same shape as label and image."
            )
    warnings = []
    if (
        float(np.min(image)) < -0.05
        or float(np.max(image)) > 1.05
    ):
        warnings.append(
            "Image values are outside the expected normalized [0, 1] range."
        )
    prepared_image, gradient = _gradient_magnitude(
        image, spacing, config.image_smoothing_sigma_mm
    )

    vessels: Dict[str, VesselResult] = {}
    for vessel, label_value in config.label_values.items():
        mask = label == label_value
        if not np.any(mask):
            vessels[vessel] = VesselResult(
                vessel=vessel,
                label_value=label_value,
                status="missing",
                warnings=[f"Label value {label_value} is absent."],
            )
            continue
        vessels[vessel] = _process_vessel(
            vessel,
            label_value,
            mask,
            prepared_image,
            gradient,
            spacing,
            config,
            aorta_mask,
        )
    return DetectionResult(
        spacing_zyx=tuple(float(value) for value in spacing),
        vessels=vessels,
        warnings=warnings,
    )
