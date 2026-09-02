from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage as ndi

from ..cpr import CPRResult, generate_branch_cpr
from ..detection.cross_section import (
    make_sampling_grid,
    sample_plane,
    segment_cross_section,
)
from ..models import DetectionResult, DetectorConfig, StenosisCandidate

from .volumes import (
    build_anatomical_centerline_volume,
    build_stenosis_marker_volume,
)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required to create PNG visualizations.") from error
    return plt


def _shade_candidates(axes, candidates) -> None:
    for candidate in candidates:
        axes.axvspan(
            candidate.start_distance_mm,
            candidate.end_distance_mm,
            color="#d95f02",
            alpha=0.14,
            linewidth=0,
        )


def _anatomical_filename_stem(
    anatomical_label: str | None,
    fallback: str,
) -> str:
    """Convert an anatomical label into a filesystem-safe filename stem."""
    source = anatomical_label or fallback
    stem = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")
    return stem or "unlabeled"


def _unique_output_path(
    output: Path,
    stem: str,
    suffix: str,
    used_names: set[str],
) -> Path:
    """Avoid collisions without exposing the internal branch identifier."""
    candidate_name = f"{stem}{suffix}"
    sequence = 2
    while candidate_name.lower() in used_names:
        candidate_name = f"{stem}_{sequence}{suffix}"
        sequence += 1
    used_names.add(candidate_name.lower())
    return output / candidate_name


def plot_diameter_profiles(
    result: DetectionResult,
    output_dir: str,
    dpi: int = 160,
) -> List[Path]:
    """Save a diameter and stenosis curve for each available vessel."""
    plt = _pyplot()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    used_names: set[str] = set()

    for vessel_name, vessel in result.vessels.items():
        if vessel.branches:
            branch_items = [
                (
                    branch.branch_id,
                    branch.anatomical_label,
                    branch.is_main,
                    branch.measurements,
                    branch.candidates,
                )
                for branch in vessel.branches
            ]
        else:
            branch_items = [
                (
                    vessel_name,
                    vessel_name,
                    True,
                    vessel.measurements,
                    vessel.candidates,
                )
            ]
        for (
            branch_id,
            anatomical_label,
            is_main,
            measurements,
            candidates,
        ) in branch_items:
            if not measurements:
                continue
            distance = np.asarray(
                [item.distance_mm for item in measurements]
            )
            diameter = np.asarray(
                [item.equivalent_diameter_mm for item in measurements]
            )
            reference = np.asarray(
                [
                    np.nan
                    if item.reference_diameter_mm is None
                    else item.reference_diameter_mm
                    for item in measurements
                ]
            )
            stenosis = 100.0 * np.asarray(
                [
                    np.nan
                    if item.diameter_stenosis is None
                    else item.diameter_stenosis
                    for item in measurements
                ]
            )
            quality = np.asarray(
                [item.quality for item in measurements]
            )

            figure, axes = plt.subplots(
                2,
                1,
                figsize=(10.0, 6.2),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 2]},
                constrained_layout=True,
            )
            _shade_candidates(axes[0], candidates)
            _shade_candidates(axes[1], candidates)
            axes[0].plot(
                distance,
                diameter,
                color="#0072b2",
                linewidth=1.8,
                label="Measured",
            )
            axes[0].plot(
                distance,
                reference,
                color="#333333",
                linewidth=1.5,
                linestyle="--",
                label="Local reference",
            )
            low_quality = quality <= 0.2
            if np.any(low_quality):
                axes[0].scatter(
                    distance[low_quality],
                    diameter[low_quality],
                    color="#777777",
                    marker="x",
                    s=18,
                    label="Low quality",
                    zorder=3,
                )
            axes[0].set_ylabel("Equivalent diameter (mm)")
            branch_kind = "main" if is_main else "side"
            axes[0].set_title(
                f"{anatomical_label} ({branch_kind}) "
                "diameter profile"
            )
            point_labels = [
                item.anatomical_label for item in measurements
            ]
            for index in range(1, len(point_labels)):
                if point_labels[index] == point_labels[index - 1]:
                    continue
                transition = distance[index]
                axes[0].axvline(
                    transition,
                    color="#9467bd",
                    linewidth=0.9,
                    linestyle=":",
                )
                axes[1].axvline(
                    transition,
                    color="#9467bd",
                    linewidth=0.9,
                    linestyle=":",
                )
                axes[0].text(
                    transition,
                    axes[0].get_ylim()[1],
                    point_labels[index],
                    ha="left",
                    va="top",
                    color="#6a3d9a",
                )
            axes[0].legend(loc="best", frameon=False)
            axes[0].grid(
                axis="y",
                color="#bdbdbd",
                alpha=0.35,
                linewidth=0.6,
            )

            axes[1].plot(
                distance,
                stenosis,
                color="#009e73",
                linewidth=1.7,
            )
            for threshold, threshold_label in (
                (25, "25%"),
                (50, "50%"),
                (70, "70%"),
            ):
                axes[1].axhline(
                    threshold,
                    color="#777777",
                    linewidth=0.7,
                    linestyle=":",
                )
                axes[1].text(
                    distance[-1],
                    threshold,
                    threshold_label,
                    ha="right",
                    va="bottom",
                    color="#555555",
                )
            axes[1].set_ylim(0, 100)
            axes[1].set_ylabel("Diameter stenosis (%)")
            axes[1].set_xlabel("Branch distance (mm)")
            axes[1].grid(
                axis="y",
                color="#bdbdbd",
                alpha=0.35,
                linewidth=0.6,
            )

            filename_stem = _anatomical_filename_stem(
                anatomical_label,
                branch_id,
            )
            path = _unique_output_path(
                output,
                filename_stem,
                "_diameter_profile.png",
                used_names,
            )
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def plot_intensity_profiles(
    result: DetectionResult,
    output_dir: str,
    dpi: int = 160,
) -> List[Path]:
    """Save lumen mean-intensity and local-reference curves."""
    plt = _pyplot()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    used_names: set[str] = set()

    for vessel_name, vessel in result.vessels.items():
        if vessel.branches:
            branch_items = [
                (
                    branch.branch_id,
                    branch.anatomical_label,
                    branch.is_main,
                    branch.measurements,
                    branch.candidates,
                )
                for branch in vessel.branches
            ]
        else:
            branch_items = [
                (
                    vessel_name,
                    vessel_name,
                    True,
                    vessel.measurements,
                    vessel.candidates,
                )
            ]
        for (
            branch_id,
            anatomical_label,
            is_main,
            measurements,
            candidates,
        ) in branch_items:
            if not measurements:
                continue
            distance = np.asarray(
                [item.distance_mm for item in measurements],
                dtype=float,
            )
            mean_intensity = np.asarray(
                [
                    np.nan
                    if item.mean_lumen_intensity is None
                    else item.mean_lumen_intensity
                    for item in measurements
                ],
                dtype=float,
            )
            reference = np.asarray(
                [
                    np.nan
                    if item.reference_mean_intensity is None
                    else item.reference_mean_intensity
                    for item in measurements
                ],
                dtype=float,
            )
            change_percent = 100.0 * np.asarray(
                [
                    np.nan
                    if item.intensity_change_ratio is None
                    else item.intensity_change_ratio
                    for item in measurements
                ],
                dtype=float,
            )
            quality = np.asarray(
                [item.quality for item in measurements],
                dtype=float,
            )

            if not np.any(np.isfinite(mean_intensity)):
                continue

            figure, axes = plt.subplots(
                2,
                1,
                figsize=(10.0, 6.2),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 2]},
                constrained_layout=True,
            )
            _shade_candidates(axes[0], candidates)
            _shade_candidates(axes[1], candidates)
            axes[0].plot(
                distance,
                mean_intensity,
                color="#d55e00",
                linewidth=1.8,
                label="Mean lumen intensity",
            )
            axes[0].plot(
                distance,
                reference,
                color="#333333",
                linewidth=1.5,
                linestyle="--",
                label="Local median reference",
            )
            low_quality = quality <= 0.2
            if np.any(low_quality):
                axes[0].scatter(
                    distance[low_quality],
                    mean_intensity[low_quality],
                    color="#777777",
                    marker="x",
                    s=18,
                    label="Low quality",
                    zorder=3,
                )
            branch_kind = "main" if is_main else "side"
            axes[0].set_ylabel("Normalized intensity")
            axes[0].set_title(
                f"{anatomical_label} ({branch_kind}) intensity profile"
            )
            axes[0].legend(loc="best", frameon=False)
            axes[0].grid(
                axis="y",
                color="#bdbdbd",
                alpha=0.35,
                linewidth=0.6,
            )

            axes[1].plot(
                distance,
                change_percent,
                color="#009e73",
                linewidth=1.7,
            )
            axes[1].axhline(
                0.0,
                color="#777777",
                linewidth=0.8,
                linestyle=":",
            )
            axes[1].set_ylabel("Signal change (%)")
            axes[1].set_xlabel("Branch distance (mm)")
            axes[1].grid(
                axis="y",
                color="#bdbdbd",
                alpha=0.35,
                linewidth=0.6,
            )

            point_labels = [
                item.anatomical_label for item in measurements
            ]
            for index in range(1, len(point_labels)):
                if point_labels[index] == point_labels[index - 1]:
                    continue
                transition = distance[index]
                for axis in axes:
                    axis.axvline(
                        transition,
                        color="#9467bd",
                        linewidth=0.9,
                        linestyle=":",
                    )
                axes[0].text(
                    transition,
                    axes[0].get_ylim()[1],
                    point_labels[index],
                    ha="left",
                    va="top",
                    color="#6a3d9a",
                )

            filename_stem = _anatomical_filename_stem(
                anatomical_label,
                branch_id,
            )
            path = _unique_output_path(
                output,
                filename_stem,
                "_intensity_profile.png",
                used_names,
            )
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def _perpendicular_frame(tangent_zyx: Sequence[float]):
    tangent = np.asarray(tangent_zyx, dtype=float)
    tangent /= max(np.linalg.norm(tangent), 1e-8)
    axes = np.eye(3)
    helper = axes[int(np.argmin(np.abs(axes @ tangent)))]
    frame_u = np.cross(tangent, helper)
    frame_u /= max(np.linalg.norm(frame_u), 1e-8)
    frame_v = np.cross(tangent, frame_u)
    frame_v /= max(np.linalg.norm(frame_v), 1e-8)
    return frame_u, frame_v


def plot_candidate_cross_sections(
    result: DetectionResult,
    image: np.ndarray,
    label: np.ndarray,
    output_dir: str,
    radius_mm: float = 5.0,
    pixel_mm: float = 0.15,
    config: DetectorConfig | None = None,
    dpi: int = 160,
) -> List[Path]:
    """Save a centerline-orthogonal MPR at each candidate's narrowest point."""
    if radius_mm <= 0 or pixel_mm <= 0:
        raise ValueError("Cross-section radius and pixel size must be positive.")
    plt = _pyplot()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    used_names: set[str] = set()
    spacing = np.asarray(result.spacing_zyx, dtype=float)
    config = config or DetectorConfig(
        cross_section_radius_mm=radius_mm,
        cross_section_pixel_mm=pixel_mm,
    )
    grid_u, grid_v, radial_distance = make_sampling_grid(radius_mm, pixel_mm)
    extent = [
        float(grid_v.min()),
        float(grid_v.max()),
        float(grid_u.min()),
        float(grid_u.max()),
    ]
    sigma_voxel = config.image_smoothing_sigma_mm / spacing
    prepared_image = ndi.gaussian_filter(image, sigma=sigma_voxel, mode="nearest")
    derivatives = np.gradient(prepared_image, *spacing)
    gradient = np.sqrt(sum(component * component for component in derivatives))

    for vessel_name, vessel in result.vessels.items():
        vessel_mask_3d = label == vessel.label_value
        allowed_volume = (
            ndi.distance_transform_edt(~vessel_mask_3d, sampling=spacing)
            <= config.mask_dilation_mm
        )
        for candidate_index, candidate in enumerate(vessel.candidates, start=1):
            branch_result = next(
                (
                    branch
                    for branch in vessel.branches
                    if branch.branch_id == candidate.branch_id
                ),
                None,
            )
            branch_measurements = (
                branch_result.measurements
                if branch_result is not None
                else vessel.measurements
            )
            if not branch_measurements:
                continue
            measurement = min(
                branch_measurements,
                key=lambda item: abs(
                    item.distance_mm - candidate.min_distance_mm
                ),
            )
            center = np.asarray(measurement.center_voxel_zyx, dtype=float)
            if measurement.frame_u_zyx is not None and measurement.frame_v_zyx is not None:
                frame_u = np.asarray(measurement.frame_u_zyx, dtype=float)
                frame_v = np.asarray(measurement.frame_v_zyx, dtype=float)
            else:
                frame_u, frame_v = _perpendicular_frame(measurement.tangent_zyx)
            image_plane = sample_plane(
                prepared_image,
                center,
                frame_u,
                frame_v,
                spacing,
                grid_u,
                grid_v,
                order=1,
                cval=float(np.min(image)),
            )
            gradient_plane = sample_plane(
                gradient,
                center,
                frame_u,
                frame_v,
                spacing,
                grid_u,
                grid_v,
                order=1,
                cval=float(np.max(gradient)),
            )
            label_plane = sample_plane(
                vessel_mask_3d.astype(np.float32),
                center,
                frame_u,
                frame_v,
                spacing,
                grid_u,
                grid_v,
                order=0,
                cval=0.0,
            ) > 0.5
            allowed_plane = sample_plane(
                allowed_volume.astype(np.float32),
                center,
                frame_u,
                frame_v,
                spacing,
                grid_u,
                grid_v,
                order=0,
                cval=0.0,
            ) > 0.5
            allowed_plane &= radial_distance <= radius_mm
            segmented_plane, _, _, _, segmentation_flags = segment_cross_section(
                image_plane,
                gradient_plane,
                label_plane,
                allowed_plane,
                radial_distance,
                config,
            )
            finite_values = image_plane[np.isfinite(image_plane)]
            low, high = (
                np.percentile(finite_values, (2, 98))
                if finite_values.size
                else (0.0, 1.0)
            )
            if high <= low:
                high = low + 1.0

            figure, axes = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
            axes.imshow(
                image_plane,
                cmap="gray",
                origin="lower",
                extent=extent,
                vmin=low,
                vmax=high,
                interpolation="bilinear",
            )
            if np.any(label_plane):
                axes.contour(
                    grid_v,
                    grid_u,
                    label_plane.astype(float),
                    levels=[0.5],
                    colors=["#00bcd4"],
                    linewidths=1.1,
                )
            if np.any(segmented_plane):
                axes.contour(
                    grid_v,
                    grid_u,
                    segmented_plane.astype(float),
                    levels=[0.5],
                    colors=["#cc79a7"],
                    linewidths=1.8,
                )
            measured_circle = plt.Circle(
                (0, 0),
                candidate.min_diameter_mm / 2.0,
                fill=False,
                color="#f0e442",
                linewidth=1.5,
                label="Measured equivalent diameter",
            )
            axes.add_patch(measured_circle)
            if np.isfinite(candidate.reference_diameter_mm):
                reference_circle = plt.Circle(
                    (0, 0),
                    candidate.reference_diameter_mm / 2.0,
                    fill=False,
                    color="#d55e00",
                    linewidth=1.4,
                    linestyle="--",
                    label="Local reference diameter",
                )
                axes.add_patch(reference_circle)
            axes.plot(0, 0, marker="+", color="#ffffff", markersize=11, markeredgewidth=1.5)
            percent = 100.0 * candidate.max_diameter_stenosis
            axes.set_title(
                f"{candidate.anatomical_label} | "
                f"{vessel_name} candidate {candidate_index}: "
                f"{percent:.1f}% at {candidate.min_distance_mm:.1f} mm"
            )
            axes.set_xlabel("Cross-section v (mm)")
            axes.set_ylabel("Cross-section u (mm)")
            axes.set_xlim(-radius_mm, radius_mm)
            axes.set_ylim(-radius_mm, radius_mm)
            axes.set_aspect("equal")
            axes.plot([], [], color="#00bcd4", linewidth=1.1, label="Input label contour")
            segmentation_label = (
                "Segmented lumen (label fallback)"
                if "used_label_fallback" in segmentation_flags
                else "Region-grown lumen"
            )
            axes.plot([], [], color="#cc79a7", linewidth=1.8, label=segmentation_label)
            axes.legend(loc="upper right", frameon=True)

            filename_stem = _anatomical_filename_stem(
                candidate.anatomical_label,
                vessel_name,
            )
            path = _unique_output_path(
                output,
                filename_stem,
                f"_candidate_{candidate_index:02d}_cross_section.png",
                used_names,
            )
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def plot_candidate_overlays(
    result: DetectionResult,
    image: np.ndarray,
    label: np.ndarray,
    marker: np.ndarray,
    output_dir: str,
    dpi: int = 160,
) -> List[Path]:
    """Backward-compatible wrapper that now creates orthogonal MPR images."""
    del marker
    return plot_candidate_cross_sections(result, image, label, output_dir, dpi=dpi)


def write_detection_visualizations(
    result: DetectionResult,
    image: np.ndarray,
    label: np.ndarray,
    reference_itk,
    output_dir: str,
    marker_radius_mm: float = 1.5,
    centerline_label_radius_mm: float = 0.6,
    cross_section_radius_mm: float = 5.0,
    cross_section_pixel_mm: float = 0.15,
    config: DetectorConfig | None = None,
) -> Dict[str, object]:
    """Write stenosis, centerline, profile, and cross-section outputs."""
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError("SimpleITK is required to write the marker NIfTI.") from error

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker = build_stenosis_marker_volume(result, label.shape, marker_radius_mm)
    marker_path = output / "stenosis_markers.nii.gz"
    marker_itk = sitk.GetImageFromArray(marker)
    marker_itk.CopyInformation(reference_itk)
    sitk.WriteImage(marker_itk, str(marker_path))

    anatomy_volume, anatomy_mapping = (
        build_anatomical_centerline_volume(
            result,
            label.shape,
            line_radius_mm=centerline_label_radius_mm,
        )
    )
    anatomy_path = output / "centerline_anatomy_labels.nii.gz"
    anatomy_itk = sitk.GetImageFromArray(anatomy_volume)
    anatomy_itk.CopyInformation(reference_itk)
    sitk.WriteImage(anatomy_itk, str(anatomy_path))
    anatomy_mapping_path = (
        output / "centerline_anatomy_labels.json"
    )
    with anatomy_mapping_path.open("w", encoding="utf-8") as stream:
        json.dump(
            anatomy_mapping,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    profile_paths = plot_diameter_profiles(result, str(output))
    intensity_profile_paths = plot_intensity_profiles(
        result,
        str(output),
    )
    cross_section_paths = plot_candidate_cross_sections(
        result,
        image,
        label,
        str(output),
        radius_mm=cross_section_radius_mm,
        pixel_mm=cross_section_pixel_mm,
        config=config,
    )
    return {
        "marker_path": marker_path,
        "anatomy_path": anatomy_path,
        "anatomy_mapping_path": anatomy_mapping_path,
        "profile_paths": profile_paths,
        "intensity_profile_paths": intensity_profile_paths,
        "cross_section_paths": cross_section_paths,
    }
