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


def _angle_filename_stem(angle_degrees: float) -> str:
    value = f"{abs(float(angle_degrees)):.6f}".rstrip("0").rstrip(".")
    value = value.replace(".", "p") or "0"
    prefix = "m" if angle_degrees < 0 else ""
    return f"phi_{prefix}{value}deg"


def _candidate_annotation(candidate: StenosisCandidate) -> str:
    percent = 100.0 * float(candidate.max_diameter_stenosis)
    return f"{percent:.1f}%"


def _stretched_candidate_marker_geometry(
    cpr: CPRResult,
    candidate: StenosisCandidate,
    label_level: int = 0,
):
    distances = np.asarray(cpr.centerline_distances_mm, dtype=float)
    centerline = np.column_stack(
        [
            np.asarray(cpr.centerline_x_mm, dtype=float),
            np.asarray(cpr.centerline_y_mm, dtype=float),
        ]
    )
    index = int(
        np.argmin(np.abs(distances - candidate.min_distance_mm))
    )
    center = centerline[index]
    if len(centerline) < 2:
        tangent = np.asarray([1.0, 0.0])
    elif index == 0:
        tangent = centerline[1] - centerline[0]
    elif index == len(centerline) - 1:
        tangent = centerline[-1] - centerline[-2]
    else:
        tangent = centerline[index + 1] - centerline[index - 1]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-8:
        tangent = np.asarray([1.0, 0.0])
    else:
        tangent = tangent / tangent_norm
    normal = np.asarray([-tangent[1], tangent[0]])

    marker_length = float(candidate.reference_diameter_mm)
    if not np.isfinite(marker_length) or marker_length <= 0:
        marker_length = float(candidate.min_diameter_mm)
    if not np.isfinite(marker_length) or marker_length <= 0:
        marker_length = 1.0
    half_length = 0.5 * marker_length
    start = center - half_length * normal
    end = center + half_length * normal

    x_min = float(cpr.display_x_mm[0])
    x_max = float(cpr.display_x_mm[-1])
    y_min = float(cpr.display_y_mm[0])
    y_max = float(cpr.display_y_mm[-1])

    def available_room(direction: np.ndarray) -> float:
        rooms = []
        for coordinate, component, lower, upper in zip(
            center,
            direction,
            (x_min, y_min),
            (x_max, y_max),
        ):
            if abs(component) <= 1e-8:
                continue
            boundary = upper if component > 0 else lower
            room = (boundary - coordinate) / component
            if room >= 0:
                rooms.append(float(room))
        return min(rooms) if rooms else 0.0

    plus_room = available_room(normal)
    minus_room = available_room(-normal)
    label_direction = normal if plus_room >= minus_room else -normal
    label_gap = max(1.0, 0.35 * marker_length) + 1.2 * max(
        int(label_level),
        0,
    )
    label_position = center + label_direction * (
        half_length + label_gap
    )
    return start, end, label_position


def _write_cpr_result(
    cpr: CPRResult,
    output: Path,
    candidates: Sequence[StenosisCandidate],
    dpi: int,
    used_names: set[str],
):
    plt = _pyplot()
    finite = cpr.image[np.isfinite(cpr.image)]
    low, high = (
        np.percentile(finite, (1, 99))
        if finite.size
        else (0.0, 1.0)
    )
    if high <= low:
        high = low + 1.0

    distances = cpr.centerline_distances_mm
    figure, axes = plt.subplots(
        figsize=(10.5, 5.2),
        constrained_layout=True,
    )
    point_labels = cpr.point_anatomical_labels
    if cpr.mode == "stretched":
        if (
            cpr.display_x_mm is None
            or cpr.display_y_mm is None
            or cpr.centerline_x_mm is None
            or cpr.centerline_y_mm is None
        ):
            raise ValueError("Stretched CPR result is missing display geometry.")
        display = axes.imshow(
            cpr.image,
            cmap="gray",
            origin="lower",
            aspect="equal",
            extent=[
                float(cpr.display_x_mm[0]),
                float(cpr.display_x_mm[-1]),
                float(cpr.display_y_mm[0]),
                float(cpr.display_y_mm[-1]),
            ],
            vmin=float(low),
            vmax=float(high),
            interpolation="nearest",
        )
        axes.plot(
            cpr.centerline_x_mm,
            cpr.centerline_y_mm,
            color="#00bcd4",
            linewidth=1.0,
            label="Unfolded centerline",
        )
        for marker_index, candidate in enumerate(candidates):
            marker_start, marker_end, label_position = (
                _stretched_candidate_marker_geometry(
                    cpr,
                    candidate,
                    label_level=marker_index % 3,
                )
            )
            axes.plot(
                [marker_start[0], marker_end[0]],
                [marker_start[1], marker_end[1]],
                color="#d95f02",
                linewidth=1.8,
                solid_capstyle="round",
                zorder=4,
            )
            axes.text(
                float(label_position[0]),
                float(label_position[1]),
                _candidate_annotation(candidate),
                color="#b34700",
                fontsize=8,
                ha="center",
                va="center",
                clip_on=False,
                zorder=5,
            )
        for index in range(1, len(point_labels)):
            if point_labels[index] != point_labels[index - 1]:
                axes.text(
                    cpr.centerline_x_mm[index],
                    cpr.centerline_y_mm[index],
                    point_labels[index],
                    ha="left",
                    va="bottom",
                    color="#6a3d9a",
                )
        axes.set_xlabel("Coordinate along vector of interest (mm)")
        axes.set_ylabel("Accumulated perpendicular distance (mm)")
    else:
        x_start = float(distances[0])
        x_end = float(distances[-1])
        if x_end <= x_start:
            x_end = x_start + 1e-6
        display = axes.imshow(
            cpr.image.T,
            cmap="gray",
            origin="lower",
            aspect="auto",
            extent=[
                x_start,
                x_end,
                float(cpr.offsets_mm[0]),
                float(cpr.offsets_mm[-1]),
            ],
            vmin=float(low),
            vmax=float(high),
            interpolation="nearest",
        )
        _shade_candidates(axes, candidates)
        axes.axhline(0.0, color="#00bcd4", linewidth=0.8, alpha=0.8)
        y_max = float(cpr.offsets_mm[-1])
        for marker_index, candidate in enumerate(candidates):
            marker_x = float(candidate.min_distance_mm)
            axes.axvline(
                marker_x,
                color="#d95f02",
                linewidth=1.2,
                linestyle="--",
                alpha=0.95,
                zorder=3,
            )
            axes.annotate(
                _candidate_annotation(candidate),
                xy=(marker_x, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 5 + 12 * (marker_index % 3)),
                textcoords="offset points",
                color="#b34700",
                fontsize=8,
                ha="center",
                va="bottom",
                annotation_clip=False,
                zorder=5,
            )
            axes.scatter(
                marker_x,
                y_max,
                color="#d95f02",
                marker="v",
                s=22,
                zorder=4,
            )
        for index in range(1, len(point_labels)):
            if point_labels[index] == point_labels[index - 1]:
                continue
            transition = float(distances[index])
            axes.axvline(
                transition,
                color="#9467bd",
                linewidth=0.9,
                linestyle=":",
            )
            axes.text(
                transition,
                float(cpr.offsets_mm[-1]),
                point_labels[index],
                ha="left",
                va="top",
                color="#6a3d9a",
            )
        axes.set_xlabel("Centerline distance (mm)")
        axes.set_ylabel("Signed offset along sampling line (mm)")

    axes.set_title(
        f"{cpr.anatomical_label} {cpr.mode} CPR, "
        f"phi={cpr.angle_degrees:g} deg"
    )
    colorbar = figure.colorbar(display, ax=axes, pad=0.02)
    colorbar.set_label("Normalized intensity")

    anatomical_stem = _anatomical_filename_stem(
        cpr.anatomical_label,
        cpr.branch_id,
    )
    mode_stem = "_stretched" if cpr.mode == "stretched" else ""
    filename_stem = (
        f"{anatomical_stem}{mode_stem}_CPR_"
        f"{_angle_filename_stem(cpr.angle_degrees)}"
    )
    png_path = _unique_output_path(
        output,
        filename_stem,
        ".png",
        used_names,
    )
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return png_path


def write_cpr_result(
    cpr: CPRResult,
    output_dir: str,
    dpi: int = 160,
    candidates: Sequence[StenosisCandidate] = (),
) -> Dict[str, Path]:
    """Write one standalone CPR result with optional stenosis markers."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    png_path = _write_cpr_result(
        cpr=cpr,
        output=output,
        candidates=candidates,
        dpi=dpi,
        used_names=set(),
    )
    return {"png_path": png_path}


def write_cpr_visualizations(
    result: DetectionResult,
    image: np.ndarray,
    output_dir: str,
    angles_degrees: Sequence[float],
    radius_mm: float = 5.0,
    pixel_mm: float = 0.15,
    dpi: int = 160,
) -> Dict[str, List[Path]]:
    """Write one CPR PNG per branch and angle."""
    if radius_mm <= 0 or pixel_mm <= 0:
        raise ValueError("CPR radius and pixel size must be positive.")
    angles = [float(angle) for angle in angles_degrees]
    if any(not np.isfinite(angle) for angle in angles):
        raise ValueError("All CPR angles must be finite.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    png_paths: List[Path] = []
    used_names: set[str] = set()

    for vessel in result.vessels.values():
        for branch in vessel.branches:
            if not branch.measurements:
                continue
            for angle in angles:
                cpr = generate_branch_cpr(
                    image=image,
                    spacing_zyx=result.spacing_zyx,
                    branch=branch,
                    angle_degrees=angle,
                    radius_mm=radius_mm,
                    pixel_mm=pixel_mm,
                )
                png_path = _write_cpr_result(
                    cpr=cpr,
                    output=output,
                    candidates=branch.candidates,
                    dpi=dpi,
                    used_names=used_names,
                )
                png_paths.append(png_path)

    return {"png_paths": png_paths}
