from __future__ import annotations

import inspect
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..cpr import generate_centerline_cpr, generate_stretched_cpr
from ..models import CPRResult
from ..visualization import write_cpr_result
from .centerline_io import (
    INTERPOLATION_ORDER,
    RESAMPLE_SPACING_MM,
    SMOOTHING_SIGMA_MM,
    write_centerline_json,
)
from .image_io import LoadedVolume
from .models import ControlPath


class CPRCancelled(RuntimeError):
    """Raised when a manual CPR request is cancelled at a safe boundary."""


@dataclass(frozen=True)
class ManualCPRRequest:
    volume: LoadedVolume
    path: ControlPath
    output_root: Path
    overwrite: bool = False


def _manual_centerline_zyx(request: ManualCPRRequest) -> np.ndarray:
    points = np.asarray(request.path.control_points_lps_mm, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
        raise ValueError("At least two ordered control points are required.")
    if np.any(~np.isfinite(points)):
        raise ValueError("Control points must contain finite coordinates.")
    if len(np.unique(points, axis=0)) < 2:
        raise ValueError("The path must contain at least two distinct points.")
    if any(
        not request.volume.geometry.contains_lps(point)
        for point in points
    ):
        raise ValueError("Every control point must lie inside the loaded image.")
    indices_xyz = np.asarray(
        [
            request.volume.geometry.lps_to_continuous_index_xyz(point)
            for point in points
        ],
        dtype=float,
    )
    return indices_xyz[:, ::-1].copy()


def _manual_order_arguments(generator: Callable) -> dict[str, bool]:
    parameters = inspect.signature(generator).parameters.values()
    supports_keyword = any(
        parameter.name == "preserve_input_order"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    return {"preserve_input_order": True} if supports_keyword else {}


def generate_manual_cpr(
    request: ManualCPRRequest,
    *,
    is_cancelled: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
) -> list[CPRResult]:
    settings = request.path.cpr_settings
    settings.validate()
    centerline_zyx = _manual_centerline_zyx(request)
    spacing_zyx = tuple(
        float(value)
        for value in reversed(request.volume.geometry.spacing_xyz)
    )
    angles = settings.angles_degrees()
    results: list[CPRResult] = []
    for index, angle in enumerate(angles, start=1):
        if is_cancelled():
            raise CPRCancelled("CPR generation was cancelled.")
        common = dict(
            image=request.volume.array_zyx,
            centerline_voxel_zyx=centerline_zyx,
            spacing_zyx=spacing_zyx,
            angle_degrees=float(angle),
            interpolation_order=INTERPOLATION_ORDER,
            name=request.path.name,
            curve_resolution_mm=RESAMPLE_SPACING_MM,
            centerline_smoothing_sigma_mm=SMOOTHING_SIGMA_MM,
            start_extension_mm=0.0,
            end_extension_mm=0.0,
        )
        if str(settings.mode).lower() == "straightened":
            result = generate_centerline_cpr(
                **common,
                **_manual_order_arguments(generate_centerline_cpr),
                radius_mm=0.5 * float(settings.sampling_line_length_mm),
                pixel_mm=float(settings.slice_resolution_mm),
            )
        else:
            result = generate_stretched_cpr(
                **common,
                **_manual_order_arguments(generate_stretched_cpr),
                sampling_line_length_mm=float(
                    settings.sampling_line_length_mm
                ),
                slice_resolution_mm=float(settings.slice_resolution_mm),
                image_direction_xyz=request.volume.geometry.direction_xyz,
            )
        results.append(result)
        on_progress(index, len(angles))
    return results


def sanitized_path_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "Path"


def _direct_child(root: Path, candidate: Path) -> bool:
    return candidate.resolve(strict=False).parent == root.resolve(strict=False)


def _remove_staging_directory(root: Path, candidate: Path) -> None:
    if not _direct_child(root, candidate):
        raise ValueError(f"Refusing to remove path outside output root: {candidate}")
    if candidate.exists():
        shutil.rmtree(candidate)


def export_manual_cpr(
    request: ManualCPRRequest,
    *,
    is_cancelled: Callable[[], bool],
    on_progress: Callable[[int, int], None],
) -> Path:
    root = Path(request.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = sanitized_path_name(request.path.name)
    target = root / stem
    if not _direct_child(root, target):
        raise ValueError("The output path must be a direct child of the output root.")
    if target.exists() and not request.overwrite:
        raise FileExistsError(f"Output already exists: {target}")

    token = uuid4().hex
    temporary = root / f".{stem}.tmp-{token}"
    backup = root / f".{stem}.backup-{token}"
    temporary.mkdir()
    backup_moved = False
    committed = False
    try:
        results = generate_manual_cpr(
            request,
            is_cancelled=is_cancelled,
            on_progress=on_progress,
        )
        if is_cancelled():
            raise CPRCancelled("CPR generation was cancelled.")
        for result in results:
            write_cpr_result(
                result,
                str(temporary),
                show_centerline=bool(
                    request.path.cpr_settings.show_centerline
                ),
                include_mode_in_filename=True,
            )
            if is_cancelled():
                raise CPRCancelled("CPR export was cancelled.")
        write_centerline_json(
            request.path,
            request.volume.geometry,
            temporary / f"{stem}_centerline.json",
        )
        if target.exists():
            target.rename(backup)
            backup_moved = True
        try:
            temporary.rename(target)
            committed = True
        except Exception:
            if backup_moved and backup.exists() and not target.exists():
                backup.rename(target)
                backup_moved = False
            raise
        if backup_moved:
            _remove_staging_directory(root, backup)
            backup_moved = False
        return target
    finally:
        if temporary.exists():
            _remove_staging_directory(root, temporary)
        if backup.exists():
            if not target.exists():
                backup.rename(target)
            elif committed:
                _remove_staging_directory(root, backup)
