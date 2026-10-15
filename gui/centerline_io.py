from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from .models import CPRSettings, ControlPath, ImageGeometry


FORMAT_NAME = "narrowdetecter-centerline"
FORMAT_VERSION = 1

RESAMPLE_SPACING_MM = 0.2
SMOOTHING_SIGMA_MM = 0.8
INTERPOLATION_ORDER = 3
START_EXTENSION_MM = 0.0
END_EXTENSION_MM = 0.0


def _finite_triplet(values: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values.")
    return tuple(float(value) for value in array)


def _settings_document(settings: CPRSettings) -> dict[str, object]:
    settings.validate()
    return {
        "mode": str(settings.mode).lower(),
        "angle_start_degrees": float(settings.angle_start_degrees),
        "angle_end_degrees": float(settings.angle_end_degrees),
        "angle_step_degrees": float(settings.angle_step_degrees),
        "sampling_line_length_mm": float(settings.sampling_line_length_mm),
        "slice_resolution_mm": float(settings.slice_resolution_mm),
        "show_centerline": bool(settings.show_centerline),
    }


def path_to_document(
    path: ControlPath,
    geometry: ImageGeometry,
) -> dict[str, object]:
    name = str(path.name).strip()
    if not name:
        raise ValueError("Path name cannot be empty.")
    if len(path.control_points_lps_mm) < 2:
        raise ValueError("At least two control points are required for export.")
    points = [
        list(_finite_triplet(point, "control point"))
        for point in path.control_points_lps_mm
    ]
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "name": name,
        "coordinate_system": "LPS_mm",
        "image_geometry": {
            "size": [int(value) for value in geometry.size_xyz],
            "spacing": [float(value) for value in geometry.spacing_xyz],
            "origin": [float(value) for value in geometry.origin_lps_xyz],
            "direction": [float(value) for value in geometry.direction_xyz],
        },
        "control_points_lps_mm": points,
        "fit_settings": {
            "resample_spacing_mm": RESAMPLE_SPACING_MM,
            "smoothing_sigma_mm": SMOOTHING_SIGMA_MM,
            "interpolation_order": INTERPOLATION_ORDER,
            "start_extension_mm": START_EXTENSION_MM,
            "end_extension_mm": END_EXTENSION_MM,
        },
        "cpr_settings": _settings_document(path.cpr_settings),
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def write_centerline_json(
    path: ControlPath,
    geometry: ImageGeometry,
    destination: str | Path,
) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = path_to_document(path, geometry)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str):
    raise ValueError(f"Non-finite JSON number: {value}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _read_geometry(document: dict[str, Any]) -> ImageGeometry:
    geometry = _mapping(document.get("image_geometry"), "image_geometry")
    try:
        size_values = geometry["size"]
        size = tuple(int(value) for value in size_values)
        if any(float(value) != int(value) for value in size_values):
            raise ValueError("Image size values must be integers.")
        spacing = tuple(float(value) for value in geometry["spacing"])
        origin = tuple(float(value) for value in geometry["origin"])
        direction = tuple(float(value) for value in geometry["direction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid image_geometry values.") from error
    return ImageGeometry(
        size_xyz=size,
        spacing_xyz=spacing,
        origin_lps_xyz=origin,
        direction_xyz=direction,
    )


def _read_settings(document: dict[str, Any]) -> CPRSettings:
    values = _mapping(document.get("cpr_settings"), "cpr_settings")
    try:
        settings = CPRSettings(
            mode=str(values["mode"]),
            angle_start_degrees=float(values["angle_start_degrees"]),
            angle_end_degrees=float(values["angle_end_degrees"]),
            angle_step_degrees=float(values["angle_step_degrees"]),
            sampling_line_length_mm=float(values["sampling_line_length_mm"]),
            slice_resolution_mm=float(values["slice_resolution_mm"]),
            show_centerline=bool(values["show_centerline"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid cpr_settings values.") from error
    settings.validate()
    return settings


def _validate_fit_settings(document: dict[str, Any]) -> None:
    values = _mapping(document.get("fit_settings"), "fit_settings")
    expected = {
        "resample_spacing_mm": RESAMPLE_SPACING_MM,
        "smoothing_sigma_mm": SMOOTHING_SIGMA_MM,
        "interpolation_order": INTERPOLATION_ORDER,
        "start_extension_mm": START_EXTENSION_MM,
        "end_extension_mm": END_EXTENSION_MM,
    }
    for key, expected_value in expected.items():
        try:
            actual = float(values[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid fit_settings.{key} value.") from error
        if not np.isfinite(actual) or not np.isclose(actual, expected_value):
            raise ValueError(
                f"Unsupported fit_settings.{key}: {actual}."
            )


def read_centerline_json(
    source: str | Path,
) -> tuple[ControlPath, ImageGeometry]:
    try:
        document = json.loads(
            Path(source).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid centerline JSON: {error}") from error
    document = _mapping(document, "document")
    if document.get("format") != FORMAT_NAME:
        raise ValueError("Unsupported centerline JSON format.")
    if document.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported centerline JSON version.")
    if document.get("coordinate_system") != "LPS_mm":
        raise ValueError("Only LPS_mm centerline coordinates are supported.")
    geometry = _read_geometry(document)
    _validate_fit_settings(document)
    settings = _read_settings(document)
    name = str(document.get("name", "")).strip()
    if not name:
        raise ValueError("Centerline name cannot be empty.")
    raw_points = document.get("control_points_lps_mm")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise ValueError("At least two control points are required.")
    points = [
        _finite_triplet(point, "control point")
        for point in raw_points
    ]
    path = ControlPath(
        path_id=str(uuid4()),
        name=name,
        control_points_lps_mm=points,
        visible=True,
        dirty=False,
        cpr_settings=settings,
    )
    return path, geometry


def geometry_matches(left: ImageGeometry, right: ImageGeometry) -> bool:
    return bool(
        left.size_xyz == right.size_xyz
        and np.allclose(left.spacing_xyz, right.spacing_xyz, atol=1e-5)
        and np.allclose(left.origin_lps_xyz, right.origin_lps_xyz, atol=1e-4)
        and np.allclose(left.direction_xyz, right.direction_xyz, atol=1e-5)
    )
