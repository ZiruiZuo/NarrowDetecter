from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence
from uuid import uuid4

import numpy as np


def _finite_triplet(values: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values.")
    return tuple(float(value) for value in array)


@dataclass(frozen=True)
class ImageGeometry:
    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_lps_xyz: tuple[float, float, float]
    direction_xyz: tuple[float, ...]

    def __post_init__(self) -> None:
        size = np.asarray(self.size_xyz)
        spacing = np.asarray(self.spacing_xyz, dtype=float)
        origin = np.asarray(self.origin_lps_xyz, dtype=float)
        direction = np.asarray(self.direction_xyz, dtype=float)
        if size.shape != (3,) or np.any(size <= 0):
            raise ValueError("size_xyz must contain three positive values.")
        if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("spacing_xyz must contain three positive finite values.")
        if origin.shape != (3,) or np.any(~np.isfinite(origin)):
            raise ValueError("origin_lps_xyz must contain three finite values.")
        if direction.size != 9 or np.any(~np.isfinite(direction)):
            raise ValueError("direction_xyz must contain nine finite values.")
        matrix = direction.reshape(3, 3)
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
            raise ValueError("direction_xyz must be orthonormal.")

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction_xyz, dtype=float).reshape(3, 3)

    def continuous_index_xyz_to_lps(
        self,
        index_xyz: Sequence[float],
    ) -> tuple[float, float, float]:
        index = np.asarray(_finite_triplet(index_xyz, "index_xyz"))
        spacing = np.asarray(self.spacing_xyz, dtype=float)
        origin = np.asarray(self.origin_lps_xyz, dtype=float)
        point = origin + self.direction_matrix @ (index * spacing)
        return tuple(float(value) for value in point)

    def lps_to_continuous_index_xyz(
        self,
        point_lps: Sequence[float],
    ) -> tuple[float, float, float]:
        point = np.asarray(_finite_triplet(point_lps, "point_lps"))
        origin = np.asarray(self.origin_lps_xyz, dtype=float)
        spacing = np.asarray(self.spacing_xyz, dtype=float)
        index = self.direction_matrix.T @ (point - origin)
        index /= spacing
        return tuple(float(value) for value in index)

    def contains_lps(
        self,
        point_lps: Sequence[float],
        tolerance_voxels: float = 0.5,
    ) -> bool:
        tolerance = float(tolerance_voxels)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance_voxels must be finite and nonnegative.")
        index = np.asarray(self.lps_to_continuous_index_xyz(point_lps))
        upper = np.asarray(self.size_xyz, dtype=float) - 1.0 + tolerance
        return bool(np.all(index >= -tolerance) and np.all(index <= upper))


@dataclass
class CPRSettings:
    mode: str = "stretched"
    angle_start_degrees: float = 0.0
    angle_end_degrees: float = 165.0
    angle_step_degrees: float = 15.0
    sampling_line_length_mm: float = 40.0
    slice_resolution_mm: float = 1.0
    show_centerline: bool = True

    def validate(self) -> None:
        if str(self.mode).lower() not in {"straightened", "stretched"}:
            raise ValueError("mode must be 'straightened' or 'stretched'.")
        numbers = np.asarray(
            [
                self.angle_start_degrees,
                self.angle_end_degrees,
                self.angle_step_degrees,
                self.sampling_line_length_mm,
                self.slice_resolution_mm,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(numbers)):
            raise ValueError("CPR settings must contain finite values.")
        if self.sampling_line_length_mm <= 0 or self.slice_resolution_mm <= 0:
            raise ValueError("CPR width and resolution must be positive.")
        if self.angle_step_degrees == 0:
            raise ValueError("angle_step_degrees cannot be zero.")
        delta = self.angle_end_degrees - self.angle_start_degrees
        if delta != 0 and delta * self.angle_step_degrees < 0:
            raise ValueError("angle_step_degrees points away from the end angle.")
        if len(self._angles_unchecked()) > 360:
            raise ValueError("At most 360 CPR angles can be generated.")

    def _angles_unchecked(self) -> tuple[float, ...]:
        start = float(self.angle_start_degrees)
        end = float(self.angle_end_degrees)
        step = float(self.angle_step_degrees)
        if step == 0:
            return ()
        if start == end:
            return (start,)
        delta = end - start
        if delta * step < 0:
            return ()
        count = int(np.floor(abs(delta / step) + 1e-10)) + 1
        return tuple(start + index * step for index in range(count))

    def angles_degrees(self) -> tuple[float, ...]:
        self.validate()
        return self._angles_unchecked()


@dataclass
class ControlPath:
    path_id: str
    name: str
    control_points_lps_mm: list[tuple[float, float, float]] = field(
        default_factory=list
    )
    visible: bool = True
    dirty: bool = False
    cpr_settings: CPRSettings = field(default_factory=CPRSettings)


class PathStore:
    def __init__(self) -> None:
        self._paths: dict[str, ControlPath] = {}
        self._active_path_id: str | None = None
        self._next_default_index = 1

    @property
    def paths(self) -> tuple[ControlPath, ...]:
        return tuple(self._paths.values())

    @property
    def active_path_id(self) -> str | None:
        return self._active_path_id

    @property
    def active_path(self) -> ControlPath | None:
        if self._active_path_id is None:
            return None
        return self._paths[self._active_path_id]

    def get_path(self, path_id: str) -> ControlPath:
        try:
            return self._paths[path_id]
        except KeyError as error:
            raise KeyError(f"Unknown path id: {path_id}") from error

    def _name_exists(self, name: str, excluding: str | None = None) -> bool:
        folded = name.casefold()
        return any(
            path.path_id != excluding and path.name.casefold() == folded
            for path in self._paths.values()
        )

    def _normalize_name(self, name: str) -> str:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Path name cannot be empty.")
        return normalized

    def _default_name(self) -> str:
        while True:
            name = f"Path_{self._next_default_index:03d}"
            self._next_default_index += 1
            if not self._name_exists(name):
                return name

    def create_path(self, name: str | None = None) -> ControlPath:
        normalized = self._default_name() if name is None else self._normalize_name(name)
        if self._name_exists(normalized):
            raise ValueError(f"A path named '{normalized}' already exists.")
        settings = (
            replace(self.active_path.cpr_settings)
            if self.active_path is not None
            else CPRSettings()
        )
        path = ControlPath(
            path_id=str(uuid4()),
            name=normalized,
            cpr_settings=settings,
        )
        self._paths[path.path_id] = path
        self._active_path_id = path.path_id
        return path

    def add_imported_path(self, source: ControlPath) -> ControlPath:
        base_name = self._normalize_name(source.name)
        name = base_name
        sequence = 2
        while self._name_exists(name):
            name = f"{base_name}_{sequence}"
            sequence += 1
        imported = ControlPath(
            path_id=str(uuid4()),
            name=name,
            control_points_lps_mm=[
                _finite_triplet(point, "control point")
                for point in source.control_points_lps_mm
            ],
            visible=bool(source.visible),
            dirty=False,
            cpr_settings=replace(source.cpr_settings),
        )
        imported.cpr_settings.validate()
        self._paths[imported.path_id] = imported
        self._active_path_id = imported.path_id
        return imported

    def set_active(self, path_id: str) -> None:
        self.get_path(path_id)
        self._active_path_id = path_id

    def rename_path(self, path_id: str, name: str) -> None:
        path = self.get_path(path_id)
        normalized = self._normalize_name(name)
        if self._name_exists(normalized, excluding=path_id):
            raise ValueError(f"A path named '{normalized}' already exists.")
        if path.name != normalized:
            path.name = normalized
            path.dirty = True

    def set_visibility(self, path_id: str, visible: bool) -> None:
        self.get_path(path_id).visible = bool(visible)

    def delete_path(self, path_id: str) -> None:
        self.get_path(path_id)
        path_ids = list(self._paths)
        deleted_index = path_ids.index(path_id)
        del self._paths[path_id]
        if self._active_path_id != path_id:
            return
        remaining_ids = list(self._paths)
        if not remaining_ids:
            self._active_path_id = None
        elif deleted_index < len(remaining_ids):
            self._active_path_id = remaining_ids[deleted_index]
        else:
            self._active_path_id = remaining_ids[-1]

    def _require_active(self, path_id: str | None = None) -> ControlPath:
        path = self.active_path
        if path is None:
            raise ValueError("No active path.")
        if path_id is not None and path.path_id != path_id:
            raise PermissionError("Only the active path can be edited.")
        return path

    def append_point(self, point_lps: Sequence[float]) -> int:
        path = self._require_active()
        point = _finite_triplet(point_lps, "point_lps")
        path.control_points_lps_mm.append(point)
        path.dirty = True
        return len(path.control_points_lps_mm) - 1

    def move_point_in_view(
        self,
        path_id: str,
        point_index: int,
        orientation: str,
        new_lps: Sequence[float],
        geometry: ImageGeometry,
    ) -> None:
        path = self._require_active(path_id)
        normal_axis = {"sagittal": 0, "coronal": 1, "axial": 2}
        try:
            axis = normal_axis[str(orientation).lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported orientation: {orientation}") from error
        old = path.control_points_lps_mm[point_index]
        moved_index = list(geometry.lps_to_continuous_index_xyz(new_lps))
        old_index = geometry.lps_to_continuous_index_xyz(old)
        moved_index[axis] = old_index[axis]
        path.control_points_lps_mm[point_index] = (
            geometry.continuous_index_xyz_to_lps(moved_index)
        )
        path.dirty = True

    def delete_point(self, path_id: str, point_index: int) -> None:
        path = self._require_active(path_id)
        del path.control_points_lps_mm[point_index]
        path.dirty = True

    def clear_active_path(self) -> None:
        path = self._require_active()
        if path.control_points_lps_mm:
            path.control_points_lps_mm.clear()
            path.dirty = True

    def mark_exported(self, path_id: str) -> None:
        self.get_path(path_id).dirty = False
