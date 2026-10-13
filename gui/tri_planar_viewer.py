from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QWidget

from .image_io import LoadedVolume
from .image_view import ImageView
from .models import ControlPath


class TriPlanarViewer(QWidget):
    point_append_requested = Signal(str, object)
    point_move_requested = Signal(str, str, int, object)
    point_delete_requested = Signal(str, int)
    path_clear_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._crosshair_lps = np.zeros(3, dtype=float)
        self._volume: LoadedVolume | None = None
        self._maximized: ImageView | None = None
        self.views = {
            name: ImageView(name, self)
            for name in ("axial", "coronal", "sagittal")
        }
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.views["axial"], 0, 0)
        layout.addWidget(self.views["coronal"], 0, 1)
        layout.addWidget(self.views["sagittal"], 1, 0)
        self._grid = layout
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        for view in self.views.values():
            view.point_append_requested.connect(self.point_append_requested)
            view.point_move_requested.connect(self.point_move_requested)
            view.point_delete_requested.connect(self.point_delete_requested)
            view.path_clear_requested.connect(self.path_clear_requested)
            view.slice_position_changed.connect(self._on_slice_position_changed)
            view.maximize_requested.connect(self.toggle_maximize)

    @property
    def crosshair_lps(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._crosshair_lps)

    def set_volume(self, volume: LoadedVolume) -> None:
        self._volume = volume
        center_index = np.asarray(volume.geometry.size_xyz, dtype=int) // 2
        center = volume.geometry.continuous_index_xyz_to_lps(center_index)
        self._crosshair_lps = np.asarray(center, dtype=float)
        for view in self.views.values():
            view.set_volume(volume, center)

    def set_crosshair_lps(self, point_lps: Sequence[float]) -> None:
        point = np.asarray(point_lps, dtype=float)
        if point.shape != (3,) or np.any(~np.isfinite(point)):
            raise ValueError("Crosshair must contain three finite LPS coordinates.")
        self._crosshair_lps = point.copy()
        for view in self.views.values():
            view.set_crosshair_lps(point)

    def set_paths(
        self,
        paths: Sequence[ControlPath],
        active_path_id: str | None,
    ) -> None:
        for view in self.views.values():
            view.set_paths(paths, active_path_id)

    def set_window_level(self, window: float, level: float) -> None:
        for view in self.views.values():
            view.set_window_level(window, level)

    def _on_slice_position_changed(self, orientation: str, position: float) -> None:
        if self._volume is None:
            return
        axis = {"sagittal": 0, "coronal": 1, "axial": 2}[orientation]
        geometry = self._volume.geometry
        index = np.asarray(geometry.lps_to_continuous_index_xyz(self._crosshair_lps))
        index[axis] = float(position)
        self.set_crosshair_lps(geometry.continuous_index_xyz_to_lps(index))

    def toggle_maximize(self, view: ImageView) -> None:
        if view not in self.views.values():
            raise ValueError("The requested view does not belong to this viewer.")
        if self._maximized is view:
            self._maximized = None
            for index in (0, 1):
                self._grid.setRowStretch(index, 1)
                self._grid.setColumnStretch(index, 1)
            for candidate in self.views.values():
                candidate.show()
            return
        self._maximized = view
        for candidate in self.views.values():
            candidate.setVisible(candidate is view)
        row, column, _, _ = self._grid.getItemPosition(self._grid.indexOf(view))
        for index in (0, 1):
            self._grid.setRowStretch(index, int(index == row))
            self._grid.setColumnStretch(index, int(index == column))

