from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QMenu, QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPlane, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkImageSlice,
    vtkPolyDataMapper,
    vtkPropPicker,
    vtkRenderer,
)
from vtkmodules.vtkRenderingImage import vtkImageResliceMapper
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  (register OpenGL factories)

from .image_io import LoadedVolume
from .interactions import ClickDisambiguator, HitTarget
from .models import ControlPath
from .vtk_utils import display_position_to_lps, loaded_volume_to_vtk, orthogonal_plane_axes


class ImageView(QWidget):
    point_append_requested = Signal(str, object)
    point_move_requested = Signal(str, str, int, object)
    point_delete_requested = Signal(str, int)
    path_clear_requested = Signal(str)
    slice_position_changed = Signal(str, float)
    maximize_requested = Signal(object)

    def __init__(self, orientation: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.orientation = str(orientation).lower()
        self._slice_axis = {"sagittal": 0, "coronal": 1, "axial": 2}[self.orientation]
        self._axes = orthogonal_plane_axes(self.orientation)
        self._normal = self._axes[2]
        self._volume: LoadedVolume | None = None
        self._vtk_image = None
        self._crosshair_lps = np.zeros(3, dtype=float)
        self._paths: tuple[ControlPath, ...] = ()
        self._active_path_id: str | None = None
        self._point_actors: dict[object, HitTarget] = {}
        self._path_line_actors: list[object] = []
        self._crosshair_actors: list[object] = []
        self._drag_target: HitTarget | None = None
        self._dragged = False
        self._middle_last: QPoint | None = None
        self._suppress_next_release = False

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.04, 0.04, 0.04)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.GetRenderWindow().SetOffScreenRendering(
            1 if QApplication.platformName().lower() == "offscreen" else 0
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)

        self._slice_plane = vtkPlane()
        self._slice_plane.SetNormal(*self._normal)
        self._image_mapper = vtkImageResliceMapper()
        self._image_mapper.SetSlicePlane(self._slice_plane)
        self._image_mapper.SliceFacesCameraOff()
        self._image_mapper.SliceAtFocalPointOff()
        self._image_actor = vtkImageSlice()
        self._image_actor.SetMapper(self._image_mapper)
        self.renderer.AddViewProp(self._image_actor)

        self._picker = vtkPropPicker()
        app = QApplication.instance()
        interval = app.styleHints().mouseDoubleClickInterval() if app else 400
        self._clicks = ClickDisambiguator(
            interval,
            self._schedule,
            self._cancel_timer,
        )
        self.vtk_widget.installEventFilter(self)

    @property
    def crosshair_lps(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._crosshair_lps)

    def _schedule(self, delay_ms, callback):
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(callback)
        timer.start(int(delay_ms))
        return timer

    @staticmethod
    def _cancel_timer(timer) -> None:
        timer.stop()
        timer.deleteLater()

    def set_volume(
        self,
        volume: LoadedVolume,
        crosshair_lps: Sequence[float] | None = None,
    ) -> None:
        self._volume = volume
        self._axes = orthogonal_plane_axes(self.orientation) @ volume.geometry.direction_matrix.T
        self._normal = self._axes[2]
        self._slice_plane.SetNormal(*self._normal)
        self._vtk_image = loaded_volume_to_vtk(volume)
        self._image_mapper.SetInputData(self._vtk_image)
        if crosshair_lps is None:
            center_index = np.asarray(volume.geometry.size_xyz, dtype=int) // 2
            crosshair_lps = volume.geometry.continuous_index_xyz_to_lps(center_index)
        self.set_crosshair_lps(crosshair_lps)
        self._configure_camera()
        self._refresh_crosshair_actors()
        # Initializing before the mapper has input can crash native VTK.
        if (
            QApplication.platformName().lower() != "offscreen"
            and not self.vtk_widget.GetEnabled()
        ):
            self.vtk_widget.Initialize()
        self.vtk_widget.update()

    def _configure_camera(self) -> None:
        camera = self.renderer.GetActiveCamera()
        # Reverse the viewing side to mirror the horizontal screen axis only.
        normal = -np.cross(self._axes[0], self._axes[1])
        view_up = self._axes[1]
        center = self._crosshair_lps
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + normal * 500.0))
        camera.SetViewUp(*view_up)
        camera.ParallelProjectionOn()
        self.renderer.ResetCamera()

    def set_crosshair_lps(self, point_lps: Sequence[float]) -> None:
        point = np.asarray(point_lps, dtype=float)
        if point.shape != (3,) or np.any(~np.isfinite(point)):
            raise ValueError("Crosshair must contain three finite LPS coordinates.")
        if self._volume is not None:
            geometry = self._volume.geometry
            index = np.asarray(geometry.lps_to_continuous_index_xyz(point))
            index[self._slice_axis] = np.clip(
                np.floor(index[self._slice_axis] + 0.5),
                0,
                geometry.size_xyz[self._slice_axis] - 1,
            )
            point = np.asarray(geometry.continuous_index_xyz_to_lps(index))
        self._crosshair_lps = point.copy()
        self._slice_plane.SetOrigin(*point)
        self._refresh_crosshair_actors()
        self._refresh_point_actors()
        self.vtk_widget.update()

    def set_paths(
        self,
        paths: Sequence[ControlPath],
        active_path_id: str | None,
    ) -> None:
        self._paths = tuple(paths)
        self._active_path_id = active_path_id
        self._refresh_point_actors()
        self.vtk_widget.update()

    def set_window_level(self, window: float, level: float) -> None:
        window = float(window)
        level = float(level)
        if not np.isfinite(window) or window <= 0 or not np.isfinite(level):
            raise ValueError("Window must be positive and window/level must be finite.")
        prop = self._image_actor.GetProperty()
        prop.SetColorWindow(window)
        prop.SetColorLevel(level)
        self.vtk_widget.update()

    def _slice_spacing_mm(self) -> float:
        if self._volume is None:
            return 1.0
        return float(self._volume.geometry.spacing_xyz[self._slice_axis])

    def _clear_point_actors(self) -> None:
        for actor in tuple(self._point_actors):
            self.renderer.RemoveActor(actor)
        self._point_actors.clear()
        for actor in self._path_line_actors:
            self.renderer.RemoveActor(actor)
        self._path_line_actors.clear()

    def _line_actor(
        self,
        start,
        end,
        *,
        color=(0.0, 0.75, 0.9),
        opacity=0.65,
        width=1.0,
    ):
        source = vtkLineSource()
        source.SetPoint1(*start)
        source.SetPoint2(*end)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(float(opacity))
        actor.GetProperty().SetLineWidth(float(width))
        actor.SetPickable(False)
        return actor

    def _refresh_crosshair_actors(self) -> None:
        for actor in self._crosshair_actors:
            self.renderer.RemoveActor(actor)
        self._crosshair_actors.clear()
        if self._volume is None:
            return
        geometry = self._volume.geometry
        upper = np.asarray(geometry.size_xyz, dtype=float) - 1.0
        corners = np.asarray(
            [
                geometry.continuous_index_xyz_to_lps((x, y, z))
                for x in (0.0, upper[0])
                for y in (0.0, upper[1])
                for z in (0.0, upper[2])
            ],
            dtype=float,
        )
        for axis in self._axes[:2]:
            values = corners @ axis
            center_value = float(np.dot(self._crosshair_lps, axis))
            start = self._crosshair_lps + (float(values.min()) - center_value) * axis
            end = self._crosshair_lps + (float(values.max()) - center_value) * axis
            actor = self._line_actor(start, end, opacity=0.55)
            self.renderer.AddActor(actor)
            self._crosshair_actors.append(actor)

    def _add_path_line_actor(
        self,
        path: ControlPath,
        spacing: float,
        slice_position: float,
        active: bool,
    ) -> None:
        points = vtkPoints()
        lines = vtkCellArray()
        segment_count = 0
        coordinates = path.control_points_lps_mm
        for left, right in zip(coordinates, coordinates[1:]):
            left_array = np.asarray(left, dtype=float)
            right_array = np.asarray(right, dtype=float)
            left_distance = abs(float(np.dot(self._normal, left_array)) - slice_position)
            right_distance = abs(float(np.dot(self._normal, right_array)) - slice_position)
            if max(left_distance, right_distance) > 1.5 * spacing:
                continue
            left_id = points.InsertNextPoint(*left_array)
            right_id = points.InsertNextPoint(*right_array)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(left_id)
            lines.InsertCellPoint(right_id)
            segment_count += 1
        if segment_count == 0:
            return
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        color = (1.0, 0.75, 0.1) if active else (0.45, 0.7, 1.0)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.9 if active else 0.45)
        actor.GetProperty().SetLineWidth(2.0 if active else 1.0)
        actor.SetPickable(False)
        self.renderer.AddActor(actor)
        self._path_line_actors.append(actor)

    def _refresh_point_actors(self) -> None:
        self._clear_point_actors()
        spacing = self._slice_spacing_mm()
        slice_position = float(np.dot(self._normal, self._crosshair_lps))
        for path in self._paths:
            if not path.visible:
                continue
            active = path.path_id == self._active_path_id
            self._add_path_line_actor(path, spacing, slice_position, active)
            for index, coordinates in enumerate(path.control_points_lps_mm):
                point = np.asarray(coordinates, dtype=float)
                distance = abs(float(np.dot(self._normal, point)) - slice_position)
                if distance > 1.5 * spacing:
                    continue
                sphere = vtkSphereSource()
                sphere.SetCenter(*point)
                sphere.SetRadius(max(0.7, 0.65 * float(np.min(
                    self._volume.geometry.spacing_xyz if self._volume else (1.0,)
                ))))
                sphere.SetThetaResolution(12)
                sphere.SetPhiResolution(12)
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                color = (1.0, 0.75, 0.1) if active else (0.45, 0.7, 1.0)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetOpacity(
                    1.0 if distance <= 0.5 * spacing else 0.25
                )
                # Solid inactive points remain pickable for hit suppression,
                # but event handling only permits edits on the active path.
                actor.SetPickable(distance <= 0.5 * spacing)
                self.renderer.AddActor(actor)
                self._point_actors[actor] = HitTarget(path.path_id, index, active)

    def _display_xy(self, event) -> tuple[float, float]:
        position = event.position()
        return float(position.x()), float(self.vtk_widget.height() - position.y())

    def _display_to_plane(self, event) -> tuple[float, float, float]:
        return display_position_to_lps(
            self.renderer,
            self._display_xy(event),
            self._crosshair_lps,
            self._normal,
        )

    def _pick(self, event) -> HitTarget | None:
        x, y = self._display_xy(event)
        if not self._picker.Pick(x, y, 0.0, self.renderer):
            return None
        return self._point_actors.get(self._picker.GetViewProp())

    def _emit_append(self, point_lps) -> None:
        self.point_append_requested.emit(self.orientation, point_lps)

    def _show_context_menu(self, target: HitTarget, global_position) -> None:
        menu = QMenu(self)
        delete_action = menu.addAction("Delete point")
        clear_action = menu.addAction("Clear path")
        chosen = menu.exec(global_position)
        if chosen is delete_action:
            self.point_delete_requested.emit(target.path_id, target.point_index)
        elif chosen is clear_action:
            self.path_clear_requested.emit(target.path_id)

    def _zoom(self, direction: int) -> None:
        camera = self.renderer.GetActiveCamera()
        camera.Zoom(1.12 if direction > 0 else 1.0 / 1.12)
        self.vtk_widget.update()

    def _slice(self, direction: int) -> None:
        if self._volume is None:
            return
        geometry = self._volume.geometry
        index = np.asarray(geometry.lps_to_continuous_index_xyz(self._crosshair_lps))
        axis = self._slice_axis
        next_slice = int(np.clip(
            int(np.floor(index[axis] + 0.5)) + (1 if direction > 0 else -1),
            0,
            geometry.size_xyz[axis] - 1,
        ))
        index[axis] = next_slice
        self.set_crosshair_lps(geometry.continuous_index_xyz_to_lps(index))
        self.slice_position_changed.emit(self.orientation, float(next_slice))

    def _pan(self, event) -> None:
        if self._middle_last is None:
            return
        old = self._middle_last
        current = event.position().toPoint()
        try:
            old_lps = display_position_to_lps(
                self.renderer,
                (old.x(), self.vtk_widget.height() - old.y()),
                self._crosshair_lps,
                self._normal,
            )
            new_lps = self._display_to_plane(event)
        except ValueError:
            self._middle_last = current
            return
        shift = np.asarray(old_lps) - np.asarray(new_lps)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*(np.asarray(camera.GetPosition()) + shift))
        camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint()) + shift))
        self._middle_last = current
        self.vtk_widget.update()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is not self.vtk_widget:
            return super().eventFilter(watched, event)
        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonDblClick:
            mouse = event
            if mouse.button() == Qt.MouseButton.LeftButton:
                self._suppress_next_release = True
                self._clicks.double_click(lambda: self.maximize_requested.emit(self))
                return True
        if event_type == QEvent.Type.MouseButtonPress:
            mouse: QMouseEvent = event
            if mouse.button() == Qt.MouseButton.MiddleButton:
                self._middle_last = mouse.position().toPoint()
                return True
            target = self._pick(mouse)
            if mouse.button() == Qt.MouseButton.LeftButton:
                if target and target.is_active:
                    self._clicks.cancel_pending()
                    self._drag_target = target
                    self._dragged = False
                return True
            if mouse.button() == Qt.MouseButton.RightButton:
                if target and target.is_active:
                    self._show_context_menu(target, mouse.globalPosition().toPoint())
                return True
        if event_type == QEvent.Type.MouseMove:
            mouse = event
            if self._drag_target is not None:
                self._dragged = True
                try:
                    point = self._display_to_plane(mouse)
                except ValueError:
                    return True
                target = self._drag_target
                self.point_move_requested.emit(
                    self.orientation,
                    target.path_id,
                    target.point_index,
                    point,
                )
                return True
            if self._middle_last is not None:
                self._pan(mouse)
                return True
            if mouse.buttons() & Qt.MouseButton.LeftButton:
                return True
        if event_type == QEvent.Type.MouseButtonRelease:
            mouse = event
            if mouse.button() == Qt.MouseButton.MiddleButton:
                self._middle_last = None
                return True
            if mouse.button() == Qt.MouseButton.LeftButton:
                if self._drag_target is not None:
                    self._drag_target = None
                    self._dragged = False
                    return True
                if self._suppress_next_release:
                    self._suppress_next_release = False
                    return True
                if self._pick(mouse) is None:
                    try:
                        point = self._display_to_plane(mouse)
                    except ValueError:
                        return True
                    self._clicks.release_empty(
                        point,
                        self._emit_append,
                        lambda: self.maximize_requested.emit(self),
                    )
                    return True
                return True
            if mouse.button() == Qt.MouseButton.RightButton:
                return True
        if event_type == QEvent.Type.Wheel:
            wheel: QWheelEvent = event
            direction = wheel.angleDelta().y()
            if direction == 0:
                return True
            if wheel.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._zoom(direction)
            else:
                self._slice(direction)
            return True
        return super().eventFilter(watched, event)

