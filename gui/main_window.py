from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QToolBar,
    QToolButton,
)

from .centerline_io import geometry_matches, read_centerline_json, write_centerline_json
from .cpr_adapter import ManualCPRRequest, sanitized_path_name
from .cpr_panel import CPRPanel
from .cpr_worker import CPRWorker
from .dialogs import DicomSeriesDialog
from .image_io import DicomSeriesSelectionRequired, LoadedVolume, load_image
from .models import PathStore
from .path_panel import PathPanel
from .tri_planar_viewer import TriPlanarViewer


class MainWindow(QMainWindow):
    def __init__(
        self,
        parent=None,
        *,
        viewer_factory: Callable[[], object] = TriPlanarViewer,
        image_loader: Callable[..., LoadedVolume] = load_image,
        worker_factory: Callable[[ManualCPRRequest], CPRWorker] = CPRWorker,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NarrowDetecter CPR")
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        self._image_loader = image_loader
        self._worker_factory = worker_factory
        self.volume: LoadedVolume | None = None
        self.path_store = PathStore()
        self._thread: QThread | None = None
        self._worker: CPRWorker | None = None
        self._running_path_id: str | None = None
        self._display_window = 1.0
        self._display_level = 0.0

        self.path_panel = PathPanel(self)
        self.viewer = viewer_factory()
        self.cpr_panel = CPRPanel(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.path_panel)
        self.splitter.addWidget(self.viewer)
        self.splitter.addWidget(self.cpr_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([240, 1000, 300])
        self.setCentralWidget(self.splitter)

        self._create_actions()
        self._create_toolbar()
        self._connect_signals()
        self.statusBar().showMessage("请导入 NIfTI 或 DICOM 影像")

    def _create_actions(self) -> None:
        self.import_nifti_action = QAction("导入 NIfTI…", self)
        self.import_dicom_action = QAction("导入 DICOM 文件夹…", self)
        self.import_centerline_action = QAction("导入中心线 JSON…", self)
        self.export_centerline_action = QAction("导出当前中心线 JSON…", self)
        self.import_nifti_action.triggered.connect(self._choose_nifti)
        self.import_dicom_action.triggered.connect(self._choose_dicom)
        self.import_centerline_action.triggered.connect(self._choose_centerlines)
        self.export_centerline_action.triggered.connect(self._export_active_centerline)

        file_menu = self.menuBar().addMenu("文件")
        import_menu = file_menu.addMenu("导入影像")
        import_menu.addAction(self.import_nifti_action)
        import_menu.addAction(self.import_dicom_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_centerline_action)
        file_menu.addAction(self.export_centerline_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        self.addToolBar(toolbar)
        import_button = QToolButton(toolbar)
        import_button.setText("导入影像")
        import_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(import_button)
        menu.addAction(self.import_nifti_action)
        menu.addAction(self.import_dicom_action)
        import_button.setMenu(menu)
        toolbar.addWidget(import_button)
        toolbar.addAction(self.import_centerline_action)
        toolbar.addAction(self.export_centerline_action)
        toolbar.addSeparator()
        self.window_level_button = QToolButton(toolbar)
        self.window_level_button.setObjectName("windowLevelToolButton")
        self.window_level_button.setText("亮度/对比度")
        self.window_level_button.setCheckable(True)
        self.window_level_button.setToolTip(
            "启用后按住中键：上下调整亮度，左右调整对比度"
        )
        self.window_level_button.toggled.connect(
            self._set_window_level_adjustment_enabled
        )
        toolbar.addWidget(self.window_level_button)

    def _connect_signals(self) -> None:
        self.path_panel.create_requested.connect(self.create_path)
        self.path_panel.active_path_requested.connect(self.set_active_path)
        self.path_panel.rename_requested.connect(self._rename_path)
        self.path_panel.visibility_changed.connect(self._set_path_visibility)
        self.path_panel.delete_requested.connect(self._delete_path)
        self.viewer.point_append_requested.connect(self._append_point)
        self.viewer.point_move_requested.connect(self._move_point)
        self.viewer.point_delete_requested.connect(self._delete_point)
        self.viewer.path_clear_requested.connect(self._clear_path)
        self.viewer.window_level_adjust_requested.connect(
            self._on_window_level_adjust_requested
        )
        self.cpr_panel.start_requested.connect(self.start_cpr)
        self.cpr_panel.cancel_requested.connect(self.cancel_cpr)

    def _refresh_paths(self) -> None:
        paths = self.path_store.paths
        active = self.path_store.active_path_id
        self.path_panel.set_paths(paths, active)
        self.viewer.set_paths(paths, active)

    def _store_current_settings(self) -> bool:
        path = self.path_store.active_path
        if path is None:
            return True
        try:
            settings = self.cpr_panel.settings()
        except ValueError:
            return False
        if path.cpr_settings != settings:
            path.cpr_settings = settings
            path.dirty = True
        return True

    def create_path(self, name: str | None = None):
        if not self._store_current_settings():
            return None
        try:
            path = self.path_store.create_path(name)
        except ValueError as error:
            self._show_error(str(error))
            return None
        self.cpr_panel.set_settings(path.cpr_settings)
        self._refresh_paths()
        return path

    def set_active_path(self, path_id: str) -> None:
        old_id = self.path_store.active_path_id
        if old_id == path_id:
            return
        if not self._store_current_settings():
            self._refresh_paths()
            return
        self.path_store.set_active(path_id)
        self.cpr_panel.set_settings(self.path_store.active_path.cpr_settings)
        self._refresh_paths()

    def _rename_path(self, path_id: str, name: str) -> None:
        try:
            self.path_store.rename_path(path_id, name)
        except ValueError as error:
            self._show_error(str(error))
        self._refresh_paths()

    def _set_path_visibility(self, path_id: str, visible: bool) -> None:
        self.path_store.set_visibility(path_id, visible)
        self.viewer.set_paths(self.path_store.paths, self.path_store.active_path_id)

    def _delete_path(self, path_id: str) -> None:
        try:
            path = self.path_store.get_path(path_id)
        except KeyError:
            return
        answer = QMessageBox.question(
            self,
            "删除路径",
            f"确定删除路径“{path.name}”及其全部中心点吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        deleted_active_path = path_id == self.path_store.active_path_id
        self.path_store.delete_path(path_id)
        active_path = self.path_store.active_path
        if deleted_active_path and active_path is not None:
            self.cpr_panel.set_settings(active_path.cpr_settings)
        self._refresh_paths()

    def _append_point(self, orientation: str, point_lps) -> None:
        del orientation
        if self.volume is None or self.path_store.active_path is None:
            return
        if not self.volume.geometry.contains_lps(point_lps):
            self.cpr_panel.set_status("点位于影像范围之外")
            return
        self.path_store.append_point(point_lps)
        self.viewer.set_crosshair_lps(point_lps)
        self._refresh_paths()

    def _move_point(
        self,
        orientation: str,
        path_id: str,
        point_index: int,
        point_lps,
    ) -> None:
        if self.volume is None:
            return
        try:
            self.path_store.move_point_in_view(
                path_id, point_index, orientation, point_lps, self.volume.geometry
            )
        except (PermissionError, IndexError, KeyError):
            return
        self._refresh_paths()

    def _delete_point(self, path_id: str, point_index: int) -> None:
        try:
            self.path_store.delete_point(path_id, point_index)
        except (PermissionError, IndexError, KeyError):
            return
        self._refresh_paths()

    def _clear_path(self, path_id: str) -> None:
        if path_id != self.path_store.active_path_id:
            return
        self.path_store.clear_active_path()
        self._refresh_paths()

    def set_loaded_volume(self, volume: LoadedVolume) -> None:
        self.window_level_button.setChecked(False)
        self._set_window_level_adjustment_enabled(False)
        self.volume = volume
        self.path_store = PathStore()
        self._refresh_paths()
        self.viewer.set_volume(volume)
        values = np.asarray(volume.array_zyx).reshape(-1)
        maximum_samples = 1_000_000
        stride = max(1, (values.size + maximum_samples - 1) // maximum_samples)
        sample = values[::stride]
        finite = sample[np.isfinite(sample)]
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.0))
            if not np.isfinite(high - low) or high <= low:
                low = float(finite.min()) - 0.5
                high = float(finite.max()) + 0.5
        else:
            low, high = 0.0, 1.0
        self._display_window = max(0.001, float(high - low))
        self._display_level = float((high + low) * 0.5)
        self._apply_window_level()
        self.create_path()
        self.statusBar().showMessage(f"已载入：{volume.source_path}")

    def _apply_window_level(self) -> None:
        self.viewer.set_window_level(
            self._display_window, self._display_level
        )

    def _set_window_level_adjustment_enabled(self, enabled: bool) -> None:
        self.viewer.set_window_level_adjustment_enabled(bool(enabled))
        if enabled:
            self.statusBar().showMessage(
                "亮度/对比度调整模式：中键上下调整亮度，左右调整对比度"
            )
        else:
            self.statusBar().showMessage("普通模式：中键拖动平移")

    def _on_window_level_adjust_requested(
        self,
        window: float,
        level: float,
    ) -> None:
        if not np.isfinite(window) or not np.isfinite(level):
            return
        self._display_window = max(0.001, float(window))
        self._display_level = float(level)
        self._apply_window_level()
        self.statusBar().showMessage(
            f"亮度/对比度调整：窗宽 {self._display_window:.2f}，"
            f"窗位 {self._display_level:.2f}"
        )

    def _choose_nifti(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self,
            "导入 NIfTI",
            "",
            "NIfTI (*.nii *.nii.gz);;所有文件 (*)",
        )
        if source:
            self.import_image_source(source)

    def _choose_dicom(self) -> None:
        source = QFileDialog.getExistingDirectory(self, "导入 DICOM 文件夹")
        if source:
            self.import_image_source(source)

    def import_image_source(self, source: str | Path) -> bool:
        if self._worker is not None:
            self.cpr_panel.set_status("请先取消或等待当前 CPR 任务结束")
            return False
        if not self._confirm_dirty_paths():
            return False
        try:
            volume = self._image_loader(source)
        except DicomSeriesSelectionRequired as required:
            dialog = DicomSeriesDialog(required.series, self)
            if dialog.exec() != dialog.DialogCode.Accepted:
                return False
            uid = dialog.selected_series_uid()
            if uid is None:
                return False
            try:
                volume = self._image_loader(source, series_uid=uid)
            except Exception as error:
                self._show_error(f"影像导入失败：{error}")
                return False
        except Exception as error:
            self._show_error(f"影像导入失败：{error}")
            return False
        self.set_loaded_volume(volume)
        return True

    def _choose_centerlines(self) -> None:
        sources, _ = QFileDialog.getOpenFileNames(
            self,
            "导入中心线 JSON",
            "",
            "中心线 JSON (*.json)",
        )
        if sources:
            self.import_centerlines(sources)

    def import_centerlines(self, sources) -> int:
        if self.volume is None:
            self._show_error("请先导入对应的影像。")
            return 0
        if not self._store_current_settings():
            return 0
        imported = 0
        for source in sources:
            try:
                path, geometry = read_centerline_json(source)
                if not geometry_matches(geometry, self.volume.geometry):
                    answer = QMessageBox.question(
                        self,
                        "影像几何不一致",
                        f"{Path(source).name} 的影像几何与当前影像不一致，仍要导入吗？",
                        QMessageBox.StandardButton.Yes
                        | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        continue
                added = self.path_store.add_imported_path(path)
                self.cpr_panel.set_settings(added.cpr_settings)
                imported += 1
            except Exception as error:
                self._show_error(f"中心线导入失败：{error}")
        self._refresh_paths()
        return imported

    def _export_active_centerline(self) -> bool:
        path = self.path_store.active_path
        if self.volume is None or path is None:
            self._show_error("没有可导出的中心线。")
            return False
        if not self._store_current_settings():
            return False
        return self._export_path_centerline(path)

    def _export_path_centerline(self, path) -> bool:
        if self.volume is None:
            self._show_error("没有可导出的中心线。")
            return False
        destination, _ = QFileDialog.getSaveFileName(
            self,
            f"导出中心线：{path.name}",
            f"{path.name}_centerline.json",
            "中心线 JSON (*.json)",
        )
        if not destination:
            return False
        try:
            write_centerline_json(path, self.volume.geometry, destination)
        except Exception as error:
            self._show_error(f"中心线导出失败：{error}")
            return False
        path.dirty = False
        self._refresh_paths()
        return True

    def _confirm_dirty_paths(self) -> bool:
        if not self._store_current_settings():
            return False
        dirty = [path for path in self.path_store.paths if path.dirty]
        if not dirty:
            return True
        answer = QMessageBox.warning(
            self,
            "尚未保存的中心线",
            f"当前影像包含 {len(dirty)} 条尚未保存的路径。是否逐条保存？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return all(self._export_path_centerline(path) for path in dirty)
        return True

    def start_cpr(self) -> None:
        if self._worker is not None:
            return
        if self.volume is None or self.path_store.active_path is None:
            self._show_error("请先导入影像并创建路径。")
            return
        if not self._store_current_settings():
            return
        path = self.path_store.active_path
        if len(path.control_points_lps_mm) < 2:
            self._show_error("至少需要两个中心点才能生成 CPR。")
            return
        output_root = self.cpr_panel.output_directory()
        if output_root is None:
            selected = QFileDialog.getExistingDirectory(self, "选择 CPR 输出目录")
            if not selected:
                return
            output_root = Path(selected)
            self.cpr_panel.set_output_directory(output_root)
        target = output_root / sanitized_path_name(path.name)
        overwrite = False
        if target.exists():
            answer = QMessageBox.question(
                self,
                "覆盖已有结果",
                f"输出目录已存在：\n{target}\n\n是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        frozen_path = copy.deepcopy(path)
        request = ManualCPRRequest(
            volume=self.volume,
            path=frozen_path,
            output_root=Path(output_root),
            overwrite=overwrite,
        )
        worker = self._worker_factory(request)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.cpr_panel.set_progress)
        worker.completed.connect(self._worker_completed)
        worker.cancelled.connect(self._worker_cancelled)
        worker.failed.connect(self._worker_failed)
        worker.completed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._worker_thread_finished)
        self._worker = worker
        self._thread = thread
        self._running_path_id = path.path_id
        self._set_busy(True)
        self.cpr_panel.set_progress(0, len(path.cpr_settings.angles_degrees()))
        self.cpr_panel.set_status("正在生成 CPR…")
        thread.start()

    def cancel_cpr(self) -> None:
        if self._worker is not None:
            self.cpr_panel.set_status("正在取消…")
            self._worker.cancel()

    def _worker_completed(self, output: str) -> None:
        if self._running_path_id is not None:
            try:
                self.path_store.mark_exported(self._running_path_id)
            except KeyError:
                pass
        self.cpr_panel.set_status("完成")
        self.statusBar().showMessage(f"CPR 已完成：{output}")
        self._refresh_paths()
        self._set_busy(False)

    def _worker_cancelled(self) -> None:
        self.cpr_panel.set_status("已取消，原有完整结果未改变")
        self._set_busy(False)

    def _worker_failed(self, message: str) -> None:
        self.cpr_panel.set_status(f"失败：{message}")
        self._set_busy(False)

    def _worker_thread_finished(self) -> None:
        thread = self._thread
        self._worker = None
        self._thread = None
        self._running_path_id = None
        if thread is not None:
            thread.deleteLater()

    def _set_busy(self, busy: bool) -> None:
        self.cpr_panel.set_busy(busy)
        self.path_panel.setEnabled(not busy)
        self.viewer.setEnabled(not busy)
        self.import_nifti_action.setEnabled(not busy)
        self.import_dicom_action.setEnabled(not busy)
        self.import_centerline_action.setEnabled(not busy)
        self.setAcceptDrops(not busy)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "NarrowDetecter CPR", str(message))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1:
            source = urls[0].toLocalFile()
            if source:
                self.import_image_source(source)
                event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None:
            self._show_error("请先取消或等待当前 CPR 任务结束。")
            event.ignore()
            return
        if self._confirm_dirty_paths():
            self.viewer.close()
            event.accept()
        else:
            event.ignore()

