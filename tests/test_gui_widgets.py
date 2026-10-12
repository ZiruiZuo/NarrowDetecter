import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QWidget

from NarrowDetecter.gui.dialogs import DicomSeriesDialog
from NarrowDetecter.gui.image_io import DicomSeriesInfo
from NarrowDetecter.gui.image_io import LoadedVolume
from NarrowDetecter.gui.models import ImageGeometry
from NarrowDetecter.gui.models import CPRSettings, ControlPath


class DicomSeriesDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_multiple_series_remain_separate_and_require_selection(self):
        dialog = DicomSeriesDialog(
            (
                DicomSeriesInfo("one", "First", "1", 20),
                DicomSeriesInfo("two", "Second", "2", 30),
            )
        )
        ok_button = dialog.button_box.button(QDialogButtonBox.Ok)

        self.assertEqual(dialog.table.rowCount(), 2)
        self.assertFalse(ok_button.isEnabled())
        self.assertIsNone(dialog.selected_series_uid())

        dialog.table.selectRow(1)
        self.app.processEvents()

        self.assertTrue(ok_button.isEnabled())
        self.assertEqual(dialog.selected_series_uid(), "two")


class TriPlanarViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _volume():
        geometry = ImageGeometry(
            size_xyz=(5, 6, 7),
            spacing_xyz=(0.7, 1.1, 1.8),
            origin_lps_xyz=(10.0, -4.0, 20.0),
            direction_xyz=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        return LoadedVolume(
            array_zyx=np.zeros((7, 6, 5), dtype=np.float32),
            image_itk=None,
            geometry=geometry,
            source_kind="nifti",
            source_path="synthetic.nii.gz",
        )

    def test_crosshair_uses_physical_center_and_survives_maximize_restore(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        volume = self._volume()
        expected = volume.geometry.continuous_index_xyz_to_lps((2.0, 2.5, 3.0))

        viewer.set_volume(volume)
        self.assertTrue(np.allclose(viewer.crosshair_lps, expected))
        original_views = tuple(viewer.views.values())

        viewer.toggle_maximize(viewer.views["axial"])
        self.assertFalse(viewer.views["axial"].isHidden())
        self.assertTrue(viewer.views["coronal"].isHidden())
        viewer.toggle_maximize(viewer.views["axial"])

        self.assertEqual(tuple(viewer.views.values()), original_views)
        self.assertTrue(all(not view.isHidden() for view in viewer.views.values()))
        self.assertTrue(np.allclose(viewer.crosshair_lps, expected))


class ControlPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_path_visibility_and_activation_are_separate_actions(self):
        from NarrowDetecter.gui.path_panel import PathPanel

        panel = PathPanel()
        first = ControlPath("one", "First")
        second = ControlPath("two", "Second")
        panel.set_paths((first, second), "one")
        activated = []
        visibility = []
        panel.active_path_requested.connect(activated.append)
        panel.visibility_changed.connect(
            lambda path_id, visible: visibility.append((path_id, visible))
        )

        panel.set_path_visibility("two", False)
        self.assertEqual(visibility, [("two", False)])
        self.assertEqual(activated, [])

        panel.activate_path("two")
        self.assertEqual(activated, ["two"])

    def test_cpr_panel_exposes_only_common_settings_with_twelve_defaults(self):
        from NarrowDetecter.gui.cpr_panel import CPRPanel

        panel = CPRPanel()
        settings = panel.settings()

        self.assertEqual(settings.mode, "stretched")
        self.assertEqual(len(settings.angles_degrees()), 12)
        self.assertFalse(hasattr(panel, "start_extension"))
        self.assertFalse(hasattr(panel, "smoothing_sigma"))
        panel.set_settings(CPRSettings(mode="straightened", show_centerline=False))
        restored = panel.settings()
        self.assertEqual(restored.mode, "straightened")
        self.assertFalse(restored.show_centerline)


class _FakeViewer(QWidget):
    point_append_requested = Signal(str, object)
    point_move_requested = Signal(str, str, int, object)
    point_delete_requested = Signal(str, int)
    path_clear_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.volume = None
        self.paths = ()
        self.active_path_id = None

    def set_volume(self, volume):
        self.volume = volume

    def set_paths(self, paths, active_path_id):
        self.paths = tuple(paths)
        self.active_path_id = active_path_id

    def set_window_level(self, window, level):
        self.window_level = (window, level)


class MainWindowStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window(self):
        from NarrowDetecter.gui.main_window import MainWindow

        return MainWindow(viewer_factory=_FakeViewer)

    def test_viewer_edits_only_the_active_path(self):
        window = self._window()
        window.set_loaded_volume(TriPlanarViewerTests._volume())
        first = window.path_store.active_path
        window.viewer.point_append_requested.emit("axial", (8.0, -2.6, 20.0))
        self.assertEqual(len(first.control_points_lps_mm), 1)

        second = window.create_path("Second")
        window.viewer.point_append_requested.emit("axial", (7.0, -2.0, 21.0))
        self.assertEqual(len(second.control_points_lps_mm), 1)
        original = first.control_points_lps_mm[0]
        window.viewer.point_move_requested.emit(
            "axial", first.path_id, 0, (0.0, 0.0, original[2])
        )
        self.assertEqual(first.control_points_lps_mm[0], original)

    def test_path_switch_saves_and_restores_per_path_settings(self):
        window = self._window()
        window.set_loaded_volume(TriPlanarViewerTests._volume())
        first = window.path_store.active_path
        window.cpr_panel.mode.setCurrentIndex(
            window.cpr_panel.mode.findData("straightened")
        )
        second = window.create_path("Second")
        self.assertEqual(first.cpr_settings.mode, "straightened")

        window.cpr_panel.mode.setCurrentIndex(
            window.cpr_panel.mode.findData("stretched")
        )
        window.set_active_path(first.path_id)
        self.assertEqual(window.cpr_panel.settings().mode, "straightened")
        self.assertEqual(second.cpr_settings.mode, "stretched")


if __name__ == "__main__":
    unittest.main()
