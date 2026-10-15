import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QToolBar,
    QToolButton,
    QWidget,
)

from NarrowDetecter.gui.dialogs import DicomSeriesDialog
from NarrowDetecter.gui.image_io import DicomSeriesInfo
from NarrowDetecter.gui.image_io import LoadedVolume
from NarrowDetecter.gui.models import ImageGeometry
from NarrowDetecter.gui.models import CPRSettings, ControlPath
from NarrowDetecter.gui.interactions import HitTarget


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

    def test_crosshair_uses_center_voxel_and_survives_maximize_restore(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        volume = self._volume()
        expected = volume.geometry.continuous_index_xyz_to_lps((2.0, 3.0, 3.0))

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

    def test_each_view_orientation_controls_transform_the_rendered_screen_axes(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        viewer.set_volume(self._volume())
        button_names = (
            ("rotate", "rotateClockwiseButton"),
            ("horizontal", "flipHorizontalButton"),
            ("vertical", "flipVerticalButton"),
            ("reset", "resetOrientationButton"),
        )
        for orientation, candidate in viewer.views.items():
            self.assertTrue(
                all(
                    candidate.findChild(QToolButton, object_name)
                    for _, object_name in button_names
                ),
                f"{orientation} view needs all orientation controls",
            )

        view = viewer.views["axial"]
        buttons = {
            name: view.findChild(QToolButton, object_name)
            for name, object_name in button_names
        }

        def screen_axes():
            camera = view.renderer.GetActiveCamera()
            up = np.asarray(camera.GetViewUp(), dtype=float)
            up /= np.linalg.norm(up)
            direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
            direction /= np.linalg.norm(direction)
            right = np.cross(direction, up)
            right /= np.linalg.norm(right)
            return right, up

        initial_right, initial_up = screen_axes()
        buttons["rotate"].click()
        rotated_right, rotated_up = screen_axes()
        np.testing.assert_allclose(rotated_right, initial_up, atol=1e-6)
        np.testing.assert_allclose(rotated_up, -initial_right, atol=1e-6)

        buttons["horizontal"].click()
        flipped_right, flipped_up = screen_axes()
        np.testing.assert_allclose(flipped_right, -rotated_right, atol=1e-6)
        np.testing.assert_allclose(flipped_up, rotated_up, atol=1e-6)

        buttons["vertical"].click()
        twice_flipped_right, twice_flipped_up = screen_axes()
        np.testing.assert_allclose(twice_flipped_right, flipped_right, atol=1e-6)
        np.testing.assert_allclose(twice_flipped_up, -flipped_up, atol=1e-6)

        buttons["reset"].click()
        reset_right, reset_up = screen_axes()
        np.testing.assert_allclose(reset_right, initial_right, atol=1e-6)
        np.testing.assert_allclose(reset_up, initial_up, atol=1e-6)
        viewer.close()

    def test_view_headers_do_not_show_anatomical_orientation_names(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        displayed_labels = {
            label.text()
            for view in viewer.views.values()
            for label in view.findChildren(QLabel)
        }

        self.assertTrue(
            displayed_labels.isdisjoint({"轴位", "冠状位", "矢状位"})
        )
        viewer.close()

    def test_window_level_tool_uses_middle_drag_and_syncs_all_views(self):
        from NarrowDetecter.gui.main_window import MainWindow

        window = MainWindow()
        window.resize(1200, 800)
        window.show()
        window.set_loaded_volume(self._volume())
        self.app.processEvents()
        button = window.findChild(QToolButton, "windowLevelToolButton")
        self.assertIsNotNone(button, "The toolbar needs a window/level tool button")
        toolbar = window.findChild(QToolBar)
        self.assertEqual(toolbar.findChildren(QDoubleSpinBox), [])

        button.click()
        self.assertTrue(button.isChecked())
        for view in window.viewer.views.values():
            self.assertEqual(
                view.vtk_widget.cursor().shape(), Qt.CursorShape.CrossCursor
            )

        axial = window.viewer.views["axial"]
        widget = axial.vtk_widget
        camera_before = np.asarray(
            axial.renderer.GetActiveCamera().GetPosition(), dtype=float
        )
        before_window = axial._image_actor.GetProperty().GetColorWindow()
        before_level = axial._image_actor.GetProperty().GetColorLevel()
        center = QPoint(widget.width() // 2, widget.height() // 2)
        QTest.mousePress(widget, Qt.MouseButton.MiddleButton, pos=center)
        QTest.mouseMove(widget, center + QPoint(60, -60), delay=10)
        QTest.mouseRelease(
            widget,
            Qt.MouseButton.MiddleButton,
            pos=center + QPoint(60, -60),
        )
        self.app.processEvents()

        updated = [
            (
                view._image_actor.GetProperty().GetColorWindow(),
                view._image_actor.GetProperty().GetColorLevel(),
            )
            for view in window.viewer.views.values()
        ]
        self.assertTrue(all(np.allclose(values, updated[0]) for values in updated))
        self.assertLess(updated[0][0], before_window)
        self.assertLess(updated[0][1], before_level)
        np.testing.assert_allclose(
            axial.renderer.GetActiveCamera().GetPosition(), camera_before, atol=1e-6
        )

        button.click()
        self.assertFalse(button.isChecked())
        for view in window.viewer.views.values():
            self.assertNotEqual(
                view.vtk_widget.cursor().shape(), Qt.CursorShape.CrossCursor
            )
        window.close()

    def test_loading_another_volume_exits_window_level_adjustment_mode(self):
        from NarrowDetecter.gui.main_window import MainWindow

        window = MainWindow()
        volume = self._volume()
        window.set_loaded_volume(volume)
        button = window.findChild(QToolButton, "windowLevelToolButton")
        button.click()
        self.assertTrue(button.isChecked())

        window.set_loaded_volume(volume)

        self.assertFalse(button.isChecked())
        for view in window.viewer.views.values():
            self.assertNotEqual(
                view.vtk_widget.cursor().shape(), Qt.CursorShape.CrossCursor
            )
        window.close()

    def test_transformed_screen_positions_map_back_to_the_same_lps_point(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer
        from NarrowDetecter.gui.vtk_utils import display_position_to_lps

        viewer = TriPlanarViewer()
        volume = self._volume()
        viewer.resize(900, 700)
        viewer.show()
        viewer.set_volume(volume)
        self.app.processEvents()
        view = viewer.views["axial"]
        expected = volume.geometry.continuous_index_xyz_to_lps((1.25, 4.25, 3.0))

        operations = (
            (view.rotate_clockwise,),
            (view.flip_horizontal,),
            (view.flip_vertical,),
            (view.rotate_clockwise, view.flip_horizontal, view.flip_vertical),
        )
        for sequence in operations:
            view.reset_orientation()
            for operation in sequence:
                operation()
            view.renderer.SetWorldPoint(*expected, 1.0)
            view.renderer.WorldToDisplay()
            display_x, display_y, _ = view.renderer.GetDisplayPoint()
            recovered = display_position_to_lps(
                view.renderer,
                (display_x, display_y),
                view.crosshair_lps,
                view._normal,
            )
            np.testing.assert_allclose(recovered, expected, atol=1e-6)
            recovered_index = volume.geometry.lps_to_continuous_index_xyz(recovered)
            self.assertAlmostEqual(recovered_index[2], 3.0, places=6)
        viewer.close()

    def test_view_orientation_is_independent_and_new_volume_resets_all_views(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        volume = self._volume()
        viewer.set_volume(volume)

        def screen_axes(view):
            camera = view.renderer.GetActiveCamera()
            up = np.asarray(camera.GetViewUp(), dtype=float)
            up /= np.linalg.norm(up)
            direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
            direction /= np.linalg.norm(direction)
            right = np.cross(direction, up)
            right /= np.linalg.norm(right)
            return right, up

        untouched_before = {
            name: screen_axes(viewer.views[name])
            for name in ("coronal", "sagittal")
        }
        viewer.views["axial"].rotate_clockwise()
        for name, expected in untouched_before.items():
            actual = screen_axes(viewer.views[name])
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
            np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)

        for view in viewer.views.values():
            view.flip_horizontal()
        viewer.set_volume(volume)
        for view in viewer.views.values():
            right, up = screen_axes(view)
            np.testing.assert_allclose(right, -view._axes[0], atol=1e-6)
            np.testing.assert_allclose(up, view._axes[1], atol=1e-6)
        viewer.close()

    @unittest.skipUnless(sys.platform == "win32", "Native VTK window test requires Windows")
    def test_closing_main_window_finalizes_native_vtk_windows_first(self):
        script = textwrap.dedent(
            """
            import numpy as np
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication
            from NarrowDetecter.gui.image_io import LoadedVolume
            from NarrowDetecter.gui.main_window import MainWindow
            from NarrowDetecter.gui.models import ImageGeometry

            app = QApplication([])
            geometry = ImageGeometry(
                (12, 12, 12), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
            volume = LoadedVolume(
                np.arange(12 ** 3, dtype=np.float32).reshape(12, 12, 12),
                None, geometry, "nifti", "synthetic",
            )
            window = MainWindow()
            window.resize(900, 700)
            window.show()
            window.set_loaded_volume(volume)
            QTimer.singleShot(300, window.close)
            raise SystemExit(app.exec())
            """
        )
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ, QT_QPA_PLATFORM="windows"),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output[-4000:])
        self.assertNotIn("wglMakeCurrent failed", output, output[-4000:])

    @unittest.skipUnless(sys.platform == "win32", "Native VTK window test requires Windows")
    def test_loaded_volume_renders_in_all_three_native_views(self):
        script = textwrap.dedent(
            """
            import numpy as np
            import ctypes
            from PySide6.QtCore import QPoint, Qt, QTimer
            from PySide6.QtTest import QTest
            from PySide6.QtWidgets import QApplication
            from NarrowDetecter.gui.image_io import LoadedVolume
            from NarrowDetecter.gui.models import ImageGeometry
            from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

            app = QApplication([])
            geometry = ImageGeometry(
                size_xyz=(16, 18, 20),
                spacing_xyz=(1.0, 1.0, 1.0),
                origin_lps_xyz=(0.0, 0.0, 0.0),
                direction_xyz=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
            z, y, x = np.mgrid[:20, :18, :16]
            volume = LoadedVolume(
                (x + y + 20 * z).astype(np.float32), None, geometry, "nifti", "synthetic"
            )
            viewer = TriPlanarViewer()
            viewer.resize(900, 700)
            viewer.show()
            viewer.set_volume(volume)

            def views_have_image():
                visible_image = True
                for view in viewer.views.values():
                    shot = app.primaryScreen().grabWindow(int(view.vtk_widget.winId())).toImage()
                    samples = [
                        shot.pixelColor(int(shot.width() * fraction), shot.height() // 3).red()
                        for fraction in (.25, .33, .4)
                    ]
                    visible_image &= max(samples) - min(samples) >= 5
                return all(
                    view.vtk_widget.GetEnabled()
                    and not view.vtk_widget.GetRenderWindow().GetNeverRendered()
                    for view in viewer.views.values()
                ) and visible_image

            def check_after_resize_and_slice():
                ready = views_have_image()
                print("ALL_THREE_RENDERED=" + str(ready), flush=True)
                viewer.close()
                app.exit(0 if ready else 1)

            def sample_axial():
                shot = app.primaryScreen().grabWindow(int(viewer.views["axial"].vtk_widget.winId())).toImage()
                return shot.pixelColor(shot.width() // 3, shot.height() // 3).red()

            def check_after_wheel(before_z, before_pixel):
                axial = viewer.views["axial"]
                if viewer.crosshair_lps[2] <= before_z or sample_axial() == before_pixel:
                    print("NATIVE_WHEEL_DID_NOT_CHANGE_SLICE", flush=True)
                    viewer.close()
                    app.exit(1)
                    return
                for view in viewer.views.values():
                    QTest.mouseDClick(view.vtk_widget, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
                    app.processEvents()
                    fills_center = (
                        viewer._maximized is view
                        and view.width() >= viewer.width() - 2
                        and view.height() >= viewer.height() - 2
                    )
                    if not fills_center:
                        print("MAXIMIZE_DID_NOT_FILL_CENTER", view.orientation, flush=True)
                        viewer.close()
                        app.exit(1)
                        return
                    QTest.mouseDClick(view.vtk_widget, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
                    app.processEvents()
                QTimer.singleShot(300, check_after_resize_and_slice)

            def check_initial():
                if not views_have_image():
                    print("INITIAL_RENDER_FAILED", flush=True)
                    viewer.close()
                    app.exit(1)
                    return
                axial = viewer.views["axial"]
                before_z = viewer.crosshair_lps[2]
                before_pixel = sample_axial()
                global_point = axial.vtk_widget.mapToGlobal(QPoint(100, 100))
                packed_point = ((global_point.y() & 0xffff) << 16) | (global_point.x() & 0xffff)
                ctypes.windll.user32.PostMessageW(int(axial.vtk_widget.winId()), 0x020A, 120 << 16, packed_point)
                QTimer.singleShot(300, lambda: check_after_wheel(before_z, before_pixel))

            QTimer.singleShot(300, check_initial)
            raise SystemExit(app.exec())
            """
        )
        environment = dict(os.environ, QT_QPA_PLATFORM="windows")
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ALL_THREE_RENDERED=True", result.stdout)

    @unittest.skipUnless(sys.platform == "win32", "Native VTK window test requires Windows")
    def test_middle_drag_repaints_native_image(self):
        script = textwrap.dedent(
            """
            import numpy as np
            from PySide6.QtCore import QPoint, Qt, QTimer
            from PySide6.QtTest import QTest
            from PySide6.QtWidgets import QApplication
            from NarrowDetecter.gui.image_io import LoadedVolume
            from NarrowDetecter.gui.models import ImageGeometry
            from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

            app = QApplication([])
            geometry = ImageGeometry(
                (30, 30, 30), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
            z, y, x = np.mgrid[:30, :30, :30]
            volume = LoadedVolume(
                (20 * x + 5 * y + 100 * z).astype(np.float32),
                None, geometry, "nifti", "synthetic",
            )
            viewer = TriPlanarViewer()
            viewer.resize(900, 700)
            viewer.show()
            viewer.set_volume(volume)
            viewer.set_window_level(700.0, 2000.0)
            viewer.activateWindow()
            view = viewer.views["axial"]
            view.rotate_clockwise()
            view.flip_horizontal()
            widget = view.vtk_widget
            camera = view.renderer.GetActiveCamera()

            def sample_pixels():
                shot = app.primaryScreen().grabWindow(int(widget.winId())).toImage()
                return tuple(
                    shot.pixelColor(int(shot.width() * horizontal),
                                    int(shot.height() * vertical)).red()
                    for horizontal in (.2, .3, .4, .5, .6, .7, .8)
                    for vertical in (.2, .3, .4, .5, .6, .7, .8)
                )

            def begin_drag():
                global before_pixels, before_camera
                before_pixels = sample_pixels()
                before_camera = np.asarray(camera.GetPosition())
                QTest.mousePress(widget, Qt.MouseButton.MiddleButton,
                                 pos=QPoint(200, 160))
                QTest.mouseMove(widget, QPoint(250, 190))
                QTest.mouseRelease(widget, Qt.MouseButton.MiddleButton,
                                   pos=QPoint(250, 190))
                QTimer.singleShot(300, check_drag)

            def check_drag():
                camera_moved = np.linalg.norm(
                    np.asarray(camera.GetPosition()) - before_camera
                ) > 1.0
                changed = sum(
                    left != right
                    for left, right in zip(before_pixels, sample_pixels())
                )
                print("PAN_CAMERA_MOVED=" + str(camera_moved),
                      "PAN_PIXELS_CHANGED=" + str(changed), flush=True)
                viewer.close()
                app.exit(0 if camera_moved and changed >= 5 else 1)

            QTimer.singleShot(500, begin_drag)
            raise SystemExit(app.exec())
            """
        )
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ, QT_QPA_PLATFORM="windows"),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-1500:] + result.stderr[-1500:])
        self.assertIn("PAN_CAMERA_MOVED=True", result.stdout)

    def test_slice_scrolling_is_clamped_to_volume_extent(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        volume = self._volume()
        viewer.set_volume(volume)
        axial = viewer.views["axial"]

        for _ in range(100):
            axial._slice(1)
        maximum = max(
            volume.geometry.continuous_index_xyz_to_lps((x, y, z))[2]
            for x in (0, volume.geometry.size_xyz[0] - 1)
            for y in (0, volume.geometry.size_xyz[1] - 1)
            for z in (0, volume.geometry.size_xyz[2] - 1)
        )
        self.assertLessEqual(viewer.crosshair_lps[2], maximum)

        for _ in range(200):
            axial._slice(-1)
        minimum = min(
            volume.geometry.continuous_index_xyz_to_lps((x, y, z))[2]
            for x in (0, volume.geometry.size_xyz[0] - 1)
            for y in (0, volume.geometry.size_xyz[1] - 1)
            for z in (0, volume.geometry.size_xyz[2] - 1)
        )
        self.assertGreaterEqual(viewer.crosshair_lps[2], minimum)

    def test_mouse_wheel_moves_about_one_voxel_in_oblique_volume(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        x_angle = np.deg2rad(2.0)
        z_angle = np.deg2rad(3.0)
        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x_angle), -np.sin(x_angle)],
            [0.0, np.sin(x_angle), np.cos(x_angle)],
        ])
        rz = np.array([
            [np.cos(z_angle), -np.sin(z_angle), 0.0],
            [np.sin(z_angle), np.cos(z_angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        geometry = ImageGeometry(
            size_xyz=(20, 20, 20),
            spacing_xyz=(0.5, 0.5, 0.75),
            origin_lps_xyz=(0.0, 0.0, 0.0),
            direction_xyz=tuple((rz @ rx).ravel()),
        )
        volume = LoadedVolume(
            np.zeros((20, 20, 20), dtype=np.float32),
            None,
            geometry,
            "nifti",
            "oblique.nii.gz",
        )
        viewer = TriPlanarViewer()
        viewer.set_volume(volume)

        for orientation, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
            before = viewer.crosshair_lps[axis]
            wheel = QWheelEvent(
                QPointF(100, 100), QPointF(100, 100), QPoint(0, 0),
                QPoint(0, 120), Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False,
            )
            QApplication.sendEvent(viewer.views[orientation].vtk_widget, wheel)
            step = viewer.crosshair_lps[axis] - before
            self.assertGreater(step, 0.4, orientation)
            self.assertLess(step, 0.9, orientation)
        viewer.close()

    def test_oblique_views_display_integer_native_slices_and_scroll_one_index(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        angle = np.deg2rad(12.0)
        direction = (
            np.cos(angle), 0.0, np.sin(angle),
            0.0, 1.0, 0.0,
            -np.sin(angle), 0.0, np.cos(angle),
        )
        geometry = ImageGeometry(
            size_xyz=(8, 10, 12),
            spacing_xyz=(0.5, 0.75, 1.5),
            origin_lps_xyz=(10.0, 20.0, 30.0),
            direction_xyz=direction,
        )
        volume = LoadedVolume(
            np.zeros((12, 10, 8), dtype=np.float32), None, geometry,
            "nifti", "oblique.nii.gz",
        )
        viewer = TriPlanarViewer()
        viewer.set_volume(volume)

        np.testing.assert_allclose(
            geometry.lps_to_continuous_index_xyz(viewer.crosshair_lps),
            (4.0, 5.0, 6.0),
            atol=1e-6,
        )
        for orientation, axis in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
            view = viewer.views[orientation]
            np.testing.assert_allclose(
                view._slice_plane.GetNormal(),
                geometry.direction_matrix[:, axis],
                atol=1e-6,
            )
            camera = view.renderer.GetActiveCamera()
            right = np.cross(camera.GetDirectionOfProjection(), camera.GetViewUp())
            expected_right_axis = {"axial": 0, "coronal": 0, "sagittal": 1}[orientation]
            self.assertLess(
                np.dot(right, geometry.direction_matrix[:, expected_right_axis]),
                -0.99,
            )
            before = np.asarray(geometry.lps_to_continuous_index_xyz(viewer.crosshair_lps))
            view._slice(1)
            after = np.asarray(geometry.lps_to_continuous_index_xyz(viewer.crosshair_lps))
            expected = before.copy()
            expected[axis] += 1.0
            np.testing.assert_allclose(after, expected, atol=1e-6)
            np.testing.assert_allclose(
                geometry.lps_to_continuous_index_xyz(view._slice_plane.GetOrigin())[axis],
                expected[axis],
                atol=1e-6,
            )
        viewer.close()

    def test_mirrored_views_map_screen_left_to_positive_array_index(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer
        from NarrowDetecter.gui.vtk_utils import display_position_to_lps

        viewer = TriPlanarViewer()
        volume = self._volume()
        viewer.resize(900, 700)
        viewer.show()
        viewer.set_volume(volume)
        self.app.processEvents()
        direction = volume.geometry.direction_matrix
        for orientation, horizontal_axis in (("axial", 0), ("coronal", 0), ("sagittal", 1)):
            view = viewer.views[orientation]
            renderer = view.renderer
            renderer.SetWorldPoint(*view.crosshair_lps, 1.0)
            renderer.WorldToDisplay()
            center_x, center_y, _ = renderer.GetDisplayPoint()
            point_lps = display_position_to_lps(
                renderer,
                (center_x - 10.0, center_y),
                view.crosshair_lps,
                view._normal,
            )
            displacement = np.asarray(point_lps) - np.asarray(view.crosshair_lps)
            self.assertGreater(
                np.dot(displacement, direction[:, horizontal_axis]),
                0.0,
                orientation,
            )
        viewer.close()

    def test_replacing_volume_resets_every_view_to_new_native_center(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        viewer.set_volume(self._volume())
        geometry = ImageGeometry(
            size_xyz=(8, 10, 12),
            spacing_xyz=(1.0, 1.0, 2.0),
            origin_lps_xyz=(100.0, 200.0, 300.0),
            direction_xyz=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        replacement = LoadedVolume(
            np.zeros((12, 10, 8), dtype=np.float32), None, geometry,
            "nifti", "replacement.nii.gz",
        )

        viewer.set_volume(replacement)

        expected = (4.0, 5.0, 6.0)
        for orientation, axis in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
            plane_origin = viewer.views[orientation]._slice_plane.GetOrigin()
            index = geometry.lps_to_continuous_index_xyz(plane_origin)
            self.assertAlmostEqual(index[axis], expected[axis])
        viewer.close()

    def test_all_views_share_one_vtk_scalar_buffer(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        viewer.set_volume(self._volume())

        vtk_images = tuple(view._vtk_image for view in viewer.views.values())
        self.assertEqual(len({id(image) for image in vtk_images}), 3)
        scalar_arrays = tuple(
            image.GetPointData().GetScalars()
            for image in vtk_images
        )
        self.assertEqual(len({id(scalars) for scalars in scalar_arrays}), 1)
        viewer.close()

    def test_control_point_marker_uses_a_fixed_screen_size(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        volume = self._volume()
        viewer.set_volume(volume)
        center = viewer.crosshair_lps
        path = ControlPath("active", "Active", [center])
        viewer.set_paths((path,), path.path_id)

        for view in viewer.views.values():
            actors = tuple(view._point_actors)
            self.assertEqual(len(actors), 1)
            point_property = actors[0].GetProperty()
            self.assertEqual(point_property.GetPointSize(), 12.0)
            self.assertTrue(point_property.GetRenderPointsAsSpheres())
        viewer.close()

    def test_crosshair_and_path_lines_are_rendered_and_inactive_points_are_hit_testable(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        volume = self._volume()
        viewer.set_volume(volume)
        center = viewer.crosshair_lps
        active = ControlPath("active", "Active", [center, (center[0] + 1.0, center[1], center[2])])
        inactive = ControlPath("inactive", "Inactive", [center])
        viewer.set_paths((active, inactive), active.path_id)

        axial = viewer.views["axial"]
        self.assertEqual(len(axial._crosshair_actors), 2)
        self.assertGreaterEqual(len(axial._path_line_actors), 1)
        inactive_actors = [
            actor
            for actor, target in axial._point_actors.items()
            if target.path_id == inactive.path_id
        ]
        self.assertTrue(inactive_actors)
        self.assertTrue(all(actor.GetPickable() for actor in inactive_actors))

    def test_double_click_maximizes_without_appending_a_point(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        viewer.set_volume(self._volume())
        appended = []
        viewer.point_append_requested.connect(lambda *args: appended.append(args))
        widget = viewer.views["axial"].vtk_widget
        position = QPoint(widget.width() // 2, widget.height() // 2)

        QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=position)
        QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, pos=position)
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=position)
        QTest.qWait(500)

        self.assertTrue(viewer.views["coronal"].isHidden())
        self.assertEqual(appended, [])
        viewer.close()

    def test_click_on_inactive_point_does_not_append_to_active_path(self):
        from NarrowDetecter.gui.tri_planar_viewer import TriPlanarViewer

        viewer = TriPlanarViewer()
        viewer.set_volume(self._volume())
        axial = viewer.views["axial"]
        appended = []
        viewer.point_append_requested.connect(lambda *args: appended.append(args))
        widget = axial.vtk_widget
        position = QPoint(widget.width() // 2, widget.height() // 2)

        with mock.patch.object(
            axial, "_pick", return_value=HitTarget("other", 0, False)
        ):
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=position)
            QTest.qWait(500)

        self.assertEqual(appended, [])
        viewer.close()


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

    def test_path_context_menu_requests_deletion_for_the_clicked_row(self):
        from NarrowDetecter.gui.path_panel import PathPanel

        panel = PathPanel()
        first = ControlPath("one", "First")
        second = ControlPath("two", "Second")
        panel.set_paths((first, second), "one")
        panel.show()
        self.app.processEvents()
        deleted = []
        activated = []
        panel.delete_requested.connect(deleted.append)
        panel.active_path_requested.connect(activated.append)
        position = panel.tree.visualItemRect(panel._items["two"]).center()

        panel.tree.customContextMenuRequested.emit(position)
        self.app.processEvents()
        menu = panel._context_menu
        delete_action = menu.actions()[0]
        QTest.mouseClick(
            menu,
            Qt.MouseButton.LeftButton,
            pos=menu.actionGeometry(delete_action).center(),
        )
        self.app.processEvents()

        self.assertEqual(deleted, ["two"])
        self.assertEqual(activated, [])
        panel.close()

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
    window_level_adjust_requested = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.volume = None
        self.paths = ()
        self.active_path_id = None
        self.events = []

    def set_volume(self, volume):
        self.volume = volume
        self.events.append(("volume", volume.source_path))

    def set_paths(self, paths, active_path_id):
        self.paths = tuple(paths)
        self.active_path_id = active_path_id
        self.events.append(("paths", len(self.paths), active_path_id))

    def set_window_level(self, window, level):
        self.window_level = (window, level)

    def set_window_level_adjustment_enabled(self, enabled):
        self.window_level_adjustment_enabled = bool(enabled)

    def set_crosshair_lps(self, point):
        self.crosshair_lps = tuple(point)


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
        self.assertEqual(window.viewer.crosshair_lps, (8.0, -2.6, 20.0))

        second = window.create_path("Second")
        window.viewer.point_append_requested.emit("axial", (7.0, -2.0, 21.0))
        self.assertEqual(len(second.control_points_lps_mm), 1)
        original = first.control_points_lps_mm[0]
        window.viewer.point_move_requested.emit(
            "axial", first.path_id, 0, (0.0, 0.0, original[2])
        )
        self.assertEqual(first.control_points_lps_mm[0], original)

    def test_replacing_volume_clears_old_point_actors_before_loading(self):
        window = self._window()
        volume = TriPlanarViewerTests._volume()
        window.set_loaded_volume(volume)
        window.viewer.point_append_requested.emit(
            "axial", volume.geometry.continuous_index_xyz_to_lps((2.0, 3.0, 3.0))
        )
        window.viewer.events.clear()

        replacement = LoadedVolume(
            array_zyx=volume.array_zyx.copy(),
            image_itk=None,
            geometry=volume.geometry,
            source_kind="nifti",
            source_path="replacement.nii.gz",
        )
        window.set_loaded_volume(replacement)

        self.assertGreaterEqual(len(window.viewer.events), 2)
        self.assertEqual(window.viewer.events[0], ("paths", 0, None))
        self.assertEqual(window.viewer.events[1], ("volume", "replacement.nii.gz"))

    def test_display_window_percentiles_use_a_bounded_sample(self):
        from NarrowDetecter.gui import main_window as main_window_module

        window = self._window()
        geometry = ImageGeometry(
            size_xyz=(200, 200, 50),
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_lps_xyz=(0.0, 0.0, 0.0),
            direction_xyz=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        volume = LoadedVolume(
            array_zyx=np.linspace(0.0, 100.0, 2_000_000, dtype=np.float32).reshape(
                50, 200, 200
            ),
            image_itk=None,
            geometry=geometry,
            source_kind="nifti",
            source_path="large.nii.gz",
        )
        original_percentile = np.percentile
        sample_sizes = []

        def record_percentile(values, *args, **kwargs):
            sample_sizes.append(np.asarray(values).size)
            return original_percentile(values, *args, **kwargs)

        with mock.patch.object(
            main_window_module.np,
            "percentile",
            side_effect=record_percentile,
        ):
            window.set_loaded_volume(volume)

        self.assertEqual(len(sample_sizes), 1)
        self.assertLessEqual(sample_sizes[0], 1_000_000)

    def test_dragging_point_on_oblique_slice_preserves_array_layer(self):
        angle = np.deg2rad(20.0)
        geometry = ImageGeometry(
            size_xyz=(30, 30, 30),
            spacing_xyz=(0.5, 0.75, 1.5),
            origin_lps_xyz=(10.0, 20.0, 30.0),
            direction_xyz=(
                np.cos(angle), 0.0, np.sin(angle),
                0.0, 1.0, 0.0,
                -np.sin(angle), 0.0, np.cos(angle),
            ),
        )
        volume = LoadedVolume(
            np.zeros((30, 30, 30), dtype=np.float32), None, geometry,
            "nifti", "oblique.nii.gz",
        )
        window = self._window()
        window.set_loaded_volume(volume)
        path = window.path_store.active_path
        path.control_points_lps_mm.append(
            geometry.continuous_index_xyz_to_lps((4.0, 5.0, 6.0))
        )
        candidate = geometry.continuous_index_xyz_to_lps((9.0, 11.0, 13.0))

        window.viewer.point_move_requested.emit("axial", path.path_id, 0, candidate)

        np.testing.assert_allclose(
            geometry.lps_to_continuous_index_xyz(path.control_points_lps_mm[0]),
            (9.0, 11.0, 6.0),
            atol=1e-6,
        )

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

    def test_save_decision_persists_every_dirty_path_before_replacing_image(self):
        window = self._window()
        window.set_loaded_volume(TriPlanarViewerTests._volume())
        first = window.path_store.active_path
        first.control_points_lps_mm[:] = [(8.0, -2.6, 20.0), (7.0, -2.0, 21.0)]
        first.dirty = True
        second = window.create_path("Second")
        second.control_points_lps_mm[:] = [(8.0, -2.6, 20.0), (7.0, -2.0, 21.0)]
        second.dirty = True

        with tempfile.TemporaryDirectory() as directory:
            outputs = [
                (str(Path(directory) / "first.json"), "JSON"),
                (str(Path(directory) / "second.json"), "JSON"),
            ]
            with mock.patch.object(
                QMessageBox,
                "warning",
                return_value=QMessageBox.StandardButton.Save,
            ), mock.patch.object(
                QFileDialog,
                "getSaveFileName",
                side_effect=outputs,
            ):
                accepted = window._confirm_dirty_paths()

            self.assertTrue(accepted)
            self.assertTrue(all(not path.dirty for path in (first, second)))
            self.assertTrue((Path(directory) / "first.json").is_file())
            self.assertTrue((Path(directory) / "second.json").is_file())

    def test_importing_path_saves_panel_settings_to_previous_active_path(self):
        window = self._window()
        volume = TriPlanarViewerTests._volume()
        window.set_loaded_volume(volume)
        previous = window.path_store.active_path
        window.cpr_panel.mode.setCurrentIndex(
            window.cpr_panel.mode.findData("straightened")
        )
        imported = ControlPath(
            "source",
            "Imported",
            [(8.0, -2.6, 20.0), (7.0, -2.0, 21.0)],
        )

        with mock.patch(
            "NarrowDetecter.gui.main_window.read_centerline_json",
            return_value=(imported, volume.geometry),
        ):
            count = window.import_centerlines(("path.json",))

        self.assertEqual(count, 1)
        self.assertEqual(previous.cpr_settings.mode, "straightened")

    def test_busy_state_disables_drag_and_drop_image_replacement(self):
        window = self._window()
        self.assertTrue(window.acceptDrops())
        window._set_busy(True)
        self.assertFalse(window.acceptDrops())
        window._set_busy(False)
        self.assertTrue(window.acceptDrops())

    def test_confirmed_path_deletion_removes_the_target_and_selects_neighbor(self):
        window = self._window()
        window.set_loaded_volume(TriPlanarViewerTests._volume())
        first = window.path_store.active_path
        second = window.create_path("Second")

        with mock.patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window.path_panel.delete_requested.emit(second.path_id)

        self.assertEqual(
            tuple(path.path_id for path in window.path_store.paths),
            (first.path_id,),
        )
        self.assertEqual(window.path_store.active_path_id, first.path_id)
        self.assertEqual(window.viewer.active_path_id, first.path_id)

    def test_deleting_inactive_path_preserves_unsaved_active_panel_settings(self):
        window = self._window()
        window.set_loaded_volume(TriPlanarViewerTests._volume())
        first = window.path_store.active_path
        second = window.create_path("Second")
        window.cpr_panel.mode.setCurrentIndex(
            window.cpr_panel.mode.findData("straightened")
        )

        with mock.patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window.path_panel.delete_requested.emit(first.path_id)

        self.assertEqual(window.path_store.active_path_id, second.path_id)
        self.assertEqual(window.cpr_panel.settings().mode, "straightened")

    def test_completed_output_path_uses_bottom_status_without_resizing_panels(self):
        window = self._window()
        window.show()
        self.app.processEvents()
        before = tuple(window.splitter.sizes())
        output = "E:\\" + "very-long-output-directory\\" * 20

        window._worker_completed(output)
        self.app.processEvents()

        after = tuple(window.splitter.sizes())
        self.assertEqual(window.cpr_panel.status_label.text(), "完成")
        self.assertEqual(window.statusBar().currentMessage(), f"CPR 已完成：{output}")
        self.assertEqual(after, before)
        window.close()


if __name__ == "__main__":
    unittest.main()
