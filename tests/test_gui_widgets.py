import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox

from NarrowDetecter.gui.dialogs import DicomSeriesDialog
from NarrowDetecter.gui.image_io import DicomSeriesInfo
from NarrowDetecter.gui.image_io import LoadedVolume
from NarrowDetecter.gui.models import ImageGeometry


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


if __name__ == "__main__":
    unittest.main()
