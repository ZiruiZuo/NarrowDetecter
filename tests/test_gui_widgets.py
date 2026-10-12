import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox

from NarrowDetecter.gui.dialogs import DicomSeriesDialog
from NarrowDetecter.gui.image_io import DicomSeriesInfo


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


if __name__ == "__main__":
    unittest.main()
