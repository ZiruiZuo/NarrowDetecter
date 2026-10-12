from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk

from NarrowDetecter.gui.cpr_adapter import ManualCPRRequest, export_manual_cpr
from NarrowDetecter.gui.image_io import load_nifti
from NarrowDetecter.gui.models import CPRSettings, ControlPath


class LauncherTests(unittest.TestCase):
    def test_main_creates_and_shows_one_window(self):
        from NarrowDetecter.gui import __main__ as launcher

        application = mock.Mock()
        application.exec.return_value = 17
        window = mock.Mock()
        with mock.patch.object(launcher, "QApplication", return_value=application), mock.patch.object(
            launcher, "MainWindow", return_value=window
        ):
            result = launcher.main(["narrowdetecter-cpr"])

        self.assertEqual(result, 17)
        window.show.assert_called_once_with()


class RealManualCPRIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_volume(path: Path):
        z, y, x = np.mgrid[:12, :14, :16]
        array = (0.1 * x + 0.2 * y + 0.3 * z).astype(np.float32)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.8, 1.1, 1.5))
        image.SetOrigin((5.0, -8.0, 12.0))
        image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(image, str(path))

    def test_real_loader_generation_rendering_and_json_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "synthetic.nii.gz"
            self._write_volume(image_path)
            volume = load_nifti(image_path)
            indices = ((4.0, 4.0, 2.0), (5.0, 5.0, 4.0), (7.0, 6.0, 6.0), (9.0, 7.0, 8.0))
            points = [
                volume.geometry.continuous_index_xyz_to_lps(index)
                for index in indices
            ]

            for mode, show_centerline in (
                ("stretched", True),
                ("straightened", False),
            ):
                path = ControlPath(
                    path_id=mode,
                    name=f"Manual_{mode}",
                    control_points_lps_mm=list(points),
                    dirty=True,
                    cpr_settings=CPRSettings(
                        mode=mode,
                        sampling_line_length_mm=8.0,
                        slice_resolution_mm=2.0,
                        show_centerline=show_centerline,
                    ),
                )
                output = export_manual_cpr(
                    ManualCPRRequest(volume, path, root / mode),
                    is_cancelled=lambda: False,
                    on_progress=lambda done, total: None,
                )
                pngs = tuple(output.glob("*.png"))
                self.assertEqual(len(pngs), 12)
                self.assertTrue(all(item.stat().st_size > 0 for item in pngs))
                self.assertTrue(all(f"_{mode}_CPR_" in item.name for item in pngs))
                document = json.loads(next(output.glob("*_centerline.json")).read_text("utf-8"))
                self.assertEqual(document["control_points_lps_mm"], [list(point) for point in points])
                self.assertEqual(document["fit_settings"]["start_extension_mm"], 0.0)
                self.assertEqual(document["fit_settings"]["end_extension_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
