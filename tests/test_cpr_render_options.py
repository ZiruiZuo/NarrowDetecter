import tempfile
import unittest
from unittest import mock

import numpy as np

from NarrowDetecter.models import CPRResult
from NarrowDetecter.visualization import cpr as cpr_visualization


class CPRRenderOptionsTests(unittest.TestCase):
    def _result(self, mode: str) -> CPRResult:
        geometry = {}
        if mode == "stretched":
            geometry = {
                "display_x_mm": np.asarray([-1.0, 0.0, 1.0]),
                "display_y_mm": np.asarray([0.0, 1.0, 2.0]),
                "centerline_x_mm": np.asarray([0.0, 0.0, 0.0]),
                "centerline_y_mm": np.asarray([0.0, 1.0, 2.0]),
            }
        return CPRResult(
            image=np.arange(9, dtype=float).reshape(3, 3),
            centerline_distances_mm=np.asarray([0.0, 1.0, 2.0]),
            distances_from_root_mm=np.asarray([0.0, 1.0, 2.0]),
            offsets_mm=np.asarray([-1.0, 0.0, 1.0]),
            angle_degrees=0.0,
            branch_id="Path_001",
            anatomical_label="Path_001",
            point_anatomical_labels=("Path_001",) * 3,
            mode=mode,
            **geometry,
        )

    def _render_and_capture(self, mode: str, **options):
        pyplot = cpr_visualization._pyplot()
        real_subplots = pyplot.subplots
        captured = {}

        def capture_subplots(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(
                pyplot,
                "subplots",
                side_effect=capture_subplots,
            ):
                paths = cpr_visualization.write_cpr_result(
                    self._result(mode),
                    output_dir,
                    **options,
                )
            filename = paths["png_path"].name
        return captured["axes"], filename

    def test_show_centerline_false_omits_cyan_line_in_both_modes(self):
        for mode in ("straightened", "stretched"):
            with self.subTest(mode=mode):
                axes, _ = self._render_and_capture(
                    mode,
                    show_centerline=False,
                )
                cyan_lines = [
                    line
                    for line in axes.lines
                    if line.get_color() == "#00bcd4"
                ]
                self.assertEqual(cyan_lines, [])

    def test_show_centerline_defaults_to_visible(self):
        for mode in ("straightened", "stretched"):
            with self.subTest(mode=mode):
                axes, _ = self._render_and_capture(mode)
                cyan_lines = [
                    line
                    for line in axes.lines
                    if line.get_color() == "#00bcd4"
                ]
                self.assertEqual(len(cyan_lines), 1)

    def test_mode_qualified_filenames_are_opt_in(self):
        _, straightened = self._render_and_capture(
            "straightened",
            include_mode_in_filename=True,
        )
        _, stretched = self._render_and_capture(
            "stretched",
            include_mode_in_filename=True,
        )

        self.assertEqual(
            straightened,
            "Path_001_straightened_CPR_phi_0deg.png",
        )
        self.assertEqual(
            stretched,
            "Path_001_stretched_CPR_phi_0deg.png",
        )

    def test_default_filenames_remain_backward_compatible(self):
        _, straightened = self._render_and_capture("straightened")
        _, stretched = self._render_and_capture("stretched")

        self.assertEqual(straightened, "Path_001_CPR_phi_0deg.png")
        self.assertEqual(
            stretched,
            "Path_001_stretched_CPR_phi_0deg.png",
        )


if __name__ == "__main__":
    unittest.main()
