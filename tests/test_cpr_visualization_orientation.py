import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from NarrowDetecter.models import CPRResult, StenosisCandidate
from NarrowDetecter.visualization import cpr as cpr_visualization


class CPRVisualizationOrientationTests(unittest.TestCase):
    def _make_result(self, mode: str) -> CPRResult:
        kwargs = {}
        if mode == "stretched":
            kwargs = {
                "display_x_mm": np.asarray([-2.0, 0.0, 2.0]),
                "display_y_mm": np.asarray([0.0, 5.0, 10.0]),
                "centerline_x_mm": np.asarray([0.0, 5.0, 10.0]),
                "centerline_y_mm": np.asarray([0.0, 5.0, 10.0]),
            }
        return CPRResult(
            image=np.asarray(
                [
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0],
                ]
            ),
            centerline_distances_mm=np.asarray([0.0, 5.0, 10.0]),
            distances_from_root_mm=np.asarray([0.0, 5.0, 10.0]),
            offsets_mm=np.asarray([-1.0, 1.0]),
            angle_degrees=90.0,
            branch_id="branch",
            anatomical_label="LAD",
            point_anatomical_labels=("start", "middle", "end"),
            mode=mode,
            **kwargs,
        )

    def test_straightened_display_starts_at_anatomical_start(self):
        helper = getattr(
            cpr_visualization,
            "_straightened_display_geometry",
            None,
        )
        self.assertIsNotNone(helper)

        image, distances, labels, distance_origin = helper(
            self._make_result("straightened")
        )

        np.testing.assert_allclose(image[0], [10.0, 11.0])
        np.testing.assert_allclose(distances, [0.0, 5.0, 10.0])
        self.assertEqual(labels, ("start", "middle", "end"))
        self.assertAlmostEqual(0.0 - distance_origin, 0.0)
        self.assertAlmostEqual(10.0 - distance_origin, 10.0)

    def test_stretched_display_places_anatomical_start_at_origin(self):
        helper = getattr(
            cpr_visualization,
            "_stretched_display_geometry",
            None,
        )
        self.assertIsNotNone(helper)

        displayed = helper(self._make_result("stretched"))

        self.assertAlmostEqual(displayed.centerline_x_mm[0], 0.0)
        self.assertAlmostEqual(displayed.centerline_y_mm[0], 0.0)
        np.testing.assert_allclose(
            displayed.centerline_x_mm,
            [0.0, 5.0, 10.0],
        )
        np.testing.assert_allclose(
            displayed.centerline_y_mm,
            [0.0, 5.0, 10.0],
        )
        np.testing.assert_allclose(
            displayed.display_x_mm,
            [-2.0, 0.0, 2.0],
        )

    def test_stretched_writer_draws_first_sample_at_top_origin(self):
        plt = cpr_visualization._pyplot()
        real_subplots = plt.subplots
        captured = {}

        def capture_subplots(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured["figure"] = figure
            captured["axes"] = axes
            return figure, axes

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(
                plt,
                "subplots",
                side_effect=capture_subplots,
            ):
                cpr_visualization.write_cpr_result(
                    self._make_result("stretched"),
                    output_dir,
                )

        axes = captured["axes"]
        image_artist = axes.images[0]
        self.assertEqual(image_artist.origin, "upper")
        np.testing.assert_allclose(
            image_artist.get_extent(),
            [-2.0, 2.0, 10.0, 0.0],
        )
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.0, 5.0, 10.0])

    def test_straightened_candidate_marker_uses_start_origin(self):
        result = replace(
            self._make_result("straightened"),
            centerline_distances_mm=np.asarray([-5.0, 0.0, 5.0]),
        )
        candidate = StenosisCandidate(
            vessel="LAD",
            start_distance_mm=2.0,
            end_distance_mm=4.0,
            length_mm=2.0,
            min_distance_mm=3.0,
            min_center_voxel_zyx=(0.0, 0.0, 0.0),
            min_diameter_mm=1.0,
            reference_diameter_mm=2.0,
            max_diameter_stenosis=0.5,
            max_area_stenosis=None,
            grade="moderate",
            confidence=1.0,
        )
        plt = cpr_visualization._pyplot()
        real_subplots = plt.subplots
        captured = {}

        def capture_subplots(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(
                plt,
                "subplots",
                side_effect=capture_subplots,
            ):
                cpr_visualization.write_cpr_result(
                    result,
                    output_dir,
                    candidates=(candidate,),
                )

        marker_lines = [
            line
            for line in captured["axes"].lines
            if line.get_color() == "#d95f02"
            and line.get_linestyle() == "--"
        ]
        self.assertEqual(len(marker_lines), 1)
        np.testing.assert_allclose(marker_lines[0].get_xdata(), [8.0, 8.0])


if __name__ == "__main__":
    unittest.main()
