import tempfile
import unittest
from unittest import mock

import numpy as np

from NarrowDetecter.cpr.generator import (
    generate_branch_cpr,
    generate_branch_cprs,
    generate_centerline_cpr,
    generate_stretched_cpr,
)
from NarrowDetecter.models import (
    BranchResult,
    CPRConfig,
    CrossSectionMeasurement,
    StenosisCandidate,
)
from NarrowDetecter.visualization import cpr as cpr_visualization


class CPREndpointOrientationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.broadcast_to(
            np.arange(64, dtype=float)[:, None, None],
            (64, 64, 64),
        ).copy()
        self.low_to_high_z = np.asarray(
            [
                [10.0, 32.0, 32.0],
                [20.0, 32.0, 32.0],
                [30.0, 32.0, 32.0],
            ]
        )
        self.frames_u = np.tile(
            [0.0, 1.0, 0.0],
            (len(self.low_to_high_z), 1),
        )
        self.frames_v = np.tile(
            [0.0, 0.0, 1.0],
            (len(self.low_to_high_z), 1),
        )

    def test_straightened_generation_starts_at_larger_z_endpoint(self):
        result = generate_centerline_cpr(
            image=self.image,
            centerline_voxel_zyx=self.low_to_high_z,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 10.0, 20.0]),
            radius_mm=1.0,
            pixel_mm=1.0,
            start_extension_mm=0.0,
            end_extension_mm=0.0,
        )

        np.testing.assert_allclose(
            result.image[:, result.image.shape[1] // 2],
            [30.0, 20.0, 10.0],
        )
        self.assertTrue(result.source_distance_reversed)
        self.assertAlmostEqual(result.source_centerline_length_mm, 20.0)

    def test_stretched_generation_starts_at_larger_z_endpoint(self):
        result = generate_stretched_cpr(
            image=self.image,
            centerline_voxel_zyx=self.low_to_high_z,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            sampling_line_length_mm=2.0,
            slice_resolution_mm=1.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 10.0, 20.0]),
            vector_of_interest_zyx=np.asarray([0.0, 0.0, 1.0]),
            start_extension_mm=0.0,
            end_extension_mm=0.0,
        )

        middle_column = int(np.argmin(np.abs(result.display_x_mm)))
        self.assertAlmostEqual(result.image[0, middle_column], 30.0)
        self.assertAlmostEqual(result.image[-1, middle_column], 10.0)
        self.assertAlmostEqual(result.centerline_y_mm[0], 0.0)
        self.assertTrue(result.source_distance_reversed)

    def test_nonuniform_explicit_distances_reverse_with_centerline(self):
        result = generate_centerline_cpr(
            image=self.image,
            centerline_voxel_zyx=self.low_to_high_z,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 5.0, 20.0]),
            radius_mm=1.0,
            pixel_mm=1.0,
            start_extension_mm=0.0,
            end_extension_mm=0.0,
        )

        np.testing.assert_allclose(
            result.centerline_distances_mm,
            [0.0, 15.0, 20.0],
        )

    def test_unequal_extensions_follow_z_defined_start_and_end(self):
        result = generate_centerline_cpr(
            image=self.image,
            centerline_voxel_zyx=self.low_to_high_z,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 10.0, 20.0]),
            radius_mm=1.0,
            pixel_mm=1.0,
            start_extension_mm=5.0,
            end_extension_mm=10.0,
        )

        self.assertAlmostEqual(result.centerline_distances_mm[0], -5.0)
        self.assertAlmostEqual(result.centerline_distances_mm[-1], 30.0)
        np.testing.assert_allclose(
            result.image[[0, -1], result.image.shape[1] // 2],
            [35.0, 0.0],
        )

    def test_equal_endpoint_z_preserves_input_order(self):
        image = np.broadcast_to(
            np.arange(64, dtype=float)[None, :, None],
            (64, 64, 64),
        ).copy()
        centerline = np.asarray(
            [
                [20.0, 10.0, 32.0],
                [20.0, 20.0, 32.0],
                [20.0, 30.0, 32.0],
            ]
        )
        frames_u = np.tile([1.0, 0.0, 0.0], (3, 1))
        frames_v = np.tile([0.0, 0.0, 1.0], (3, 1))

        result = generate_centerline_cpr(
            image=image,
            centerline_voxel_zyx=centerline,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=frames_u,
            frames_v_zyx=frames_v,
            centerline_distances_mm=np.asarray([0.0, 10.0, 20.0]),
            radius_mm=1.0,
            pixel_mm=1.0,
            start_extension_mm=0.0,
            end_extension_mm=0.0,
        )

        np.testing.assert_allclose(
            result.image[:, result.image.shape[1] // 2],
            [10.0, 20.0, 30.0],
        )
        self.assertFalse(result.source_distance_reversed)

    def test_reversed_source_distance_moves_candidate_to_same_location(self):
        result = generate_centerline_cpr(
            image=self.image,
            centerline_voxel_zyx=self.low_to_high_z,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 10.0, 20.0]),
            radius_mm=1.0,
            pixel_mm=1.0,
            start_extension_mm=5.0,
            end_extension_mm=0.0,
        )
        candidate = StenosisCandidate(
            vessel="LAD",
            start_distance_mm=4.0,
            end_distance_mm=6.0,
            length_mm=2.0,
            min_distance_mm=5.0,
            min_center_voxel_zyx=(15.0, 32.0, 32.0),
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
        np.testing.assert_allclose(marker_lines[0].get_xdata(), [20.0, 20.0])
        span = captured["axes"].patches[0]
        self.assertAlmostEqual(float(span.get_x()), 19.0)
        self.assertAlmostEqual(
            float(span.get_x() + span.get_width()),
            21.0,
        )
        self.assertEqual(candidate.start_distance_mm, 4.0)
        self.assertEqual(candidate.end_distance_mm, 6.0)
        self.assertEqual(candidate.min_distance_mm, 5.0)

    def test_branch_point_labels_reverse_with_z_orientation(self):
        measurements = []
        for index, (point, label) in enumerate(
            zip(self.low_to_high_z, ("low", "middle", "high"))
        ):
            measurements.append(
                CrossSectionMeasurement(
                    index=index,
                    distance_mm=10.0 * index,
                    center_voxel_zyx=tuple(point),
                    tangent_zyx=(1.0, 0.0, 0.0),
                    area_mm2=1.0,
                    equivalent_diameter_mm=1.0,
                    prior_area_mm2=1.0,
                    anatomical_label=label,
                    frame_u_zyx=(0.0, 1.0, 0.0),
                    frame_v_zyx=(0.0, 0.0, 1.0),
                    distance_from_root_mm=10.0 * index,
                )
            )
        branch = BranchResult(
            branch_id="branch",
            parent_branch_id=None,
            branch_order=0,
            is_main=True,
            length_mm=20.0,
            mean_radius_mm=1.0,
            distance_from_root_mm=0.0,
            anatomical_label="LAD",
            measurements=measurements,
        )
        config = CPRConfig(
            angles_degrees=(0.0,),
            curve_resolution_mm=10.0,
            slice_resolution_mm=1.0,
            sampling_line_length_mm=2.0,
            start_extension_mm=0.0,
            end_extension_mm=0.0,
        )

        result = generate_branch_cprs(
            image=self.image,
            spacing_zyx=(1.0, 1.0, 1.0),
            branch=branch,
            config=config,
        )[0]

        self.assertEqual(
            result.point_anatomical_labels,
            ("high", "middle", "low"),
        )

    def test_legacy_branch_cpr_uses_same_z_endpoint_orientation(self):
        measurements = [
            CrossSectionMeasurement(
                index=index,
                distance_mm=10.0 * index,
                center_voxel_zyx=tuple(point),
                tangent_zyx=(1.0, 0.0, 0.0),
                area_mm2=1.0,
                equivalent_diameter_mm=1.0,
                prior_area_mm2=1.0,
                anatomical_label=label,
                frame_u_zyx=(0.0, 1.0, 0.0),
                frame_v_zyx=(0.0, 0.0, 1.0),
                distance_from_root_mm=10.0 * index,
            )
            for index, (point, label) in enumerate(
                zip(self.low_to_high_z, ("low", "middle", "high"))
            )
        ]
        branch = BranchResult(
            branch_id="branch",
            parent_branch_id=None,
            branch_order=0,
            is_main=True,
            length_mm=20.0,
            mean_radius_mm=1.0,
            distance_from_root_mm=0.0,
            anatomical_label="LAD",
            measurements=measurements,
        )

        result = generate_branch_cpr(
            image=self.image,
            spacing_zyx=(1.0, 1.0, 1.0),
            branch=branch,
            angle_degrees=0.0,
            radius_mm=1.0,
            pixel_mm=1.0,
        )

        np.testing.assert_allclose(
            result.image[:, result.image.shape[1] // 2],
            [30.0, 20.0, 10.0],
        )
        self.assertEqual(
            result.point_anatomical_labels,
            ("high", "middle", "low"),
        )


if __name__ == "__main__":
    unittest.main()
