import unittest

import numpy as np

from NarrowDetecter.cpr.generator import (
    _estimate_endpoint_tangent,
    _extend_centerline_geometry,
    generate_centerline_cpr,
    generate_stretched_cpr,
)
from NarrowDetecter.models import CPRConfig


class CPRExtensionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.broadcast_to(
            np.arange(64, dtype=float)[None, :, None],
            (64, 64, 64),
        ).copy()
        self.centerline = np.asarray(
            [
                [32.0, 25.0, 32.0],
                [32.0, 30.0, 32.0],
                [32.0, 35.0, 32.0],
            ]
        )
        self.frames_u = np.tile(
            [1.0, 0.0, 0.0],
            (len(self.centerline), 1),
        )
        self.frames_v = np.tile(
            [0.0, 0.0, 1.0],
            (len(self.centerline), 1),
        )

    def test_config_rejects_negative_endpoint_extension(self):
        config = CPRConfig()
        config.start_extension_mm = -1.0

        with self.assertRaisesRegex(ValueError, "extensions"):
            config.validate()

    def test_explicit_centerline_distances_must_start_at_zero(self):
        with self.assertRaisesRegex(ValueError, "start at zero"):
            generate_centerline_cpr(
                image=self.image,
                centerline_voxel_zyx=self.centerline,
                spacing_zyx=(1.0, 1.0, 1.0),
                angle_degrees=0.0,
                frames_u_zyx=self.frames_u,
                frames_v_zyx=self.frames_v,
                centerline_distances_mm=np.asarray([10.0, 15.0, 20.0]),
                radius_mm=2.0,
                pixel_mm=1.0,
                start_extension_mm=0.0,
                end_extension_mm=0.0,
            )

    def test_straightened_cpr_extends_both_centerline_ends_by_15_mm(self):
        result = generate_centerline_cpr(
            image=self.image,
            centerline_voxel_zyx=self.centerline,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 5.0, 10.0]),
            radius_mm=2.0,
            pixel_mm=1.0,
        )

        self.assertAlmostEqual(result.centerline_distances_mm[0], -15.0)
        self.assertAlmostEqual(result.centerline_distances_mm[-1], 25.0)
        self.assertEqual(result.image.shape[0], 9)
        np.testing.assert_allclose(
            result.image[[0, -1], result.image.shape[1] // 2],
            [10.0, 50.0],
        )

    def test_extension_names_follow_anatomical_start_and_end(self):
        result = generate_centerline_cpr(
            image=self.image,
            centerline_voxel_zyx=self.centerline,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 5.0, 10.0]),
            radius_mm=2.0,
            pixel_mm=1.0,
            start_extension_mm=5.0,
            end_extension_mm=10.0,
        )

        self.assertAlmostEqual(result.centerline_distances_mm[0], -5.0)
        self.assertAlmostEqual(result.centerline_distances_mm[-1], 20.0)
        np.testing.assert_allclose(
            result.image[[0, -1], result.image.shape[1] // 2],
            [20.0, 45.0],
        )

    def test_stretched_cpr_extends_both_centerline_ends_by_15_mm(self):
        result = generate_stretched_cpr(
            image=self.image,
            centerline_voxel_zyx=self.centerline,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            sampling_line_length_mm=4.0,
            slice_resolution_mm=1.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=np.asarray([0.0, 5.0, 10.0]),
            vector_of_interest_zyx=np.asarray([0.0, 0.0, 1.0]),
        )

        self.assertAlmostEqual(result.centerline_distances_mm[0], -15.0)
        self.assertAlmostEqual(result.centerline_distances_mm[-1], 25.0)
        self.assertAlmostEqual(result.display_y_mm[-1], 40.0)
        np.testing.assert_allclose(
            result.image[[0, -1], result.image.shape[1] // 2],
            [10.0, 50.0],
        )

    def test_endpoint_extensions_use_local_quadratic_tangents(self):
        distances = np.asarray([0.0, 1.0, 2.0, 3.0])
        centerline = np.column_stack(
            [
                30.0 - distances,
                distances**2,
                np.full_like(distances, 20.0),
            ]
        )
        frames_u = np.tile([0.0, 1.0, 0.0], (4, 1))
        frames_v = np.tile([0.0, 0.0, 1.0], (4, 1))

        extended, _, _, extended_distances = _extend_centerline_geometry(
            centerline_voxel_zyx=centerline,
            spacing_zyx=np.ones(3),
            frames_u_zyx=frames_u,
            frames_v_zyx=frames_v,
            distances_mm=distances,
            start_extension_mm=2.0,
            end_extension_mm=2.0,
        )

        np.testing.assert_allclose(
            extended[0],
            [32.0, 0.0, 20.0],
            atol=1e-7,
        )
        end_tangent = np.asarray([-1.0, 6.0, 0.0])
        end_tangent /= np.linalg.norm(end_tangent)
        np.testing.assert_allclose(
            extended[-1],
            centerline[-1] + 2.0 * end_tangent,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            extended_distances[[0, -1]],
            [-2.0, 5.0],
        )

    def test_endpoint_tangent_uses_three_unique_distance_samples(self):
        start_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [100.0, 100.0, 0.0],
                [1.0, 1.0, 0.0],
                [10.0, 100.0, 0.0],
            ]
        )
        start_distances = np.asarray([0.0, 0.0, 1.0, 10.0])
        end_points = np.asarray(
            [
                [0.0, 100.0, 0.0],
                [9.0, 1.0, 0.0],
                [-100.0, -100.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        end_distances = np.asarray([0.0, 9.0, 10.0, 10.0])

        np.testing.assert_allclose(
            _estimate_endpoint_tangent(
                start_points,
                start_distances,
                at_start=True,
            ),
            [1.0, 0.0, 0.0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _estimate_endpoint_tangent(
                end_points,
                end_distances,
                at_start=False,
            ),
            [1.0, 0.0, 0.0],
            atol=1e-7,
        )

    def test_endpoint_tangent_scales_extreme_distance_ranges(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 4.0, 0.0],
            ]
        )

        for scale in (1e-12, 1e9):
            distances = scale * np.asarray([0.0, 1.0, 2.0])
            np.testing.assert_allclose(
                _estimate_endpoint_tangent(
                    points,
                    distances,
                    at_start=True,
                ),
                [1.0, 0.0, 0.0],
                atol=1e-7,
            )
            expected_end = np.asarray([1.0, 4.0, 0.0])
            expected_end /= np.linalg.norm(expected_end)
            np.testing.assert_allclose(
                _estimate_endpoint_tangent(
                    points,
                    distances,
                    at_start=False,
                ),
                expected_end,
                atol=1e-7,
            )

    def test_endpoint_tangent_replaces_near_duplicate_distance_sample(self):
        distances = np.asarray([0.0, 1e-15, 1.0, 10.0])
        points = np.column_stack(
            [
                distances,
                distances**2,
                np.zeros_like(distances),
            ]
        )

        np.testing.assert_allclose(
            _estimate_endpoint_tangent(
                points,
                distances,
                at_start=True,
            ),
            [1.0, 0.0, 0.0],
            atol=1e-7,
        )

    def test_endpoint_tangent_falls_back_with_two_unique_distances(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [-50.0, -50.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        distances = np.asarray([0.0, 0.0, 1.0])
        expected = np.asarray([1.0, 1.0, 0.0])
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(
            _estimate_endpoint_tangent(
                points,
                distances,
                at_start=True,
            ),
            expected,
            atol=1e-7,
        )

    def test_disabled_endpoint_does_not_require_a_valid_tangent(self):
        distances = np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        valid_start = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        degenerate_end = np.tile([3.0, 0.0, 0.0], (3, 1))
        centerline = np.concatenate(
            [valid_start, degenerate_end],
            axis=0,
        )
        frames_u = np.tile([0.0, 1.0, 0.0], (6, 1))
        frames_v = np.tile([0.0, 0.0, 1.0], (6, 1))

        extended, _, _, extended_distances = _extend_centerline_geometry(
            centerline_voxel_zyx=centerline,
            spacing_zyx=np.ones(3),
            frames_u_zyx=frames_u,
            frames_v_zyx=frames_v,
            distances_mm=distances,
            start_extension_mm=1.0,
            end_extension_mm=0.0,
        )

        np.testing.assert_allclose(extended[0], [-1.0, 0.0, 0.0])
        self.assertAlmostEqual(extended_distances[0], -1.0)


if __name__ == "__main__":
    unittest.main()
