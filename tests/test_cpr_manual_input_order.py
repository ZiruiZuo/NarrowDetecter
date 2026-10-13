import unittest

import numpy as np

from NarrowDetecter.cpr.generator import (
    generate_centerline_cpr,
    generate_stretched_cpr,
)


class CPRManualInputOrderTests(unittest.TestCase):
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
        self.frames_u = np.tile([0.0, 1.0, 0.0], (3, 1))
        self.frames_v = np.tile([0.0, 0.0, 1.0], (3, 1))
        self.distances = np.asarray([0.0, 10.0, 20.0])

    def test_straightened_can_preserve_manual_input_order(self):
        result = generate_centerline_cpr(
            image=self.image,
            centerline_voxel_zyx=self.low_to_high_z,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=self.distances,
            radius_mm=1.0,
            pixel_mm=1.0,
            start_extension_mm=0.0,
            end_extension_mm=0.0,
            preserve_input_order=True,
        )

        np.testing.assert_allclose(
            result.image[:, result.image.shape[1] // 2],
            [10.0, 20.0, 30.0],
        )
        np.testing.assert_allclose(
            result.centerline_distances_mm,
            [0.0, 10.0, 20.0],
        )
        self.assertFalse(result.source_distance_reversed)

    def test_stretched_can_preserve_manual_input_order(self):
        result = generate_stretched_cpr(
            image=self.image,
            centerline_voxel_zyx=self.low_to_high_z,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            sampling_line_length_mm=2.0,
            slice_resolution_mm=1.0,
            frames_u_zyx=self.frames_u,
            frames_v_zyx=self.frames_v,
            centerline_distances_mm=self.distances,
            vector_of_interest_zyx=np.asarray([0.0, 0.0, 1.0]),
            start_extension_mm=0.0,
            end_extension_mm=0.0,
            preserve_input_order=True,
        )

        middle_column = int(np.argmin(np.abs(result.display_x_mm)))
        self.assertAlmostEqual(result.image[0, middle_column], 10.0)
        self.assertAlmostEqual(result.image[-1, middle_column], 30.0)
        np.testing.assert_allclose(
            result.centerline_distances_mm,
            [0.0, 10.0, 20.0],
        )
        self.assertFalse(result.source_distance_reversed)


if __name__ == "__main__":
    unittest.main()
