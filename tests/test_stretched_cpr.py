import unittest

import numpy as np

from NarrowDetecter.cpr.generator import generate_stretched_cpr


class StretchedCPRTests(unittest.TestCase):
    def test_samples_full_global_horizontal_extent(self):
        image = np.broadcast_to(
            np.arange(64, dtype=float) + 10.0,
            (16, 16, 64),
        ).copy()
        centerline = np.asarray(
            [
                [8.0, 2.0, 20.0],
                [8.0, 8.0, 30.0],
                [8.0, 14.0, 40.0],
            ]
        )
        frames_u = np.tile([1.0, 0.0, 0.0], (len(centerline), 1))
        frames_v = np.tile([0.0, 1.0, 0.0], (len(centerline), 1))

        result = generate_stretched_cpr(
            image=image,
            centerline_voxel_zyx=centerline,
            spacing_zyx=(1.0, 1.0, 1.0),
            angle_degrees=0.0,
            sampling_line_length_mm=10.0,
            slice_resolution_mm=1.0,
            frames_u_zyx=frames_u,
            frames_v_zyx=frames_v,
            interpolation_order=1,
            vector_of_interest_zyx=np.asarray([0.0, 0.0, 1.0]),
            start_extension_mm=0.0,
            end_extension_mm=0.0,
        )

        corner_values = result.image[
            [0, 0, -1, -1],
            [0, -1, 0, -1],
        ]
        np.testing.assert_allclose(
            corner_values,
            [25.0, 55.0, 25.0, 55.0],
        )


if __name__ == "__main__":
    unittest.main()
