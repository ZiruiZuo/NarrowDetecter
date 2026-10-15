import unittest

import numpy as np

from NarrowDetecter.gui.models import (
    CPRSettings,
    ImageGeometry,
    PathStore,
)


class ImageGeometryTests(unittest.TestCase):
    def test_rotated_anisotropic_geometry_round_trips_lps(self):
        geometry = ImageGeometry(
            size_xyz=(64, 48, 32),
            spacing_xyz=(0.7, 1.2, 2.5),
            origin_lps_xyz=(12.0, -8.0, 30.0),
            direction_xyz=(
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
            ),
        )
        index_xyz = (13.25, 20.5, 7.75)

        point_lps = geometry.continuous_index_xyz_to_lps(index_xyz)
        recovered = geometry.lps_to_continuous_index_xyz(point_lps)

        np.testing.assert_allclose(recovered, index_xyz, atol=1e-6)
        self.assertTrue(geometry.contains_lps(point_lps))

    def test_contains_lps_rejects_point_beyond_tolerance(self):
        geometry = ImageGeometry(
            size_xyz=(10, 10, 10),
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_lps_xyz=(0.0, 0.0, 0.0),
            direction_xyz=tuple(np.eye(3).ravel()),
        )

        self.assertTrue(geometry.contains_lps((-0.5, 4.0, 4.0)))
        self.assertFalse(geometry.contains_lps((-0.51, 4.0, 4.0)))


class CPRSettingsTests(unittest.TestCase):
    def test_default_angles_are_twelve_fifteen_degree_views(self):
        self.assertEqual(
            CPRSettings().angles_degrees(),
            tuple(float(value) for value in range(0, 180, 15)),
        )

    def test_descending_angle_range_is_supported(self):
        settings = CPRSettings(
            angle_start_degrees=90.0,
            angle_end_degrees=0.0,
            angle_step_degrees=-30.0,
        )
        self.assertEqual(settings.angles_degrees(), (90.0, 60.0, 30.0, 0.0))

    def test_invalid_common_settings_are_rejected(self):
        invalid_settings = (
            CPRSettings(mode="curved"),
            CPRSettings(slice_resolution_mm=0.0),
            CPRSettings(sampling_line_length_mm=-1.0),
            CPRSettings(angle_step_degrees=0.0),
            CPRSettings(
                angle_start_degrees=0.0,
                angle_end_degrees=90.0,
                angle_step_degrees=-15.0,
            ),
            CPRSettings(
                angle_start_degrees=0.0,
                angle_end_degrees=360.0,
                angle_step_degrees=1.0,
            ),
        )

        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    settings.validate()


class PathStoreTests(unittest.TestCase):
    @staticmethod
    def _identity_geometry():
        return ImageGeometry(
            size_xyz=(64, 64, 64),
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_lps_xyz=(0.0, 0.0, 0.0),
            direction_xyz=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )

    def test_default_names_are_unique_and_rename_rejects_collision(self):
        store = PathStore()
        first = store.create_path()
        second = store.create_path()

        self.assertEqual(first.name, "Path_001")
        self.assertEqual(second.name, "Path_002")
        store.rename_path(first.path_id, "Left vessel")
        self.assertEqual(first.name, "Left vessel")
        with self.assertRaises(ValueError):
            store.rename_path(second.path_id, " Left vessel ")

    def test_new_path_copies_current_settings(self):
        store = PathStore()
        first = store.create_path()
        first.cpr_settings.slice_resolution_mm = 0.4

        second = store.create_path()

        self.assertEqual(second.cpr_settings.slice_resolution_mm, 0.4)
        self.assertIsNot(second.cpr_settings, first.cpr_settings)

    def test_points_append_in_order_and_only_active_path_is_editable(self):
        store = PathStore()
        first = store.create_path()
        first_index = store.append_point((1.0, 2.0, 3.0))
        second_index = store.append_point((4.0, 5.0, 6.0))
        second = store.create_path()

        self.assertEqual((first_index, second_index), (0, 1))
        self.assertEqual(
            first.control_points_lps_mm,
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        with self.assertRaises(PermissionError):
            store.delete_point(first.path_id, 0)
        with self.assertRaises(PermissionError):
            store.move_point_in_view(
                first.path_id,
                0,
                "axial",
                (9.0, 9.0, 9.0),
                self._identity_geometry(),
            )
        self.assertEqual(store.active_path_id, second.path_id)

    def test_visibility_does_not_activate_inactive_path(self):
        store = PathStore()
        first = store.create_path()
        second = store.create_path()

        store.set_visibility(first.path_id, False)

        self.assertFalse(first.visible)
        self.assertEqual(store.active_path_id, second.path_id)

    def test_deleting_paths_selects_the_next_neighbor_then_allows_empty_store(self):
        store = PathStore()
        first = store.create_path()
        second = store.create_path()
        third = store.create_path()
        store.set_active(second.path_id)

        store.delete_path(second.path_id)

        self.assertEqual(
            tuple(path.path_id for path in store.paths),
            (first.path_id, third.path_id),
        )
        self.assertEqual(store.active_path_id, third.path_id)

        store.delete_path(first.path_id)
        self.assertEqual(store.active_path_id, third.path_id)

        store.delete_path(third.path_id)
        self.assertEqual(store.paths, ())
        self.assertIsNone(store.active_path_id)

    def test_move_in_view_preserves_plane_normal_coordinate(self):
        normal_axis = {"sagittal": 0, "coronal": 1, "axial": 2}
        for orientation, axis in normal_axis.items():
            with self.subTest(orientation=orientation):
                store = PathStore()
                path = store.create_path()
                store.append_point((1.0, 2.0, 3.0))

                store.move_point_in_view(
                    path.path_id,
                    0,
                    orientation,
                    (10.0, 20.0, 30.0),
                    self._identity_geometry(),
                )

                moved = path.control_points_lps_mm[0]
                self.assertEqual(moved[axis], (1.0, 2.0, 3.0)[axis])
                for changed_axis in set(range(3)) - {axis}:
                    self.assertEqual(
                        moved[changed_axis],
                        (10.0, 20.0, 30.0)[changed_axis],
                    )

    def test_editing_marks_dirty_and_export_clears_dirty(self):
        store = PathStore()
        path = store.create_path()
        self.assertFalse(path.dirty)

        store.append_point((1.0, 2.0, 3.0))
        self.assertTrue(path.dirty)
        store.mark_exported(path.path_id)
        self.assertFalse(path.dirty)
        store.clear_active_path()
        self.assertTrue(path.dirty)


if __name__ == "__main__":
    unittest.main()
