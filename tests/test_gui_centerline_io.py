import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from NarrowDetecter.gui.centerline_io import (
    FORMAT_NAME,
    FORMAT_VERSION,
    geometry_matches,
    path_to_document,
    read_centerline_json,
    write_centerline_json,
)
from NarrowDetecter.gui.models import (
    CPRSettings,
    ControlPath,
    ImageGeometry,
    PathStore,
)


class CenterlineIOTests(unittest.TestCase):
    def setUp(self):
        self.geometry = ImageGeometry(
            size_xyz=(64, 48, 32),
            spacing_xyz=(0.7, 1.2, 2.5),
            origin_lps_xyz=(12.0, -8.0, 30.0),
            direction_xyz=(
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
            ),
        )
        self.path = ControlPath(
            path_id="source-id",
            name="Path_001",
            control_points_lps_mm=[
                (10.0, 20.0, 30.0),
                (11.0, 21.0, 31.0),
                (12.0, 22.0, 32.0),
            ],
            cpr_settings=CPRSettings(
                mode="straightened",
                angle_start_degrees=15.0,
                angle_end_degrees=75.0,
                angle_step_degrees=30.0,
                sampling_line_length_mm=32.0,
                slice_resolution_mm=0.4,
                show_centerline=False,
            ),
        )

    def test_round_trip_preserves_points_geometry_and_common_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "centerline.json"
            write_centerline_json(self.path, self.geometry, destination)

            loaded, geometry = read_centerline_json(destination)

        self.assertEqual(loaded.name, "Path_001")
        self.assertEqual(
            loaded.control_points_lps_mm,
            self.path.control_points_lps_mm,
        )
        self.assertEqual(loaded.cpr_settings, self.path.cpr_settings)
        self.assertEqual(geometry, self.geometry)
        self.assertFalse(loaded.dirty)

    def test_document_records_fixed_manual_fit_settings(self):
        document = path_to_document(self.path, self.geometry)

        self.assertEqual(document["format"], FORMAT_NAME)
        self.assertEqual(document["version"], FORMAT_VERSION)
        self.assertEqual(document["coordinate_system"], "LPS_mm")
        self.assertEqual(document["fit_settings"]["resample_spacing_mm"], 0.5)
        self.assertEqual(document["fit_settings"]["smoothing_sigma_mm"], 0.8)
        self.assertEqual(document["fit_settings"]["interpolation_order"], 1)
        self.assertEqual(document["fit_settings"]["start_extension_mm"], 0.0)
        self.assertEqual(document["fit_settings"]["end_extension_mm"], 0.0)

    def test_export_rejects_too_few_or_nonfinite_points(self):
        for points in (
            [(1.0, 2.0, 3.0)],
            [(1.0, 2.0, 3.0), (np.nan, 2.0, 3.0)],
        ):
            with self.subTest(points=points):
                path = ControlPath(
                    path_id="bad",
                    name="Bad",
                    control_points_lps_mm=points,
                )
                with self.assertRaises(ValueError):
                    path_to_document(path, self.geometry)

    def test_import_rejects_unknown_format_version_and_bad_direction(self):
        base = path_to_document(self.path, self.geometry)
        invalid_documents = []
        wrong_format = dict(base, format="other")
        invalid_documents.append(wrong_format)
        wrong_version = dict(base, version=99)
        invalid_documents.append(wrong_version)
        bad_direction = json.loads(json.dumps(base))
        bad_direction["image_geometry"]["direction"] = [1.0] * 9
        invalid_documents.append(bad_direction)

        with tempfile.TemporaryDirectory() as directory:
            for index, document in enumerate(invalid_documents):
                with self.subTest(index=index):
                    source = Path(directory) / f"bad-{index}.json"
                    source.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_centerline_json(source)

    def test_import_rejects_duplicate_json_keys(self):
        duplicate = '{"format":"narrowdetecter-centerline","format":"other"}'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "duplicate.json"
            source.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                read_centerline_json(source)

    def test_geometry_match_uses_medical_image_tolerances(self):
        close = ImageGeometry(
            size_xyz=self.geometry.size_xyz,
            spacing_xyz=(0.700001, 1.2, 2.5),
            origin_lps_xyz=(12.00001, -8.0, 30.0),
            direction_xyz=self.geometry.direction_xyz,
        )
        different = ImageGeometry(
            size_xyz=(65, 48, 32),
            spacing_xyz=self.geometry.spacing_xyz,
            origin_lps_xyz=self.geometry.origin_lps_xyz,
            direction_xyz=self.geometry.direction_xyz,
        )

        self.assertTrue(geometry_matches(self.geometry, close))
        self.assertFalse(geometry_matches(self.geometry, different))

    def test_path_store_resolves_imported_name_collisions_independently(self):
        first_source = ControlPath(
            path_id="one",
            name="Path_001",
            control_points_lps_mm=list(self.path.control_points_lps_mm),
        )
        second_source = ControlPath(
            path_id="two",
            name="Path_001",
            control_points_lps_mm=list(self.path.control_points_lps_mm),
        )
        store = PathStore()

        first = store.add_imported_path(first_source)
        second = store.add_imported_path(second_source)

        self.assertEqual((first.name, second.name), ("Path_001", "Path_001_2"))
        self.assertEqual((first_source.name, second_source.name), ("Path_001", "Path_001"))
        self.assertNotEqual(first.path_id, second.path_id)


if __name__ == "__main__":
    unittest.main()
