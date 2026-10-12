import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk

from NarrowDetecter.gui.cpr_adapter import (
    CPRCancelled,
    ManualCPRRequest,
    export_manual_cpr,
    generate_manual_cpr,
)
from NarrowDetecter.gui.image_io import LoadedVolume
from NarrowDetecter.gui.cpr_worker import CPRWorker
from NarrowDetecter.gui.models import (
    CPRSettings,
    ControlPath,
    ImageGeometry,
)
from NarrowDetecter.models import CPRResult


class ManualCPRAdapterTests(unittest.TestCase):
    def setUp(self):
        self.geometry = ImageGeometry(
            size_xyz=(20, 20, 20),
            spacing_xyz=(0.5, 1.0, 2.0),
            origin_lps_xyz=(10.0, -5.0, 3.0),
            direction_xyz=(
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
            ),
        )
        image_itk = sitk.GetImageFromArray(np.zeros((20, 20, 20), dtype=np.float32))
        image_itk.SetSpacing(self.geometry.spacing_xyz)
        image_itk.SetOrigin(self.geometry.origin_lps_xyz)
        image_itk.SetDirection(self.geometry.direction_xyz)
        self.volume = LoadedVolume(
            array_zyx=np.zeros((20, 20, 20), dtype=np.float32),
            image_itk=image_itk,
            geometry=self.geometry,
            source_kind="nifti",
            source_path="synthetic.nii.gz",
        )
        indices_xyz = ((2.0, 3.0, 4.0), (3.0, 4.0, 5.0), (4.0, 5.0, 6.0))
        self.path = ControlPath(
            path_id="path-id",
            name="Path 001",
            control_points_lps_mm=[
                self.geometry.continuous_index_xyz_to_lps(point)
                for point in indices_xyz
            ],
            cpr_settings=CPRSettings(show_centerline=False),
        )

    def _request(self, root: Path) -> ManualCPRRequest:
        return ManualCPRRequest(self.volume, self.path, root)

    def test_stretched_maps_lps_to_zyx_and_fixed_manual_parameters(self):
        calls = []

        def fake_generator(**kwargs):
            calls.append(kwargs)
            return _result(kwargs["angle_degrees"], "stretched")

        with tempfile.TemporaryDirectory() as directory:
            request = self._request(Path(directory))
            progress = []
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_stretched_cpr",
                side_effect=fake_generator,
            ):
                results = generate_manual_cpr(
                    request,
                    on_progress=lambda done, total: progress.append((done, total)),
                )

        self.assertEqual(len(results), 12)
        self.assertEqual([item["angle_degrees"] for item in calls], list(range(0, 180, 15)))
        np.testing.assert_allclose(
            calls[0]["centerline_voxel_zyx"],
            [[4.0, 3.0, 2.0], [5.0, 4.0, 3.0], [6.0, 5.0, 4.0]],
        )
        self.assertEqual(calls[0]["spacing_zyx"], (2.0, 1.0, 0.5))
        self.assertTrue(calls[0]["preserve_input_order"])
        self.assertEqual(calls[0]["curve_resolution_mm"], 0.5)
        self.assertEqual(calls[0]["centerline_smoothing_sigma_mm"], 0.8)
        self.assertEqual(calls[0]["interpolation_order"], 1)
        self.assertEqual(calls[0]["start_extension_mm"], 0.0)
        self.assertEqual(calls[0]["end_extension_mm"], 0.0)
        self.assertEqual(calls[0]["sampling_line_length_mm"], 40.0)
        self.assertEqual(calls[0]["slice_resolution_mm"], 1.0)
        self.assertEqual(calls[0]["image_direction_xyz"], self.geometry.direction_xyz)
        self.assertEqual(progress, [(index, 12) for index in range(1, 13)])

    def test_straightened_maps_width_to_radius(self):
        self.path.cpr_settings = CPRSettings(
            mode="straightened",
            sampling_line_length_mm=32.0,
            slice_resolution_mm=0.4,
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_centerline_cpr",
                side_effect=lambda **kwargs: _result(
                    kwargs["angle_degrees"],
                    "straightened",
                ),
            ) as generator:
                generate_manual_cpr(self._request(Path(directory)))

        first = generator.call_args_list[0].kwargs
        self.assertEqual(first["radius_mm"], 16.0)
        self.assertEqual(first["pixel_mm"], 0.4)
        self.assertTrue(first["preserve_input_order"])

    def test_legacy_generator_without_order_keyword_remains_supported(self):
        self.path.cpr_settings = CPRSettings(
            angle_start_degrees=0.0,
            angle_end_degrees=0.0,
        )

        def legacy_generator(
            image,
            centerline_voxel_zyx,
            spacing_zyx,
            angle_degrees,
            sampling_line_length_mm,
            slice_resolution_mm,
            interpolation_order,
            name,
            curve_resolution_mm,
            centerline_smoothing_sigma_mm,
            image_direction_xyz,
            start_extension_mm,
            end_extension_mm,
        ):
            return _result(angle_degrees, "stretched")

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_stretched_cpr",
                new=legacy_generator,
            ):
                results = generate_manual_cpr(self._request(Path(directory)))

        self.assertEqual(len(results), 1)

    def test_invalid_or_unusable_paths_are_rejected_before_generation(self):
        bad_paths = (
            [(1.0, 2.0, 3.0)],
            [(1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
            [(1.0, 2.0, 3.0), (1000.0, 2000.0, 3000.0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            for points in bad_paths:
                with self.subTest(points=points):
                    self.path.control_points_lps_mm = list(points)
                    with self.assertRaises(ValueError):
                        generate_manual_cpr(self._request(Path(directory)))
            self.path.control_points_lps_mm = [
                self.geometry.continuous_index_xyz_to_lps((2.0, 3.0, 4.0)),
                self.geometry.continuous_index_xyz_to_lps((3.0, 4.0, 5.0)),
            ]
            self.path.cpr_settings = CPRSettings(slice_resolution_mm=0.0)
            with self.assertRaises(ValueError):
                generate_manual_cpr(self._request(Path(directory)))

    def test_cancellation_is_checked_between_angles(self):
        cancelled = False
        progress = []

        def on_progress(done, total):
            nonlocal cancelled
            progress.append((done, total))
            cancelled = True

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_stretched_cpr",
                side_effect=lambda **kwargs: _result(
                    kwargs["angle_degrees"],
                    "stretched",
                ),
            ):
                with self.assertRaises(CPRCancelled):
                    generate_manual_cpr(
                        self._request(Path(directory)),
                        is_cancelled=lambda: cancelled,
                        on_progress=on_progress,
                    )
        self.assertEqual(progress, [(1, 12)])


class ManualCPRExportTests(ManualCPRAdapterTests):
    def _fake_writer(self, captured, fail=False):
        def write(result, output_dir, **kwargs):
            captured.append((Path(output_dir), kwargs))
            if fail:
                raise OSError("render failed")
            path = Path(output_dir) / f"angle-{result.angle_degrees:g}.png"
            path.write_bytes(b"png")
            return {"png_path": path}
        return write

    def test_export_stages_then_commits_images_and_json(self):
        captured = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_manual_cpr",
                return_value=[_result(0.0, "stretched"), _result(15.0, "stretched")],
            ), mock.patch(
                "NarrowDetecter.gui.cpr_adapter.write_cpr_result",
                side_effect=self._fake_writer(captured),
            ):
                target = export_manual_cpr(
                    self._request(root),
                    is_cancelled=lambda: False,
                    on_progress=lambda done, total: None,
                )

            self.assertEqual(target, root / "Path_001")
            self.assertTrue((target / "Path_001_centerline.json").is_file())
            self.assertEqual(len(list(target.glob("*.png"))), 2)
            self.assertTrue(all(item[0].parent == root for item in captured))
            self.assertTrue(all(item[0].name.startswith(".Path_001.tmp-") for item in captured))
            self.assertTrue(all(item[1]["show_centerline"] is False for item in captured))
            self.assertTrue(all(item[1]["include_mode_in_filename"] is True for item in captured))

    def test_existing_target_requires_overwrite_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "Path_001"
            target.mkdir()
            (target / "old.txt").write_text("old", encoding="utf-8")
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_manual_cpr",
                side_effect=AssertionError("generation should not run"),
            ):
                with self.assertRaises(FileExistsError):
                    export_manual_cpr(
                        self._request(root),
                        is_cancelled=lambda: False,
                        on_progress=lambda done, total: None,
                    )
            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "old")

    def test_successful_overwrite_replaces_old_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "Path_001"
            target.mkdir()
            (target / "old.txt").write_text("old", encoding="utf-8")
            request = ManualCPRRequest(self.volume, self.path, root, overwrite=True)
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_manual_cpr",
                return_value=[_result(0.0, "stretched")],
            ), mock.patch(
                "NarrowDetecter.gui.cpr_adapter.write_cpr_result",
                side_effect=self._fake_writer([]),
            ):
                export_manual_cpr(
                    request,
                    is_cancelled=lambda: False,
                    on_progress=lambda done, total: None,
                )

            self.assertFalse((target / "old.txt").exists())
            self.assertTrue((target / "angle-0.png").is_file())
            self.assertEqual(list(root.glob(".Path_001.*-*")), [])

    def test_failure_preserves_existing_target_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "Path_001"
            target.mkdir()
            (target / "old.txt").write_text("old", encoding="utf-8")
            request = ManualCPRRequest(self.volume, self.path, root, overwrite=True)
            with mock.patch(
                "NarrowDetecter.gui.cpr_adapter.generate_manual_cpr",
                return_value=[_result(0.0, "stretched")],
            ), mock.patch(
                "NarrowDetecter.gui.cpr_adapter.write_cpr_result",
                side_effect=self._fake_writer([], fail=True),
            ):
                with self.assertRaisesRegex(OSError, "render failed"):
                    export_manual_cpr(
                        request,
                        is_cancelled=lambda: False,
                        on_progress=lambda done, total: None,
                    )

            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual(list(root.glob(".Path_001.*-*")), [])


class CPRWorkerTests(unittest.TestCase):
    def test_worker_forwards_progress_and_completion(self):
        progress = []
        completed = []
        worker = CPRWorker(mock.sentinel.request)
        worker.progress.connect(lambda done, total: progress.append((done, total)))
        worker.completed.connect(completed.append)

        def fake_export(request, *, is_cancelled, on_progress):
            self.assertIs(request, mock.sentinel.request)
            self.assertFalse(is_cancelled())
            on_progress(2, 12)
            return Path("output/Path_001")

        with mock.patch(
            "NarrowDetecter.gui.cpr_worker.export_manual_cpr",
            side_effect=fake_export,
        ):
            worker.run()

        self.assertEqual(progress, [(2, 12)])
        self.assertEqual(completed, [str(Path("output/Path_001"))])

    def test_worker_cancel_emits_cancelled_not_failed(self):
        cancelled = []
        failed = []
        worker = CPRWorker(mock.sentinel.request)
        worker.cancelled.connect(lambda: cancelled.append(True))
        worker.failed.connect(failed.append)
        worker.cancel()

        def fake_export(request, *, is_cancelled, on_progress):
            if is_cancelled():
                raise CPRCancelled("cancelled")
            raise AssertionError("cancel flag was not forwarded")

        with mock.patch(
            "NarrowDetecter.gui.cpr_worker.export_manual_cpr",
            side_effect=fake_export,
        ):
            worker.run()

        self.assertEqual(cancelled, [True])
        self.assertEqual(failed, [])

def _result(angle: float, mode: str) -> CPRResult:
    geometry = {}
    if mode == "stretched":
        geometry = {
            "display_x_mm": np.asarray([-1.0, 0.0, 1.0]),
            "display_y_mm": np.asarray([0.0, 1.0]),
            "centerline_x_mm": np.asarray([0.0, 0.0]),
            "centerline_y_mm": np.asarray([0.0, 1.0]),
        }
    return CPRResult(
        image=np.ones((2, 3), dtype=float),
        centerline_distances_mm=np.asarray([0.0, 1.0]),
        distances_from_root_mm=np.asarray([0.0, 1.0]),
        offsets_mm=np.asarray([-1.0, 0.0, 1.0]),
        angle_degrees=float(angle),
        branch_id="Path 001",
        anatomical_label="Path 001",
        point_anatomical_labels=("Path 001", "Path 001"),
        mode=mode,
        **geometry,
    )


class PathNameSanitizingTests(unittest.TestCase):
    def test_unicode_names_are_preserved_and_windows_invalid_names_are_safe(self):
        from NarrowDetecter.gui.cpr_adapter import sanitized_path_name

        self.assertEqual(sanitized_path_name("左冠回旋支"), "左冠回旋支")
        self.assertEqual(sanitized_path_name("右冠动脉"), "右冠动脉")
        self.assertEqual(sanitized_path_name("A:B/C"), "A_B_C")
        self.assertEqual(sanitized_path_name("CON"), "_CON")
        self.assertEqual(sanitized_path_name("..."), "Path")


if __name__ == "__main__":
    unittest.main()
