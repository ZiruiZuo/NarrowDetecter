import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk

from NarrowDetecter.gui.image_io import (
    DicomSeriesSelectionRequired,
    discover_dicom_series,
    load_dicom_series,
    load_image,
    load_nifti,
)


class GUIImageIOTests(unittest.TestCase):
    def test_load_nifti_returns_float32_array_and_exact_geometry(self):
        array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 1.2, 2.5))
        image.SetOrigin((12.0, -8.0, 30.0))
        image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.nii.gz"
            sitk.WriteImage(image, str(path))

            loaded = load_nifti(path)

        self.assertEqual(loaded.array_zyx.dtype, np.float32)
        np.testing.assert_array_equal(loaded.array_zyx, array)
        self.assertEqual(loaded.geometry.size_xyz, (6, 5, 4))
        np.testing.assert_allclose(
            loaded.geometry.spacing_xyz,
            (0.7, 1.2, 2.5),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            loaded.geometry.origin_lps_xyz,
            (12.0, -8.0, 30.0),
            atol=1e-6,
        )
        self.assertEqual(loaded.source_kind, "nifti")
        index = (2.5, 1.25, 3.0)
        point = loaded.geometry.continuous_index_xyz_to_lps(index)
        np.testing.assert_allclose(
            loaded.geometry.lps_to_continuous_index_xyz(point),
            index,
            atol=1e-6,
        )

    def test_load_nifti_rejects_non_three_dimensional_image(self):
        image = sitk.GetImageFromArray(np.zeros((5, 6), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.nii.gz"
            sitk.WriteImage(image, str(path))
            with self.assertRaisesRegex(ValueError, "three-dimensional"):
                load_nifti(path)

    def test_discover_dicom_series_returns_every_series(self):
        fake = _FakeSimpleITK()
        fake.series_files = {
            "series-b": ["b2.dcm", "b1.dcm"],
            "series-a": ["a1.dcm"],
        }
        fake.tags_by_file = {
            "b1.dcm": {"0008|103e": "Coronary", "0020|0011": "12"},
            "a1.dcm": {},
        }
        with mock.patch(
            "NarrowDetecter.gui.image_io._simpleitk",
            return_value=fake,
        ):
            series = discover_dicom_series("dicom")

        self.assertEqual([item.series_uid for item in series], ["series-a", "series-b"])
        self.assertEqual(series[0].description, "Unnamed series")
        self.assertEqual(series[1].description, "Coronary")
        self.assertEqual(series[1].series_number, "12")
        self.assertEqual(series[1].file_count, 2)

    def test_load_dicom_series_uses_only_requested_sorted_files(self):
        fake = _FakeSimpleITK()
        fake.series_files = {
            "wanted": ["z.dcm", "a.dcm"],
            "other": ["other.dcm"],
        }
        fake.execute_image = _synthetic_image()
        with mock.patch(
            "NarrowDetecter.gui.image_io._simpleitk",
            return_value=fake,
        ):
            loaded = load_dicom_series("dicom", "wanted")

        self.assertEqual(fake.last_reader.file_names, ["a.dcm", "z.dcm"])
        self.assertEqual(loaded.series_uid, "wanted")
        self.assertEqual(loaded.source_kind, "dicom")

    def test_load_image_requires_selection_for_multiple_dicom_series(self):
        fake = _FakeSimpleITK()
        fake.series_files = {"one": ["1.dcm"], "two": ["2.dcm"]}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "NarrowDetecter.gui.image_io._simpleitk",
                return_value=fake,
            ):
                with self.assertRaises(DicomSeriesSelectionRequired) as caught:
                    load_image(directory)

        self.assertEqual(
            [item.series_uid for item in caught.exception.series],
            ["one", "two"],
        )


def _synthetic_image():
    image = sitk.GetImageFromArray(np.ones((3, 4, 5), dtype=np.float32))
    image.SetSpacing((0.8, 0.9, 1.5))
    return image


class _FakeImageFileReader:
    def __init__(self, owner):
        self.owner = owner
        self.file_name = None

    def SetFileName(self, file_name):
        self.file_name = file_name

    def LoadPrivateTagsOn(self):
        return None

    def ReadImageInformation(self):
        return None

    def HasMetaDataKey(self, key):
        return key in self.owner.tags_by_file.get(self.file_name, {})

    def GetMetaData(self, key):
        return self.owner.tags_by_file[self.file_name][key]


class _FakeSeriesReader:
    def __init__(self, owner):
        self.owner = owner
        self.file_names = []

    def SetFileNames(self, file_names):
        self.file_names = list(file_names)

    def GetGDCMSeriesIDs(self, directory):
        return tuple(self.owner.series_files)

    def GetGDCMSeriesFileNames(self, directory, series_uid):
        return tuple(self.owner.series_files[series_uid])

    def Execute(self):
        return self.owner.execute_image


class _FakeSimpleITK:
    def __init__(self):
        self.series_files = {}
        self.tags_by_file = {}
        self.execute_image = _synthetic_image()
        self.last_reader = None

    def ImageFileReader(self):
        return _FakeImageFileReader(self)

    def ImageSeriesReader(self):
        self.last_reader = _FakeSeriesReader(self)
        return self.last_reader

    def GetGDCMSeriesIDs(self, directory):
        return tuple(self.series_files)

    def GetGDCMSeriesFileNames(self, directory, series_uid):
        return tuple(self.series_files[series_uid])

    @staticmethod
    def GetArrayFromImage(image):
        return sitk.GetArrayFromImage(image)


if __name__ == "__main__":
    unittest.main()
