from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .models import ImageGeometry


def _simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "SimpleITK is required to load NIfTI and DICOM images."
        ) from error
    return sitk


@dataclass(frozen=True)
class DicomSeriesInfo:
    series_uid: str
    description: str
    series_number: str
    file_count: int


@dataclass
class LoadedVolume:
    array_zyx: np.ndarray
    image_itk: Any
    geometry: ImageGeometry
    source_kind: Literal["nifti", "dicom"]
    source_path: str
    series_uid: str | None = None


class DicomSeriesSelectionRequired(ValueError):
    def __init__(self, series: tuple[DicomSeriesInfo, ...]):
        super().__init__("The DICOM folder contains multiple series.")
        self.series = series


def _geometry_from_image(image) -> ImageGeometry:
    if int(image.GetDimension()) != 3:
        raise ValueError("The intensity image must be three-dimensional.")
    return ImageGeometry(
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_lps_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction_xyz=tuple(float(value) for value in image.GetDirection()),
    )


def _loaded_volume(
    image,
    *,
    source_kind: Literal["nifti", "dicom"],
    source_path: str | Path,
    series_uid: str | None = None,
) -> LoadedVolume:
    sitk = _simpleitk()
    geometry = _geometry_from_image(image)
    array = np.asarray(sitk.GetArrayFromImage(image))
    if array.ndim != 3 or array.size == 0:
        raise ValueError("The intensity image must be a nonempty three-dimensional volume.")
    expected_shape = tuple(reversed(geometry.size_xyz))
    if array.shape != expected_shape:
        raise ValueError("The image array and image geometry are inconsistent.")
    return LoadedVolume(
        array_zyx=np.asarray(array, dtype=np.float32),
        image_itk=image,
        geometry=geometry,
        source_kind=source_kind,
        source_path=str(Path(source_path).resolve()),
        series_uid=series_uid,
    )


def _metadata_value(reader, key: str, fallback: str) -> str:
    if reader.HasMetaDataKey(key):
        value = str(reader.GetMetaData(key)).strip()
        if value:
            return value
    return fallback


def discover_dicom_series(
    directory: str | Path,
) -> tuple[DicomSeriesInfo, ...]:
    sitk = _simpleitk()
    series_reader = sitk.ImageSeriesReader()
    series_ids = tuple(series_reader.GetGDCMSeriesIDs(str(directory)) or ())
    discovered = []
    for series_uid in sorted(str(value) for value in series_ids):
        files = sorted(
            str(value)
            for value in series_reader.GetGDCMSeriesFileNames(
                str(directory),
                series_uid,
            )
        )
        description = "Unnamed series"
        series_number = ""
        if files:
            metadata_reader = sitk.ImageFileReader()
            metadata_reader.SetFileName(files[0])
            metadata_reader.LoadPrivateTagsOn()
            metadata_reader.ReadImageInformation()
            description = _metadata_value(
                metadata_reader,
                "0008|103e",
                "Unnamed series",
            )
            series_number = _metadata_value(
                metadata_reader,
                "0020|0011",
                "",
            )
        discovered.append(
            DicomSeriesInfo(
                series_uid=series_uid,
                description=description,
                series_number=series_number,
                file_count=len(files),
            )
        )
    return tuple(discovered)


def load_nifti(path: str | Path) -> LoadedVolume:
    sitk = _simpleitk()
    image = sitk.ReadImage(str(path))
    return _loaded_volume(
        image,
        source_kind="nifti",
        source_path=path,
    )


def load_dicom_series(
    directory: str | Path,
    series_uid: str,
) -> LoadedVolume:
    sitk = _simpleitk()
    reader = sitk.ImageSeriesReader()
    files = sorted(
        str(value)
        for value in reader.GetGDCMSeriesFileNames(
            str(directory),
            str(series_uid),
        )
    )
    if not files:
        raise ValueError(f"DICOM series '{series_uid}' contains no files.")
    reader.SetFileNames(files)
    image = reader.Execute()
    return _loaded_volume(
        image,
        source_kind="dicom",
        source_path=directory,
        series_uid=str(series_uid),
    )


def load_image(
    path: str | Path,
    series_uid: str | None = None,
) -> LoadedVolume:
    source = Path(path)
    if source.is_file():
        if series_uid is not None:
            raise ValueError("series_uid is only valid for a DICOM directory.")
        return load_nifti(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Image source does not exist: {source}")
    series = discover_dicom_series(source)
    if series_uid is not None:
        known = {item.series_uid for item in series}
        if str(series_uid) not in known:
            raise ValueError(f"Unknown DICOM series UID: {series_uid}")
        return load_dicom_series(source, str(series_uid))
    if not series:
        raise ValueError("The DICOM folder contains no readable series.")
    if len(series) > 1:
        raise DicomSeriesSelectionRequired(series)
    return load_dicom_series(source, series[0].series_uid)
