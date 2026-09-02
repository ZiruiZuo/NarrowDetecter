from __future__ import annotations

import numpy as np


def _simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError("SimpleITK is required for NIfTI input.") from error
    return sitk


def _validate_geometry(candidate, reference, candidate_name: str) -> None:
    if candidate.GetSize() != reference.GetSize():
        raise ValueError(
            f"{candidate_name} and image have different voxel grids."
        )
    if not np.allclose(
        candidate.GetSpacing(),
        reference.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError(f"{candidate_name} and image spacing do not match.")
    if not np.allclose(
        candidate.GetOrigin(),
        reference.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError(f"{candidate_name} and image origins do not match.")
    if not np.allclose(
        candidate.GetDirection(),
        reference.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{candidate_name} and image directions do not match."
        )


def read_nifti_pair(label_path: str, image_path: str):
    sitk = _simpleitk()
    label_itk = sitk.ReadImage(str(label_path))
    image_itk = sitk.ReadImage(str(image_path))
    _validate_geometry(label_itk, image_itk, "Label")
    label = sitk.GetArrayFromImage(label_itk)
    image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    spacing_zyx = tuple(float(value) for value in image_itk.GetSpacing()[::-1])
    return label, image, spacing_zyx, image_itk


def read_mask_like(mask_path: str, reference_itk, foreground_value=None):
    """Read an aligned mask, optionally selecting one label value."""
    sitk = _simpleitk()
    mask_itk = sitk.ReadImage(str(mask_path))
    _validate_geometry(mask_itk, reference_itk, "Aorta mask")
    mask = sitk.GetArrayFromImage(mask_itk)
    if foreground_value is None:
        return mask > 0
    return mask == foreground_value
