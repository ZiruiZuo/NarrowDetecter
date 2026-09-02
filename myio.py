from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
import numpy as np

try:
    from .models import DetectionResult
except ImportError:
    from models import DetectionResult


def read_nifti_pair(label_path: str, image_path: str):
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError("SimpleITK is required for NIfTI input.") from error

    label_itk = sitk.ReadImage(str(label_path))
    image_itk = sitk.ReadImage(str(image_path))
    if label_itk.GetSize() != image_itk.GetSize():
        raise ValueError("Label and image have different voxel grids.")
    if not np.allclose(label_itk.GetSpacing(), image_itk.GetSpacing(), atol=1e-5):
        raise ValueError("Label and image spacing do not match.")
    if not np.allclose(label_itk.GetOrigin(), image_itk.GetOrigin(), atol=1e-4):
        raise ValueError("Label and image origins do not match.")
    if not np.allclose(label_itk.GetDirection(), image_itk.GetDirection(), atol=1e-5):
        raise ValueError("Label and image directions do not match.")

    label = sitk.GetArrayFromImage(label_itk)
    image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    spacing_zyx = tuple(float(v) for v in image_itk.GetSpacing()[::-1])
    return label, image, spacing_zyx, image_itk


def read_mask_like(mask_path: str, reference_itk, foreground_value=None):
    """Read an aligned mask, optionally selecting one label value."""
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "SimpleITK is required for NIfTI input."
        ) from error

    mask_itk = sitk.ReadImage(str(mask_path))
    if mask_itk.GetSize() != reference_itk.GetSize():
        raise ValueError(
            "Aorta mask and image have different voxel grids."
        )
    if not np.allclose(
        mask_itk.GetSpacing(),
        reference_itk.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError("Aorta mask and image spacing do not match.")
    if not np.allclose(
        mask_itk.GetOrigin(),
        reference_itk.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError("Aorta mask and image origins do not match.")
    if not np.allclose(
        mask_itk.GetDirection(),
        reference_itk.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError("Aorta mask and image directions do not match.")
    mask = sitk.GetArrayFromImage(mask_itk)
    if foreground_value is None:
        return mask > 0
    return mask == foreground_value


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_results(result: DetectionResult, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = _finite_json(result.to_dict())
    with (output / "stenosis_result.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)

    first_measurement = next(
        (
            measurement
            for vessel in result.vessels.values()
            for branch in (
                vessel.branches
                if vessel.branches
                else [vessel]
            )
            for measurement in branch.measurements[:1]
        ),
        None,
    )
    measurement_columns = (
        [item.name for item in fields(first_measurement)]
        if first_measurement is not None
        else []
    )
    if measurement_columns:
        with (output / "diameter_profiles.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["vessel"] + measurement_columns)
            writer.writeheader()
            for vessel_name, vessel in result.vessels.items():
                branch_items = (
                    vessel.branches
                    if vessel.branches
                    else [vessel]
                )
                for branch in branch_items:
                    for measurement in branch.measurements:
                        row = measurement.__dict__.copy()
                        row["vessel"] = vessel_name
                        row["flags"] = ";".join(row["flags"])
                        writer.writerow(row)

    branch_rows = []
    for vessel_name, vessel in result.vessels.items():
        for branch in vessel.branches:
            branch_rows.append(
                {
                    "vessel": vessel_name,
                    "branch_id": branch.branch_id,
                    "parent_branch_id": branch.parent_branch_id,
                    "branch_order": branch.branch_order,
                    "is_main": branch.is_main,
                    "length_mm": branch.length_mm,
                    "mean_radius_mm": branch.mean_radius_mm,
                    "distance_from_root_mm": (
                        branch.distance_from_root_mm
                    ),
                    "anatomical_label": branch.anatomical_label,
                    "measurement_count": len(branch.measurements),
                    "candidate_count": len(branch.candidates),
                    "warnings": ";".join(branch.warnings),
                }
            )
    if branch_rows:
        with (output / "centerline_branches.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(branch_rows[0])
            )
            writer.writeheader()
            writer.writerows(branch_rows)

    candidate_rows = []
    for vessel in result.vessels.values():
        for candidate in vessel.candidates:
            row = candidate.__dict__.copy()
            row["flags"] = ";".join(row["flags"])
            candidate_rows.append(row)
    if candidate_rows:
        with (output / "stenosis_candidates.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
            writer.writeheader()
            writer.writerows(candidate_rows)
