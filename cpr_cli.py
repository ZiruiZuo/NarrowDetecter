from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

try:
    from .cpr import (
        CPRResult,
        generate_centerline_cpr,
        generate_label_cprs,
        generate_stretched_cpr,
    )
    from .models import DetectorConfig
    from .io import (
        read_mask_like,
        read_nifti_pair,
        read_stenosis_candidates,
    )
    from .visualization import write_cpr_result
except ImportError:
    from cpr import (
        CPRResult,
        generate_centerline_cpr,
        generate_label_cprs,
        generate_stretched_cpr,
    )
    from models import DetectorConfig
    from io import (
        read_mask_like,
        read_nifti_pair,
        read_stenosis_candidates,
    )
    from visualization import write_cpr_result


def run_cpr(
    image_path: str,
    centerline_voxel_zyx: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
    angles_degrees: Sequence[float] = (0.0,),
    mode: str = "straightened",
    curve_resolution_mm: float = 1.0,
    slice_resolution_mm: float = 1.0,
    sampling_line_length_mm: float = 40.0,
    frames_u_zyx: Optional[np.ndarray] = None,
    frames_v_zyx: Optional[np.ndarray] = None,
    centerline_distances_mm: Optional[np.ndarray] = None,
    distance_from_root_mm: float = 0.0,
    interpolation_order: int = 1,
    name: str = "centerline",
    save_outputs: bool = True,
    print_summary: bool = True,
    label_path: Optional[str] = None,
    label_values: Optional[Dict[str, int]] = None,
    aorta_mask_path: Optional[str] = None,
    analyze_side_branches: bool = True,
    centerline_smoothing_sigma_mm: float = 0.8,
    min_branch_length_mm: float = 3.0,
    min_branch_mean_radius_mm: float = 0.4,
    max_branch_order: Optional[int] = None,
    stenosis_result_path: Optional[str] = None,
) -> list[CPRResult]:
    """Generate CPR from either an ordered centerline or a vessel label."""
    if (centerline_voxel_zyx is None) == (label_path is None):
        raise ValueError(
            "Provide exactly one centerline source: centerline_voxel_zyx "
            "or label_path."
        )
    if save_outputs and output_dir is None:
        raise ValueError("output_dir is required when save_outputs=True.")
    angles = [float(angle) for angle in angles_degrees]
    if not angles or any(not np.isfinite(angle) for angle in angles):
        raise ValueError("angles_degrees must contain finite values.")
    if curve_resolution_mm <= 0 or slice_resolution_mm <= 0:
        raise ValueError("Curve and slice resolutions must be positive.")
    if (
        not np.isfinite(sampling_line_length_mm)
        or sampling_line_length_mm <= 0
    ):
        raise ValueError("sampling_line_length_mm must be positive and finite.")
    mode_name = str(mode).lower()
    if mode_name not in {"straightened", "stretched"}:
        raise ValueError("mode must be 'straightened' or 'stretched'.")
    if label_path is not None:
        if frames_u_zyx is not None or frames_v_zyx is not None:
            raise ValueError(
                "Explicit frame arrays are only valid with an explicit centerline."
            )
        label, image, spacing_zyx, image_itk = read_nifti_pair(
            label_path,
            image_path,
        )
        aorta_mask = (
            read_mask_like(
                aorta_mask_path,
                image_itk,
                foreground_value=1,
            )
            if aorta_mask_path is not None
            else None
        )

        cpr_results = generate_label_cprs(
            label=label,
            image=image,
            spacing_zyx=spacing_zyx,
            angles_degrees=angles,
            label_values=(
                DetectorConfig().label_values
                if label_values is None
                else label_values
            ),
            aorta_mask=aorta_mask,
            analyze_side_branches=analyze_side_branches,
            centerline_step_mm=curve_resolution_mm,
            centerline_smoothing_sigma_mm=(
                centerline_smoothing_sigma_mm
            ),
            min_branch_length_mm=min_branch_length_mm,
            min_branch_mean_radius_mm=min_branch_mean_radius_mm,
            max_branch_order=max_branch_order,
            sampling_line_length_mm=sampling_line_length_mm,
            pixel_mm=slice_resolution_mm,
            interpolation_order=interpolation_order,
            mode=mode_name,
            image_direction_xyz=image_itk.GetDirection(),
        )
    else:
        try:
            import SimpleITK as sitk
        except ImportError as error:
            raise ImportError(
                "SimpleITK is required to read the intensity NIfTI."
            ) from error

        image_itk = sitk.ReadImage(str(image_path))
        image = sitk.GetArrayFromImage(image_itk)
        if image.ndim != 3:
            raise ValueError("The intensity image must be three-dimensional.")
        spacing_zyx = tuple(
            float(value) for value in image_itk.GetSpacing()[::-1]
        )
        cpr_results = []
        for angle in angles:
            common = dict(
                image=image,
                centerline_voxel_zyx=centerline_voxel_zyx,
                spacing_zyx=spacing_zyx,
                angle_degrees=float(angle),
                frames_u_zyx=frames_u_zyx,
                frames_v_zyx=frames_v_zyx,
                centerline_distances_mm=centerline_distances_mm,
                distance_from_root_mm=distance_from_root_mm,
                interpolation_order=interpolation_order,
                name=name,
                curve_resolution_mm=(
                    curve_resolution_mm
                    if frames_u_zyx is None
                    and centerline_distances_mm is None
                    else None
                ),
                centerline_smoothing_sigma_mm=(
                    centerline_smoothing_sigma_mm
                ),
            )
            if mode_name == "straightened":
                cpr = generate_centerline_cpr(
                    **common,
                    radius_mm=0.5 * float(sampling_line_length_mm),
                    pixel_mm=slice_resolution_mm,
                )
            else:
                cpr = generate_stretched_cpr(
                    **common,
                    sampling_line_length_mm=sampling_line_length_mm,
                    slice_resolution_mm=slice_resolution_mm,
                    image_direction_xyz=image_itk.GetDirection(),
                )
            cpr_results.append(cpr)

    candidates_by_branch = (
        read_stenosis_candidates(stenosis_result_path)
        if stenosis_result_path is not None
        else {}
    )
    if save_outputs:
        for cpr in cpr_results:
            write_cpr_result(
                cpr,
                output_dir,
                candidates=candidates_by_branch.get(cpr.branch_id, ()),
            )
    if print_summary:
        for cpr in cpr_results:
            candidate_count = len(
                candidates_by_branch.get(cpr.branch_id, ())
            )
            print(
                f"{cpr.anatomical_label}: phi={cpr.angle_degrees:g} deg, "
                f"shape={cpr.image.shape}, "
                f"stenosis_marks={candidate_count}"
            )
    return cpr_results


main = run_cpr

if __name__ == "__main__":
    cpr_results = run_cpr(
        image_path=r"E:\ZZR\Data\temp\water_0806.nii",
        label_path=r"E:\ZZR\Data\DataTest\ForModel\FinedPred\50\coronary_050.nii.gz",
        output_dir=r"E:\ZZR\Data\DataTest\ForModel\FinedPred\50\cpr_result",
        label_values={
            "LAD": 2,
            "LCX": 3,
            "RCA": 4,
        },
        aorta_mask_path=r"E:\ZZR\Data\DataTest\ForModel\FinedPred\50\coronary_050_seg.nii.gz",
        angles_degrees=(10, 20, 30, 45, 60, 70, 80),
        mode="straightened",
        curve_resolution_mm=0.5,
        slice_resolution_mm=0.5,
        sampling_line_length_mm=50.0,
        analyze_side_branches=False,
    )
