from __future__ import annotations

from pathlib import Path

from ..cpr import generate_branch_cprs
from ..io import read_mask_like, read_nifti_pair
from ..models import CombinedResult, CPRConfig, DetectorConfig
from ..visualization import write_cpr_result
from .detection import _run_detection_loaded


def _include_branch(branch, config: CPRConfig) -> bool:
    if not config.analyze_side_branches and not branch.is_main:
        return False
    if branch.length_mm < config.min_branch_length_mm:
        return False
    if branch.mean_radius_mm < config.min_branch_mean_radius_mm:
        return False
    if (
        config.max_branch_order is not None
        and branch.branch_order > config.max_branch_order
    ):
        return False
    return True


def run_combined(
    label_path: str,
    image_path: str,
    detection_output_dir: str,
    cpr_output_dir: str,
    detection_config: DetectorConfig | None = None,
    cpr_config: CPRConfig | None = None,
    aorta_mask_path: str | None = None,
    print_summary: bool = True,
    save_detection_visualizations: bool = True,
    marker_radius_mm: float = 1.5,
    centerline_label_radius_mm: float = 0.6,
) -> CombinedResult:
    """Run stenosis detection and marked CPR without re-extracting branches."""
    detector_settings = detection_config or DetectorConfig()
    cpr_settings = cpr_config or CPRConfig()
    detector_settings.validate()
    cpr_settings.validate()
    detection_output = Path(detection_output_dir)
    cpr_output = Path(cpr_output_dir)
    detection_output.mkdir(parents=True, exist_ok=True)
    cpr_output.mkdir(parents=True, exist_ok=True)

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
    detection = _run_detection_loaded(
        label=label,
        image=image,
        spacing_zyx=spacing_zyx,
        image_itk=image_itk,
        aorta_mask=aorta_mask,
        output_dir=str(detection_output),
        config=detector_settings,
        print_summary=print_summary,
        save_visualizations=save_detection_visualizations,
        marker_radius_mm=marker_radius_mm,
        centerline_label_radius_mm=centerline_label_radius_mm,
    )

    cpr_results = []
    warnings = []
    image_direction_xyz = image_itk.GetDirection()
    for vessel in detection.vessels.values():
        for branch in vessel.branches:
            if not _include_branch(branch, cpr_settings):
                continue
            if not branch.measurements:
                warnings.append(
                    f"{vessel.vessel}/{branch.branch_id}: "
                    "skipped CPR because the branch has no measurements."
                )
                continue
            try:
                branch_results = generate_branch_cprs(
                    image=image,
                    spacing_zyx=spacing_zyx,
                    branch=branch,
                    config=cpr_settings,
                    image_direction_xyz=image_direction_xyz,
                )
            except ValueError as error:
                warnings.append(
                    f"{vessel.vessel}/{branch.branch_id}: skipped CPR: {error}"
                )
                continue
            for cpr in branch_results:
                write_cpr_result(
                    cpr,
                    str(cpr_output),
                    candidates=branch.candidates,
                )
            cpr_results.extend(branch_results)

    if print_summary:
        print(
            f"CPR: results={len(cpr_results)}, "
            f"warnings={len(warnings)}"
        )
    return CombinedResult(
        detection=detection,
        cpr_results=cpr_results,
        detection_output_dir=str(detection_output),
        cpr_output_dir=str(cpr_output),
        warnings=warnings,
    )
