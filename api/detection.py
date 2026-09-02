from __future__ import annotations

from typing import Optional

from ..detection import detect_coronary_stenosis
from ..io import read_mask_like, read_nifti_pair, write_detection_result
from ..models import DetectionResult, DetectorConfig
from ..visualization import write_detection_visualizations


def _print_detection_summary(result: DetectionResult) -> None:
    for name, vessel in result.vessels.items():
        print(
            f"{name}: status={vessel.status}, "
            f"length={vessel.centerline_length_mm:.1f} mm, "
            f"branches={len(vessel.branches)}, "
            f"candidates={len(vessel.candidates)}"
        )
        for candidate in vessel.candidates:
            percent = 100.0 * candidate.max_diameter_stenosis
            print(
                f"  {candidate.branch_id} "
                f"{candidate.min_distance_mm:.1f} mm: "
                f"{percent:.1f}% ({candidate.grade}), "
                f"confidence={candidate.confidence:.2f}"
            )


def _run_detection_loaded(
    *,
    label,
    image,
    spacing_zyx,
    image_itk,
    aorta_mask,
    output_dir: str,
    config: DetectorConfig,
    print_summary: bool,
    save_visualizations: bool,
    marker_radius_mm: float,
    centerline_label_radius_mm: float,
) -> DetectionResult:
    result = detect_coronary_stenosis(
        label,
        image,
        spacing_zyx,
        config,
        aorta_mask=aorta_mask,
    )
    write_detection_result(result, output_dir)
    if save_visualizations:
        write_detection_visualizations(
            result=result,
            image=image,
            label=label,
            reference_itk=image_itk,
            output_dir=output_dir,
            marker_radius_mm=marker_radius_mm,
            centerline_label_radius_mm=centerline_label_radius_mm,
            cross_section_radius_mm=config.cross_section_radius_mm,
            cross_section_pixel_mm=config.cross_section_pixel_mm,
            config=config,
        )
    if print_summary:
        _print_detection_summary(result)
    return result


def run_detection(
    label_path: str,
    image_path: str,
    output_dir: str,
    config: Optional[DetectorConfig] = None,
    print_summary: bool = True,
    save_visualizations: bool = True,
    marker_radius_mm: float = 1.5,
    aorta_mask_path: Optional[str] = None,
    centerline_label_radius_mm: float = 0.6,
) -> DetectionResult:
    """Run stenosis detection from file paths and save JSON results.

    Parameters
    ----------
    label_path:
        Path to the multiclass coronary NIfTI label.
    image_path:
        Path to the normalized intensity NIfTI image on the same grid.
    output_dir:
        Directory used for JSON and visualization outputs.
    config:
        Detector settings. Defaults to LAD=2, LCX=3 and RCA=4.
    aorta_mask_path:
        Optional aorta NIfTI mask used to select the nearest vessel root.
    print_summary:
        Print a compact vessel and candidate summary when True.
    save_visualizations:
        Save branch curves, orthogonal MPR images, and marker NIfTI.
    marker_radius_mm:
        Radius of candidate markers in the output NIfTI volume.
    centerline_label_radius_mm:
        Radius of the anatomical centerline labels in the output NIfTI.

    Returns
    -------
    DetectionResult
        The complete in-memory result, also written to ``output_dir``.
    """
    config = config or DetectorConfig()
    label, image, spacing_zyx, image_itk = read_nifti_pair(label_path, image_path)
    aorta_mask = (
        read_mask_like(
            aorta_mask_path,
            image_itk,
            foreground_value=1,
        )
        if aorta_mask_path is not None
        else None
    )
    return _run_detection_loaded(
        label=label,
        image=image,
        spacing_zyx=spacing_zyx,
        image_itk=image_itk,
        aorta_mask=aorta_mask,
        output_dir=output_dir,
        config=config,
        print_summary=print_summary,
        save_visualizations=save_visualizations,
        marker_radius_mm=marker_radius_mm,
        centerline_label_radius_mm=centerline_label_radius_mm,
    )
