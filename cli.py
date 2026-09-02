from __future__ import annotations

from typing import Optional

try:
    from .detector import detect_coronary_stenosis
    from .models import DetectionResult, DetectorConfig
    from .myio import read_mask_like, read_nifti_pair, write_results
    from .visualization import write_visualizations
except ImportError:
    from detector import detect_coronary_stenosis
    from models import DetectionResult, DetectorConfig
    from myio import read_mask_like, read_nifti_pair, write_results
    from visualization import write_visualizations


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
    """Run stenosis detection from file paths and save JSON/CSV results.

    Parameters
    ----------
    label_path:
        Path to the multiclass coronary NIfTI label.
    image_path:
        Path to the normalized intensity NIfTI image on the same grid.
    output_dir:
        Directory used for JSON and CSV outputs.
    config:
        Detector settings. Defaults to LAD=1, LCX=2 and RCA=3.
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
    result = detect_coronary_stenosis(
        label,
        image,
        spacing_zyx,
        config,
        aorta_mask=aorta_mask,
    )
    write_results(result, output_dir)
    if save_visualizations:
        write_visualizations(
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
                    f"{percent:.1f}% "
                    f"({candidate.grade}), confidence={candidate.confidence:.2f}"
                )

    return result


def main(
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
    """Function-style entry point kept under the familiar ``main`` name."""
    return run_detection(
        label_path=label_path,
        image_path=image_path,
        output_dir=output_dir,
        config=config,
        aorta_mask_path=aorta_mask_path,
        print_summary=print_summary,
        save_visualizations=save_visualizations,
        marker_radius_mm=marker_radius_mm,
        centerline_label_radius_mm=centerline_label_radius_mm,
    )


if __name__ == '__main__':
    label_path=r"E:\ZZR\Data\DataTest\ForModel\FinedPred\50\coronary_050.nii.gz"
    image_path=r"E:\ZZR\Data\temp\water_0806.nii"
    output_dir=r"E:\ZZR\Data\DataTest\ForModel\FinedPred\50\NarrowResult"
    aorta_mask_path = None

    results = main(
        label_path,
        image_path,
        output_dir,
        aorta_mask_path=aorta_mask_path,
    )
