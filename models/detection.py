from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DetectorConfig:
    """All distances are in millimetres unless stated otherwise."""

    label_values: Dict[str, int] = field(
        default_factory=lambda: {"LAD": 2, "LCX": 3, "RCA": 4}
    )

    centerline_step_mm: float = 0.4
    centerline_smoothing_sigma_mm: float = 0.8
    analyze_side_branches: bool = True
    min_branch_length_mm: float = 3.0
    min_branch_mean_radius_mm: float = 0.4
    max_branch_order: Optional[int] = None

    cross_section_pixel_mm: float = 0.15
    cross_section_radius_mm: float = 5.0
    mask_dilation_mm: float = 1.2
    image_smoothing_sigma_mm: float = 0.25
    seed_radius_mm: float = 0.45

    intensity_tolerance: float = 0.20
    gradient_percentile: float = 82.0
    intensity_weight: float = 0.55
    gradient_weight: float = 0.45
    region_cost_threshold: float = 0.62

    min_lumen_area_mm2: float = 0.20
    max_area_ratio_to_prior: float = 1.8
    min_area_ratio_to_prior: float = 0.08

    reference_window_mm: float = 20.0
    reference_exclusion_mm: float = 1.5
    reference_percentile: float = 80.0
    profile_smoothing_mm: float = 1.0

    candidate_threshold: float = 0.25
    min_stenosis_length_mm: float = 1.0
    merge_gap_mm: float = 1.0

    endpoint_exclusion_mm: float = 2.0
    bifurcation_exclusion_mm: float = 1.5
    use_label_fallback: bool = True

    def validate(self) -> None:
        if self.centerline_step_mm <= 0 or self.cross_section_pixel_mm <= 0:
            raise ValueError("Sampling steps must be positive.")
        if self.cross_section_radius_mm <= 0:
            raise ValueError("cross_section_radius_mm must be positive.")
        if not 0 < self.reference_percentile <= 100:
            raise ValueError(
                "reference_percentile must be in (0, 100]."
            )
        if not 0 <= self.candidate_threshold < 1:
            raise ValueError(
                "candidate_threshold must be in [0, 1)."
            )
        if not 0 <= self.gradient_percentile <= 100:
            raise ValueError(
                "gradient_percentile must be in [0, 100]."
            )
        if (
            self.min_branch_length_mm < 0
            or self.min_branch_mean_radius_mm < 0
        ):
            raise ValueError(
                "Branch pruning thresholds cannot be negative."
            )
        if (
            self.max_branch_order is not None
            and self.max_branch_order < 0
        ):
            raise ValueError("max_branch_order cannot be negative.")


@dataclass
class CrossSectionMeasurement:
    index: int
    distance_mm: float
    center_voxel_zyx: Tuple[float, float, float]
    tangent_zyx: Tuple[float, float, float]
    area_mm2: float
    equivalent_diameter_mm: float
    prior_area_mm2: float
    mean_lumen_intensity: Optional[float] = None
    reference_mean_intensity: Optional[float] = None
    intensity_difference: Optional[float] = None
    intensity_change_ratio: Optional[float] = None
    branch_id: str = ""
    branch_index: int = 0
    distance_from_root_mm: float = 0.0
    anatomical_label: str = ""
    frame_u_zyx: Optional[Tuple[float, float, float]] = None
    frame_v_zyx: Optional[Tuple[float, float, float]] = None
    reference_diameter_mm: Optional[float] = None
    diameter_stenosis: Optional[float] = None
    area_stenosis: Optional[float] = None
    quality: float = 0.0
    flags: List[str] = field(default_factory=list)


@dataclass
class StenosisCandidate:
    vessel: str
    start_distance_mm: float
    end_distance_mm: float
    length_mm: float
    min_distance_mm: float
    min_center_voxel_zyx: Tuple[float, float, float]
    min_diameter_mm: float
    reference_diameter_mm: float
    max_diameter_stenosis: float
    max_area_stenosis: Optional[float]
    grade: str
    confidence: float
    branch_id: str = ""
    parent_branch_id: Optional[str] = None
    branch_order: int = 0
    is_main: bool = True
    start_distance_from_root_mm: float = 0.0
    end_distance_from_root_mm: float = 0.0
    min_distance_from_root_mm: float = 0.0
    anatomical_label: str = ""
    flags: List[str] = field(default_factory=list)


@dataclass
class BranchResult:
    branch_id: str
    parent_branch_id: Optional[str]
    branch_order: int
    is_main: bool
    length_mm: float
    mean_radius_mm: float
    distance_from_root_mm: float
    anatomical_label: str = ""
    measurements: List[CrossSectionMeasurement] = field(
        default_factory=list
    )
    candidates: List[StenosisCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class VesselResult:
    vessel: str
    label_value: int
    status: str
    centerline_length_mm: float = 0.0
    measurements: List[CrossSectionMeasurement] = field(
        default_factory=list
    )
    candidates: List[StenosisCandidate] = field(default_factory=list)
    branches: List[BranchResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    spacing_zyx: Tuple[float, float, float]
    vessels: Dict[str, VesselResult]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
