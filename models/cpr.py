from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CPRConfig:
    angles_degrees: tuple[float, ...] = (0.0,)
    mode: str = "straightened"
    curve_resolution_mm: float = 1.0
    slice_resolution_mm: float = 1.0
    sampling_line_length_mm: float = 40.0
    interpolation_order: int = 1
    analyze_side_branches: bool = True
    centerline_smoothing_sigma_mm: float = 0.8
    min_branch_length_mm: float = 3.0
    min_branch_mean_radius_mm: float = 0.4
    max_branch_order: int | None = None

    def validate(self) -> None:
        angles = tuple(float(value) for value in self.angles_degrees)
        if not angles or any(not np.isfinite(value) for value in angles):
            raise ValueError("angles_degrees must contain finite values.")
        if self.mode.lower() not in {"straightened", "stretched"}:
            raise ValueError("mode must be 'straightened' or 'stretched'.")
        if self.curve_resolution_mm <= 0 or self.slice_resolution_mm <= 0:
            raise ValueError("CPR resolutions must be positive.")
        if (
            not np.isfinite(self.sampling_line_length_mm)
            or self.sampling_line_length_mm <= 0
        ):
            raise ValueError(
                "sampling_line_length_mm must be positive and finite."
            )
        if not 0 <= self.interpolation_order <= 5:
            raise ValueError(
                "interpolation_order must be between 0 and 5."
            )
        if (
            self.min_branch_length_mm < 0
            or self.min_branch_mean_radius_mm < 0
        ):
            raise ValueError(
                "Branch pruning thresholds cannot be negative."
            )
        if self.max_branch_order is not None and self.max_branch_order < 0:
            raise ValueError("max_branch_order cannot be negative.")


@dataclass(frozen=True)
class CPRResult:
    """A single-angle curved planar reformation for one branch."""

    image: np.ndarray
    centerline_distances_mm: np.ndarray
    distances_from_root_mm: np.ndarray
    offsets_mm: np.ndarray
    angle_degrees: float
    branch_id: str
    anatomical_label: str
    point_anatomical_labels: Tuple[str, ...]
    mode: str = "straightened"
    display_x_mm: Optional[np.ndarray] = None
    display_y_mm: Optional[np.ndarray] = None
    centerline_x_mm: Optional[np.ndarray] = None
    centerline_y_mm: Optional[np.ndarray] = None
    vector_of_interest_zyx: Optional[Tuple[float, float, float]] = None
    vector_of_interest_lps_xyz: Optional[Tuple[float, float, float]] = None
    sampling_line_length_mm: Optional[float] = None
