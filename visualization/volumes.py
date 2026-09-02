from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage as ndi

from ..cpr import CPRResult, generate_branch_cpr
from ..detection.cross_section import (
    make_sampling_grid,
    sample_plane,
    segment_cross_section,
)
from ..models import DetectionResult, DetectorConfig, StenosisCandidate


def build_stenosis_marker_volume(
    result: DetectionResult,
    shape: Sequence[int],
    marker_radius_mm: float = 1.5,
) -> np.ndarray:
    """Create a multiclass volume marking all detected stenosis segments."""
    if marker_radius_mm <= 0:
        raise ValueError("marker_radius_mm must be positive.")
    shape = tuple(int(value) for value in shape)
    spacing = np.asarray(result.spacing_zyx, dtype=float)
    marker = np.zeros(shape, dtype=np.uint8)

    for vessel in result.vessels.values():
        seeds = np.zeros(shape, dtype=bool)
        if vessel.branches:
            branch_items = [
                (branch.measurements, branch.candidates)
                for branch in vessel.branches
            ]
        else:
            branch_items = [
                (vessel.measurements, vessel.candidates)
            ]
        for measurements, candidates in branch_items:
            for candidate in candidates:
                for measurement in measurements:
                    if (
                        candidate.start_distance_mm
                        <= measurement.distance_mm
                        <= candidate.end_distance_mm
                    ):
                        voxel = np.rint(
                            measurement.center_voxel_zyx
                        ).astype(int)
                        voxel = np.clip(
                            voxel, 0, np.asarray(shape) - 1
                        )
                        seeds[tuple(voxel)] = True
        if not np.any(seeds):
            continue
        region = ndi.distance_transform_edt(~seeds, sampling=spacing) <= marker_radius_mm
        marker[region & (marker == 0)] = np.uint8(vessel.label_value)
    return marker


def build_anatomical_centerline_volume(
    result: DetectionResult,
    shape: Sequence[int],
    line_radius_mm: float = 0.6,
):
    """Create an integer centerline volume and its anatomical name map."""
    if line_radius_mm <= 0:
        raise ValueError("line_radius_mm must be positive.")
    shape = tuple(int(value) for value in shape)
    spacing = np.asarray(result.spacing_zyx, dtype=float)
    label_to_value = {}
    seeds_by_label = {}

    for vessel in result.vessels.values():
        branch_items = (
            vessel.branches
            if vessel.branches
            else [vessel]
        )
        for branch in branch_items:
            for measurement in branch.measurements:
                anatomical_label = measurement.anatomical_label
                if not anatomical_label:
                    continue
                if anatomical_label not in label_to_value:
                    value = len(label_to_value) + 1
                    label_to_value[anatomical_label] = value
                    seeds_by_label[anatomical_label] = np.zeros(
                        shape, dtype=bool
                    )
                voxel = np.rint(
                    measurement.center_voxel_zyx
                ).astype(int)
                voxel = np.clip(voxel, 0, np.asarray(shape) - 1)
                seeds_by_label[anatomical_label][tuple(voxel)] = True

    volume = np.zeros(shape, dtype=np.uint16)
    for anatomical_label, value in label_to_value.items():
        seeds = seeds_by_label[anatomical_label]
        if not np.any(seeds):
            continue
        region = (
            ndi.distance_transform_edt(~seeds, sampling=spacing)
            <= line_radius_mm
        )
        volume[region & (volume == 0)] = np.uint16(value)
    value_to_label = {
        str(value): anatomical_label
        for anatomical_label, value in label_to_value.items()
    }
    return volume, value_to_label
