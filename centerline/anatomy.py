from __future__ import annotations

from typing import Optional, Sequence

from .extraction import CenterlineBranch


def build_anatomical_plan(
    vessel: str,
    branches: Sequence[CenterlineBranch],
):
    """Assign deterministic coronary names to a rooted centerline tree."""
    vessel_name = vessel.upper()
    side_branches = sorted(
        (branch for branch in branches if not branch.is_main),
        key=lambda branch: (
            branch.distance_from_root_mm,
            branch.branch_id,
        ),
    )
    labels = {}
    for index, branch in enumerate(side_branches, start=1):
        if vessel_name == "LAD":
            label = f"D_{index}"
        elif vessel_name == "LCX":
            label = f"OM_{index}"
        elif vessel_name == "RCA":
            label = "PLB" if index == 1 else f"PLB_{index}"
        else:
            label = f"{vessel_name}_BRANCH_{index}"
        labels[branch.branch_id] = label

    direct_side_branches = [
        branch
        for branch in side_branches
        if branch.parent_branch_id == 0
    ]
    split_distance = (
        min(
            branch.distance_from_root_mm
            for branch in direct_side_branches
        )
        if direct_side_branches
        else None
    )
    if vessel_name == "LAD":
        labels[0] = "LM/LAD" if split_distance is not None else "LAD"
    elif vessel_name == "LCX":
        labels[0] = "LCX"
    elif vessel_name == "RCA":
        labels[0] = "RCA/PDA" if split_distance is not None else "RCA"
    else:
        labels[0] = vessel_name
    return labels, split_distance


def point_anatomical_label(
    vessel: str,
    branch: CenterlineBranch,
    branch_label: str,
    distance_from_root_mm: float,
    main_split_distance_mm: Optional[float],
) -> str:
    """Return the point-level name on a composite main branch."""
    if not branch.is_main or main_split_distance_mm is None:
        return branch_label
    vessel_name = vessel.upper()
    if vessel_name == "LAD":
        return (
            "LM"
            if distance_from_root_mm <= main_split_distance_mm
            else "LAD"
        )
    if vessel_name == "RCA":
        return (
            "RCA"
            if distance_from_root_mm <= main_split_distance_mm
            else "PDA"
        )
    return branch_label
