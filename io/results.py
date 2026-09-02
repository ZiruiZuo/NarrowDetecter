from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..models import DetectionResult, StenosisCandidate


def read_stenosis_candidates(
    result_path: str,
) -> dict[str, list[StenosisCandidate]]:
    """Load stenosis candidates from a detection-result JSON by branch ID."""
    path = Path(result_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Stenosis result JSON does not exist: {result_path}"
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    vessels = payload.get("vessels")
    if not isinstance(vessels, dict):
        raise ValueError(
            "Stenosis result JSON is missing the vessels mapping."
        )
    candidate_fields = {item.name for item in fields(StenosisCandidate)}
    grouped: dict[str, list[StenosisCandidate]] = {}
    for vessel_name, vessel in vessels.items():
        if not isinstance(vessel, dict):
            continue
        branches = vessel.get("branches") or []
        sources = branches if branches else [vessel]
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_branch_id = str(source.get("branch_id") or vessel_name)
            for candidate_payload in source.get("candidates") or []:
                if not isinstance(candidate_payload, dict):
                    continue
                values = {
                    key: value
                    for key, value in candidate_payload.items()
                    if key in candidate_fields
                }
                values["branch_id"] = str(
                    values.get("branch_id") or source_branch_id
                )
                if "min_center_voxel_zyx" in values:
                    values["min_center_voxel_zyx"] = tuple(
                        values["min_center_voxel_zyx"]
                    )
                if "flags" in values:
                    values["flags"] = list(values["flags"])
                try:
                    candidate = StenosisCandidate(**values)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Invalid stenosis candidate in result JSON for "
                        f"branch {source_branch_id}."
                    ) from error
                grouped.setdefault(candidate.branch_id, []).append(candidate)
    return grouped


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


def write_detection_result(
    result: DetectionResult,
    output_dir: str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "stenosis_result.json"
    payload = _finite_json(result.to_dict())
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    return result_path
