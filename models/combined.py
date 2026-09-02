from __future__ import annotations

from dataclasses import dataclass, field

from .cpr import CPRResult
from .detection import DetectionResult


@dataclass
class CombinedResult:
    detection: DetectionResult
    cpr_results: list[CPRResult]
    detection_output_dir: str
    cpr_output_dir: str
    warnings: list[str] = field(default_factory=list)
