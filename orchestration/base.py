"""The only contract an orchestration framework has to satisfy."""
from __future__ import annotations

from typing import Protocol

from core.schemas import AssessmentReport, VendorInput


class Orchestrator(Protocol):
    name: str

    def assess_vendor(self, vendor: VendorInput) -> AssessmentReport: ...
