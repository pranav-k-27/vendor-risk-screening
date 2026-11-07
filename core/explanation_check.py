"""Deterministic check that the LLM explanation does not contradict the engine.

The explainer may only describe the computed result. We flag:
  - citing finding_ids that do not exist / were rejected
  - naming a risk band other than the computed one
  - stating a numeric score that differs from the computed score
  - calling the vendor clean/safe when the engine reported INSUFFICIENT evidence
"""
from __future__ import annotations

import re

from core.schemas import ConfidenceLevel, Explanation, ExplanationCheck, RiskAssessment, ValidatedFinding

_BAND_WORDS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
_ID = re.compile(r"F-[0-9a-f]{8}")
_SCORE = re.compile(r"(?:score|scored)\D{0,15}(\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
_BAND_PHRASE = re.compile(r"\b(LOW|MEDIUM|HIGH|CRITICAL)\b(?=\s+(?:risk|band|rating))", re.IGNORECASE)
_NEGATED = re.compile(r"\b(no|not|nor|without|none|neither|any)\b[^.]*$", re.IGNORECASE)
_CLEAN = re.compile(r"\b(no risk|clean bill|risk[- ]free|fully compliant|safe to onboard)\b", re.IGNORECASE)


def check(explanation: Explanation, assessment: RiskAssessment, findings: list[ValidatedFinding]) -> ExplanationCheck:
    issues: list[str] = []
    text = " ".join(
        [explanation.executive_summary, *explanation.key_drivers, *explanation.evidence_gaps, *explanation.recommended_actions]
    )
    valid_ids = {f.finding_id for f in findings if f.verified}
    cited = set(explanation.cited_finding_ids) | set(_ID.findall(text))
    unknown = sorted(cited - valid_ids)
    if unknown:
        issues.append(f"cites unknown or rejected finding ids: {unknown}")

    for m in _BAND_PHRASE.finditer(text):
        if _NEGATED.search(text[max(0, m.start() - 25):m.start()]):
            continue  # "no high risk findings" is not a claim about the band
        if m.group(1).upper() != assessment.band.value:
            issues.append(f"mentions '{m.group(0)}' but computed band is {assessment.band.value}")

    for m in _SCORE.finditer(text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if abs(val - assessment.score) > 0.5:
            issues.append(f"states score {val} but computed score is {assessment.score}")

    if assessment.confidence == ConfidenceLevel.INSUFFICIENT and _CLEAN.search(text):
        issues.append("describes vendor as clean although evidence is INSUFFICIENT")

    return ExplanationCheck(consistent=not issues, issues=issues)
