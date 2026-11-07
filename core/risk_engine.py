"""Deterministic risk scoring. Pure function of (findings, sanctions, entity, source status, rules).

Three separate outputs, deliberately not collapsed into one number:
  score/band    how serious are the VERIFIED findings
  confidence    how much should you trust that score (evidence quality + identity certainty)
  completeness  what share of planned sources were actually checked
Missing evidence never raises the score; it lowers confidence instead.
"""
from __future__ import annotations

from datetime import date

from core.config import load_rules
from core.schemas import (
    ConfidenceLevel,
    EntityResolution,
    EvidenceType,
    RiskAssessment,
    RiskBand,
    RiskDimension,
    SanctionsResult,
    ScoreContribution,
    SourceStatus,
    ValidatedFinding,
)

OFFICIAL_TYPES = {EvidenceType.OFFICIAL_FINDING, EvidenceType.SETTLEMENT, EvidenceType.UNDER_APPEAL}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt_len in (10, 7, 4):
        try:
            part = s[:fmt_len]
            if fmt_len == 10:
                return date.fromisoformat(part)
            if fmt_len == 7:
                y, m = part.split("-")
                return date(int(y), int(m), 1)
            return date(int(part), 1, 1)
        except (ValueError, TypeError):
            continue
    return None


def recency_multiplier(event_date: str | None, rules: dict, today: date | None = None) -> float:
    cfg = rules["scoring"]["recency"]
    d = _parse_date(event_date)
    if d is None:
        return float(cfg["undated_multiplier"])
    today = today or date.today()
    age_years = max(0.0, (today - d).days / 365.25)
    if age_years <= cfg["full_weight_years"]:
        return 1.0
    decayed = 1.0 - cfg["decay_per_year"] * (age_years - cfg["full_weight_years"])
    return round(max(float(cfg["floor"]), decayed), 3)


def band_for(score: float, rules: dict) -> RiskBand:
    for name in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        if score <= float(rules["scoring"]["bands"][name]):
            return RiskBand(name)
    return RiskBand.CRITICAL


def score(
    findings: list[ValidatedFinding],
    sanctions: SanctionsResult | None,
    entity: EntityResolution | None,
    source_status: dict[str, SourceStatus],
    rules: dict | None = None,
    today: date | None = None,
    rejected: list[ValidatedFinding] | None = None,
) -> RiskAssessment:
    rules = rules or load_rules()
    sc = rules["scoring"]
    rationale: list[str] = []
    flags: list[str] = []
    contributions: list[ScoreContribution] = []

    # ---- sanctions gate -------------------------------------------------
    if sanctions and sanctions.has_strong_hit:
        top = next(h for h in sanctions.hits if h.strength.value == "strong")
        rationale.append(
            f"Strong sanctions match: '{top.query_name}' vs {top.list_name} entry {top.entry_id} "
            f"'{top.matched_on}' (score {top.score}). Hard stop pending human review."
        )
        flags.append("SANCTIONS_STRONG_MATCH")
        return RiskAssessment(
            score=float(sc["sanctions_strong_hit_score"]),
            band=RiskBand.CRITICAL,
            confidence=ConfidenceLevel.HIGH if top.score >= 99 else ConfidenceLevel.MEDIUM,
            completeness=_completeness(source_status),
            dimension_scores={"sanctions": float(sc["sanctions_strong_hit_score"])},
            contributions=[],
            hard_stop=True,
            review_flags=flags,
            source_status=source_status,
            rules_version=str(rules["version"]),
            rationale=rationale,
        )
    if sanctions and sanctions.has_potential_hit:
        flags.append("SANCTIONS_POTENTIAL_MATCH")
        rationale.append(
            f"{sum(1 for h in sanctions.hits)} potential sanctions match(es) below the strong threshold; "
            "not scored, reviewer must clear them."
        )

    # ---- findings -------------------------------------------------------
    dim_raw: dict[str, float] = {d.value: 0.0 for d in RiskDimension}
    for f in findings:
        if not f.verified:
            continue
        base = float(sc["severity_points"][f.severity.value])
        ev = float(sc["evidence_multiplier"][f.evidence_type.value])
        rec = recency_multiplier(f.event_date, rules, today)
        pts = round(base * ev * rec, 2)
        contributions.append(
            ScoreContribution(
                finding_id=f.finding_id,
                dimension=f.dimension,
                base_points=base,
                evidence_multiplier=ev,
                recency_multiplier=rec,
                points=pts,
                rule=f"{f.severity.value}({base}) x {f.evidence_type.value}({ev}) x recency({rec})",
            )
        )
        dim_raw[f.dimension.value] += pts
        if f.evidence_type == EvidenceType.UNVERIFIED:
            flags.append(f"UNVERIFIED_CLAIM:{f.finding_id}")
        if any(w.startswith("amounts in title/summary") for w in f.warnings):
            flags.append(f"UNSUPPORTED_FIGURES:{f.finding_id}")
        if any("downgraded" in w for w in f.warnings):
            flags.append(f"EVIDENCE_RECLASSIFIED:{f.finding_id}")
        if f.merged_finding_ids:
            rationale.append(
                f"{f.finding_id} and {', '.join(f.merged_finding_ids)} describe the same event and were counted once."
            )

    dim_scores = {}
    for d, raw in dim_raw.items():
        cap = float(sc["dimension_caps"][d])
        dim_scores[d] = round(min(raw, cap), 2)
        if raw > cap:
            rationale.append(f"{d} raw points {raw:.1f} capped at {cap:.0f}.")
    total = round(min(100.0, sum(dim_scores.values())), 2)
    band = band_for(total, rules)

    scored = [c for c in contributions if c.points > 0]
    if scored:
        top = sorted(scored, key=lambda c: -c.points)[:3]
        rationale.append("Top contributors: " + ", ".join(f"{c.finding_id} ({c.points} pts)" for c in top))
    else:
        rationale.append("No verified, scorable findings.")

    completeness = _completeness(source_status)
    confidence = _confidence(findings, contributions, entity, completeness, rules)
    rejected = rejected or []
    if any("hash mismatch" in r for f in rejected for r in f.rejection_reasons):
        flags.append("SNAPSHOT_INTEGRITY_FAILURE")
        rationale.append("At least one stored source failed its integrity check; confidence capped at LOW.")
        confidence = _cap(confidence, ConfidenceLevel.LOW)
    n_verified = sum(1 for f in findings if f.verified)
    if len(rejected) >= 3 and len(rejected) > n_verified:
        flags.append(f"MANY_CLAIMS_REJECTED:{len(rejected)}")
        rationale.append(
            f"{len(rejected)} agent claims were rejected vs {n_verified} verified; confidence capped at MEDIUM."
        )
        confidence = _cap(confidence, ConfidenceLevel.MEDIUM)
    if confidence == ConfidenceLevel.INSUFFICIENT:
        flags.append("INSUFFICIENT_EVIDENCE")
        rationale.append("Not all sources could be checked and nothing was found: a low score here is NOT a clean bill.")
    failed = [k for k, v in source_status.items() if v == SourceStatus.FAILED]
    if failed:
        flags.append("SOURCE_FAILURE:" + ",".join(failed))
    if entity and entity.confidence < rules["confidence"]["entity_confidence_high"]:
        flags.append("ENTITY_IDENTITY_UNCERTAIN")

    return RiskAssessment(
        score=total,
        band=band,
        confidence=confidence,
        completeness=completeness,
        dimension_scores=dim_scores,
        contributions=contributions,
        hard_stop=False,
        review_flags=flags,
        source_status=source_status,
        rules_version=str(rules["version"]),
        rationale=rationale,
    )


_CONF_ORDER = [ConfidenceLevel.INSUFFICIENT, ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]


def _cap(level: ConfidenceLevel, ceiling: ConfidenceLevel) -> ConfidenceLevel:
    """Lower `level` to `ceiling` if it is above it; never raise it."""
    return min(level, ceiling, key=_CONF_ORDER.index)


def _completeness(source_status: dict[str, SourceStatus]) -> float:
    planned = [v for v in source_status.values() if v != SourceStatus.SKIPPED]
    if not planned:
        return 0.0
    ok = sum(1 for v in planned if v in (SourceStatus.OK, SourceStatus.NO_DATA))
    return round(ok / len(planned), 3)


def _confidence(findings, contributions, entity, completeness, rules) -> ConfidenceLevel:
    cfg = rules["confidence"]
    scored_pts = sum(c.points for c in contributions)
    if scored_pts == 0 and completeness < cfg["min_completeness"]:
        return ConfidenceLevel.INSUFFICIENT
    entity_conf = entity.confidence if entity else 0.0
    if scored_pts == 0:
        # clean result: trust depends on coverage and identity certainty
        return ConfidenceLevel.HIGH if entity_conf >= cfg["entity_confidence_high"] else ConfidenceLevel.MEDIUM
    official_pts = sum(
        c.points for c in contributions
        if any(f.finding_id == c.finding_id and f.evidence_type in OFFICIAL_TYPES for f in findings)
    )
    share = official_pts / scored_pts
    if share >= cfg["official_share_high"] and entity_conf >= cfg["entity_confidence_high"] and completeness >= 1:
        return ConfidenceLevel.HIGH
    if share > 0 or completeness >= 1:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
