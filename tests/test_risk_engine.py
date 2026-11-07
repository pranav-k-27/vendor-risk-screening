from datetime import date

from core.risk_engine import recency_multiplier, score
from core.config import load_rules
from core.schemas import (
    ConfidenceLevel, EvidenceType, MatchStrength, RiskBand, RiskDimension, SanctionsHit, SanctionsResult,
    Severity, SourceStatus, ValidatedFinding,
)

TODAY = date(2026, 11, 4)
ALL_OK = {"sanctions": SourceStatus.OK, "regulatory": SourceStatus.OK, "adverse_media": SourceStatus.OK}


def vf(i, sev="medium", ev="official_finding", dim="regulatory", d="2025-06-01", verified=True):
    return ValidatedFinding(
        finding_id=f"F-0000000{i}", agent="t", verified=verified, title=f"t{i}", summary="s",
        dimension=RiskDimension(dim), evidence_type=EvidenceType(ev), severity=Severity(sev),
        source_url="u", snapshot_id="x", supporting_passage="p" * 30, event_date=d, entity_name_in_source="e",
    )


def clean_sanctions():
    return SanctionsResult(screened_names=["x"], lists_checked=["OFAC_SDN", "UN_CONSOLIDATED"])


def test_no_findings_full_coverage_is_low_high_confidence(entity):
    a = score([], clean_sanctions(), entity, ALL_OK, today=TODAY)
    assert a.score == 0 and a.band == RiskBand.LOW and a.confidence == ConfidenceLevel.HIGH


def test_no_findings_with_failed_source_is_insufficient_not_clean(entity):
    st = dict(ALL_OK, adverse_media=SourceStatus.FAILED)
    a = score([], clean_sanctions(), entity, st, today=TODAY)
    assert a.score == 0
    assert a.confidence == ConfidenceLevel.INSUFFICIENT
    assert "INSUFFICIENT_EVIDENCE" in a.review_flags
    assert a.completeness < 1


def test_allegation_counts_less_than_official_finding(entity):
    off = score([vf(1, "high", "official_finding")], clean_sanctions(), entity, ALL_OK, today=TODAY)
    alg = score([vf(1, "high", "media_allegation", "adverse_media")], clean_sanctions(), entity, ALL_OK, today=TODAY)
    assert off.score == 40.0 and alg.score == 14.0


def test_unverified_claims_and_rejected_findings_never_score(entity):
    a = score([vf(1, "critical", "unverified", "adverse_media"), vf(2, "critical", verified=False)],
              clean_sanctions(), entity, ALL_OK, today=TODAY)
    assert a.score == 0
    assert any(f.startswith("UNVERIFIED_CLAIM") for f in a.review_flags)


def test_dimension_cap_applies(entity):
    many = [vf(i, "high", "official_finding", d="2026-01-01") for i in range(5)]  # 200 raw points
    a = score(many, clean_sanctions(), entity, ALL_OK, today=TODAY)
    assert a.dimension_scores["regulatory"] == 60
    assert a.band == RiskBand.HIGH


def test_recency_decay_and_floor():
    r = load_rules()
    assert recency_multiplier("2025-01-01", r, TODAY) == 1.0
    assert 0.3 < recency_multiplier("2020-01-01", r, TODAY) < 1.0
    assert recency_multiplier("2001-01-01", r, TODAY) == 0.3
    assert recency_multiplier(None, r, TODAY) == 0.7


def test_strong_sanctions_hit_is_hard_stop(entity):
    s = clean_sanctions()
    s.hits.append(SanctionsHit(list_name="OFAC_SDN", entry_id="1", listed_name="X", matched_on="X",
                               query_name="X", score=100, strength=MatchStrength.STRONG))
    a = score([], s, entity, ALL_OK, today=TODAY)
    assert a.hard_stop and a.score == 100 and a.band == RiskBand.CRITICAL


def test_potential_sanctions_hit_flags_but_does_not_score(entity):
    s = clean_sanctions()
    s.hits.append(SanctionsHit(list_name="OFAC_SDN", entry_id="1", listed_name="X", matched_on="X",
                               query_name="X", score=90, strength=MatchStrength.POTENTIAL))
    a = score([], s, entity, ALL_OK, today=TODAY)
    assert not a.hard_stop and a.score == 0 and "SANCTIONS_POTENTIAL_MATCH" in a.review_flags


def test_every_point_is_traceable(entity):
    fs = [vf(1, "high"), vf(2, "medium", "media_allegation", "adverse_media")]
    a = score(fs, clean_sanctions(), entity, ALL_OK, today=TODAY)
    assert round(sum(c.points for c in a.contributions), 2) == a.score
    assert {c.finding_id for c in a.contributions} == {"F-00000001", "F-00000002"}
