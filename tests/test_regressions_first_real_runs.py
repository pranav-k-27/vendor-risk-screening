"""Regressions found in the first real (live) evaluation runs."""
from datetime import date
from pathlib import Path

from core.evidence_validator import EvidenceValidator, unsupported_figures
from core.explanation_check import check
from core.risk_engine import score
from core.schemas import (
    ConfidenceLevel, EvidenceType, Explanation, RawFinding, RiskAssessment, RiskBand, RiskDimension, Severity,
    SourceStatus,
)
from core.text import normalize_text
from tests.test_risk_engine import ALL_OK, clean_sanctions, vf

TODAY = date(2026, 11, 4)

ORDER = (
    "ADJUDICATION ORDER dated 12 April 2021 in respect of Acme Infotech Limited.\n"
    "A penalty of Rs. 25 crore is imposed on Acme Infotech Limited for mis-selling of AT-1 bonds.\n"
    "A penalty of Rs. 50 lakh is imposed on Mr. A, Managing Director of the company.\n"
    "A penalty of Rs. 50 lakh is imposed on Mr. B, Head of Private Wealth Management.\n"
)
ORDER_URL = "https://www.sebi.gov.in/enforcement/orders/apr-2021/acme.html"


def test_txt_snapshot_keeps_raw_bytes_separate_from_extracted_text(snapshots):
    raw = "line one\nline two\n".encode("utf-8")  # LF only: must survive on Windows
    snap = snapshots.save(raw, "https://example.com/a.txt", "web", text="line one\nline two\n", suffix=".txt")
    assert snap.content_path != snap.text_path
    assert Path(snap.content_path).read_bytes() == raw
    assert snapshots.verify(snap.snapshot_id)
    assert snapshots.read_text(snap.snapshot_id) == "line one\nline two\n"


def test_snapshot_with_missing_files_is_rewritten(snapshots):
    raw = b"some source text about Acme"
    snap = snapshots.save(raw, "https://example.com/b", "web", suffix=".txt")
    Path(snap.text_path).unlink()
    again = snapshots.save(raw, "https://example.com/b", "web", suffix=".txt")
    assert Path(again.text_path).exists() and snapshots.verify(again.snapshot_id)


def _order_finding(sid, passage, sev, title):
    return RawFinding(
        title=title, summary="SEBI adjudication order.", dimension=RiskDimension.REGULATORY,
        evidence_type=EvidenceType.OFFICIAL_FINDING, severity=Severity(sev), source_url=ORDER_URL,
        snapshot_id=sid, supporting_passage=passage, event_date="2021-04-12",
        entity_name_in_source="Acme Infotech Limited",
    )


def test_three_findings_from_one_order_count_once_and_score_medium(snapshots, entity):
    sid = snapshots.save(ORDER.encode(), ORDER_URL, "regulatory_document", suffix=".txt").snapshot_id
    raw = [
        _order_finding(sid, "A penalty of Rs. 50 lakh is imposed on Mr. A, Managing Director", "medium",
                       "Penalty on managing director"),
        _order_finding(sid, "A penalty of Rs. 25 crore is imposed on Acme Infotech Limited", "high",
                       "SEBI penalty on Acme for AT-1 bond mis-selling"),
        _order_finding(sid, "A penalty of Rs. 50 lakh is imposed on Mr. B, Head of Private", "medium",
                       "Penalty on wealth management head"),
    ]
    verified, rejected = EvidenceValidator(snapshots).validate({"regulatory": raw}, entity)
    assert rejected == []
    assert len(verified) == 1
    kept = verified[0]
    assert kept.severity == Severity.HIGH  # strongest finding is the one kept
    assert len(kept.merged_finding_ids) == 2

    a = score(verified, clean_sanctions(), entity, ALL_OK, today=TODAY, rejected=rejected)
    assert a.band == RiskBand.MEDIUM
    assert any("counted once" in line for line in a.rationale)


def test_unsupported_figures():
    src = normalize_text("SEBI imposed a penalty of Rs. 25 crore on the company.")
    assert unsupported_figures("Penalty of Rs. 25 crores", src) == []
    assert unsupported_figures("Penalty of Rs 40 crore", src) == ["Rs 40 crore"]


def test_unsupported_figure_becomes_warning_and_flag_not_rejection(snapshots, entity):
    sid = snapshots.save(ORDER.encode(), ORDER_URL, "regulatory_document", suffix=".txt").snapshot_id
    f = _order_finding(sid, "A penalty of Rs. 25 crore is imposed on Acme Infotech Limited", "high",
                       "SEBI penalty of Rs 40 crore on Acme")
    v = EvidenceValidator(snapshots).validate_one(f, "regulatory", entity)
    assert v.verified and v.warnings
    a = score([v], clean_sanctions(), entity, ALL_OK, today=TODAY)
    assert f"UNSUPPORTED_FIGURES:{v.finding_id}" in a.review_flags


def test_many_rejected_claims_cap_confidence_at_medium(entity):
    rejected = [vf(i, verified=False) for i in range(6)]
    for r in rejected:
        r.rejection_reasons = ["supporting_passage not found in source (match 40)"]
    a = score([], clean_sanctions(), entity, ALL_OK, today=TODAY, rejected=rejected)
    assert a.confidence == ConfidenceLevel.MEDIUM
    assert "MANY_CLAIMS_REJECTED:6" in a.review_flags


def test_hash_mismatch_caps_confidence_at_low(entity):
    bad = vf(9, verified=False)
    bad.rejection_reasons = ["snapshot hash mismatch (content altered)"]
    a = score([], clean_sanctions(), entity, ALL_OK, today=TODAY, rejected=[bad])
    assert a.confidence == ConfidenceLevel.LOW
    assert "SNAPSHOT_INTEGRITY_FAILURE" in a.review_flags


def _low():
    return RiskAssessment(score=0.0, band=RiskBand.LOW, confidence=ConfidenceLevel.HIGH, completeness=1.0,
                          dimension_scores={}, contributions=[], source_status={"sanctions": SourceStatus.OK},
                          rules_version="t")


def test_negated_band_phrase_is_not_a_contradiction():
    ok = Explanation(executive_summary="There are no high risk findings; the vendor is rated LOW risk.")
    assert check(ok, _low(), []).consistent
    bad = Explanation(executive_summary="The vendor is HIGH risk.")
    assert not check(bad, _low(), []).consistent


# ---------------------------------------------------------------------------
# Deterministic guards on agent evidence classification
# ---------------------------------------------------------------------------
NEWS = ("Acme Infotech Limited was fined by SEBI. Sensitive card data, including CVVs, "
        "was allegedly compromised in a breach at Acme Infotech Limited last year.")
SEBI_PASSAGE = "Acme Infotech Limited was fined by SEBI."
BLOG_URL = "https://www.indiratrade.com/blog/acme-sebi-penalty"
SEBI_URL = "https://www.sebi.gov.in/enforcement/orders/jun-2025/acme.html"


def _claim(sid, url, ev="official_finding", passage=SEBI_PASSAGE, date_="2025-06-01", sev="high"):
    return RawFinding(
        title="SEBI fine on Acme", summary="SEBI fined the company.", dimension=RiskDimension.REGULATORY,
        evidence_type=EvidenceType(ev), severity=Severity(sev), source_url=url, snapshot_id=sid,
        supporting_passage=passage, event_date=date_, entity_name_in_source="Acme Infotech Limited",
    )


def test_official_finding_from_non_regulator_url_is_downgraded(snapshots, entity):
    sid = snapshots.save(NEWS.encode(), BLOG_URL, "web", suffix=".txt").snapshot_id
    raw = _claim(sid, BLOG_URL)
    v = EvidenceValidator(snapshots).validate_one(raw, "regulatory", entity)
    assert v.verified
    assert v.evidence_type == EvidenceType.MEDIA_CONFIRMED
    assert "evidence_type downgraded from official_finding to media_confirmed: source is not a regulator" in v.warnings
    # id comes from the original agent output, so it is unchanged by the guard
    from core.evidence_validator import _finding_id
    assert v.finding_id == _finding_id("regulatory", raw)


def test_official_finding_from_regulator_url_or_curated_document_is_kept(snapshots, entity):
    val = EvidenceValidator(snapshots)
    sid = snapshots.save(NEWS.encode(), SEBI_URL, "web", suffix=".txt").snapshot_id
    v = val.validate_one(_claim(sid, SEBI_URL), "regulatory", entity)
    assert v.evidence_type == EvidenceType.OFFICIAL_FINDING and not v.warnings

    doc = (NEWS + " (curated copy)").encode()
    sid2 = snapshots.save(doc, "file://documents/acme/order.txt", "regulatory_document", suffix=".txt").snapshot_id
    v2 = val.validate_one(_claim(sid2, "file://documents/acme/order.txt"), "regulatory", entity)
    assert v2.evidence_type == EvidenceType.OFFICIAL_FINDING and not v2.warnings


def test_hedged_media_confirmed_passage_becomes_allegation(snapshots, entity):
    sid = snapshots.save(NEWS.encode(), BLOG_URL, "news", suffix=".txt").snapshot_id
    passage = "Sensitive card data, including CVVs, was allegedly compromised"
    v = EvidenceValidator(snapshots).validate_one(
        _claim(sid, BLOG_URL, ev="media_confirmed", passage=passage), "adverse_media", entity)
    assert v.verified
    assert v.evidence_type == EvidenceType.MEDIA_ALLEGATION
    assert "evidence_type downgraded to media_allegation: passage contains 'allegedly'" in v.warnings


def test_missing_event_date_taken_from_publication_date(snapshots, entity):
    sid = snapshots.save(NEWS.encode(), SEBI_URL, "news", published_date="2025-06-03T10:00:00Z",
                         suffix=".txt").snapshot_id
    v = EvidenceValidator(snapshots).validate_one(_claim(sid, SEBI_URL, date_=None), "regulatory", entity)
    assert v.event_date == "2025-06-03"
    assert "event_date taken from source publication date" in v.warnings


def test_reclassified_finding_scores_less_and_is_flagged(snapshots, entity):
    val = EvidenceValidator(snapshots)
    sid_reg = snapshots.save(NEWS.encode(), SEBI_URL, "web", suffix=".txt").snapshot_id
    sid_blog = snapshots.save((NEWS + " ").encode(), BLOG_URL, "web", suffix=".txt").snapshot_id
    kept = val.validate_one(_claim(sid_reg, SEBI_URL), "regulatory", entity)
    down = val.validate_one(_claim(sid_blog, BLOG_URL), "regulatory", entity)
    a_kept = score([kept], clean_sanctions(), entity, ALL_OK, today=TODAY)
    a_down = score([down], clean_sanctions(), entity, ALL_OK, today=TODAY)
    assert a_down.score < a_kept.score
    assert f"EVIDENCE_RECLASSIFIED:{down.finding_id}" in a_down.review_flags
    assert not any(fl.startswith("EVIDENCE_RECLASSIFIED") for fl in a_kept.review_flags)
