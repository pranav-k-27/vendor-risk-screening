from pathlib import Path

from core.evidence_validator import EvidenceValidator
from core.schemas import EvidenceType, RawFinding, RiskDimension, Severity

SOURCE = (
    "SEBI order dated 12 March 2025. The Securities and Exchange Board of India imposed a penalty of "
    "Rs 25 lakh on Acme Infotech Limited for delayed disclosure of material events under the LODR "
    "Regulations. The company may appeal before the Securities Appellate Tribunal."
)
URL = "https://www.sebi.gov.in/enforcement/orders/mar-2025/acme.html"


def _finding(snapshot_id, passage, url=URL, **kw):
    base = dict(
        title="SEBI penalty for delayed disclosure",
        summary="SEBI imposed a Rs 25 lakh penalty for delayed disclosures.",
        dimension=RiskDimension.REGULATORY,
        evidence_type=EvidenceType.OFFICIAL_FINDING,
        severity=Severity.MEDIUM,
        source_url=url,
        snapshot_id=snapshot_id,
        supporting_passage=passage,
        event_date="2025-03-12",
        entity_name_in_source="Acme Infotech Limited",
    )
    base.update(kw)
    return RawFinding(**base)


def test_snapshot_is_content_addressed_and_verifiable(snapshots):
    a = snapshots.save(SOURCE.encode(), URL, "regulatory_document")
    b = snapshots.save(SOURCE.encode(), URL, "regulatory_document")
    assert a.snapshot_id == b.snapshot_id == a.sha256[:16]
    assert len(a.sha256) == 64
    assert snapshots.verify(a.snapshot_id)
    Path(a.content_path).write_bytes(b"tampered")
    assert not snapshots.verify(a.snapshot_id)


def test_real_passage_is_verified(snapshots, entity):
    s = snapshots.save(SOURCE.encode(), URL, "regulatory_document")
    v = EvidenceValidator(snapshots)
    vf = v.validate_one(_finding(s.snapshot_id, "imposed a penalty of Rs 25 lakh on Acme Infotech Limited for delayed disclosure"), "regulatory", entity)
    assert vf.verified, vf.rejection_reasons


def test_fabricated_passage_is_rejected(snapshots, entity):
    s = snapshots.save(SOURCE.encode(), URL, "regulatory_document")
    vf = EvidenceValidator(snapshots).validate_one(
        _finding(s.snapshot_id, "SEBI barred Acme Infotech Limited from the securities market for five years"),
        "regulatory", entity)
    assert not vf.verified
    assert any("not found in source" in r for r in vf.rejection_reasons)


def test_unknown_snapshot_and_url_mismatch_rejected(snapshots, entity):
    s = snapshots.save(SOURCE.encode(), URL, "regulatory_document")
    v = EvidenceValidator(snapshots)
    assert "snapshot_id not found" in v.validate_one(_finding("deadbeefdeadbeef", "x" * 30), "r", entity).rejection_reasons[0]
    vf = v.validate_one(_finding(s.snapshot_id, "imposed a penalty of Rs 25 lakh on Acme Infotech Limited",
                                 url="https://example.com/other"), "r", entity)
    assert "source_url does not match snapshot" in vf.rejection_reasons


def test_source_about_another_company_is_rejected(snapshots, entity):
    other = SOURCE.replace("Acme Infotech Limited", "Zenith Softlabs Limited")
    s = snapshots.save(other.encode(), URL, "regulatory_document")
    vf = EvidenceValidator(snapshots).validate_one(
        _finding(s.snapshot_id, "imposed a penalty of Rs 25 lakh on Zenith Softlabs Limited"), "r", entity)
    assert "source does not mention the resolved entity" in vf.rejection_reasons


def test_same_event_from_regulator_and_media_counted_once(snapshots, entity):
    s1 = snapshots.save(SOURCE.encode(), URL, "regulatory_document")
    news = "Business Daily: Acme Infotech Limited was fined Rs 25 lakh by SEBI for delayed disclosure, the regulator said."
    nurl = "https://news.example.com/acme-fined"
    s2 = snapshots.save(news.encode(), nurl, "web")
    reg = _finding(s1.snapshot_id, "imposed a penalty of Rs 25 lakh on Acme Infotech Limited")
    med = _finding(s2.snapshot_id, "Acme Infotech Limited was fined Rs 25 lakh by SEBI for delayed disclosure",
                   url=nurl, dimension=RiskDimension.ADVERSE_MEDIA, evidence_type=EvidenceType.MEDIA_CONFIRMED,
                   title="SEBI penalty for delayed disclosure reported")
    verified, rejected = EvidenceValidator(snapshots).validate({"regulatory": [reg], "adverse_media": [med]}, entity)
    assert rejected == []
    assert len(verified) == 1 and verified[0].evidence_type == EvidenceType.OFFICIAL_FINDING
