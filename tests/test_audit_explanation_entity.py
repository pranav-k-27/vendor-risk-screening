import json

from core.audit import AuditLogger
from core.entity_resolver import EntityResolver
from core.explanation_check import check
from core.schemas import (
    ConfidenceLevel, EntityStatus, Explanation, RiskAssessment, RiskBand, SourceStatus, VendorInput,
)
from tests.test_risk_engine import vf


# ---------------- audit ----------------
def test_audit_chain_detects_tampering(isolated_data_dir):
    log = AuditLogger()
    for i in range(3):
        log.log("run1", "event", {"i": i})
    assert log.verify_chain() == (True, None)
    lines = log.path.read_text().splitlines()
    rec = json.loads(lines[1]); rec["payload"]["i"] = 99
    lines[1] = json.dumps(rec)
    log.path.write_text("\n".join(lines) + "\n")
    assert log.verify_chain() == (False, 2)


# ---------------- explanation check ----------------
def _assessment(score=40.0, band=RiskBand.MEDIUM, conf=ConfidenceLevel.HIGH):
    return RiskAssessment(score=score, band=band, confidence=conf, completeness=1.0, dimension_scores={},
                          contributions=[], source_status={"sanctions": SourceStatus.OK}, rules_version="t")


def test_consistent_explanation_passes():
    e = Explanation(executive_summary="The vendor is rated MEDIUM risk with a score of 40.0, driven by [F-00000001].",
                    cited_finding_ids=["F-00000001"])
    assert check(e, _assessment(), [vf(1)]).consistent


def test_explanation_that_changes_band_or_score_or_invents_ids_fails():
    e = Explanation(executive_summary="Overall this is HIGH risk; score 72. See [F-deadbeef].")
    r = check(e, _assessment(), [vf(1)])
    assert not r.consistent and len(r.issues) == 3


def test_calling_vendor_clean_with_insufficient_evidence_fails():
    e = Explanation(executive_summary="No adverse items; the vendor appears safe to onboard.")
    r = check(e, _assessment(0, RiskBand.LOW, ConfidenceLevel.INSUFFICIENT), [])
    assert not r.consistent


# ---------------- entity resolver ----------------
def _record(lei, name, others=(), status="ISSUED"):
    return {"id": lei, "attributes": {"lei": lei, "registration": {"status": status}, "entity": {
        "legalName": {"name": name}, "otherNames": [{"name": o} for o in others],
        "legalAddress": {"country": "IN", "city": "Pune"}}}}


def test_clear_registry_winner_is_auto_resolved(isolated_data_dir):
    fake = lambda url, params: {"data": [_record("LEI1", "ACME INFOTECH LIMITED", ["Acme Infotech"]),
                                         _record("LEI2", "ACME INFRA PROJECTS LIMITED")]}
    r = EntityResolver(fetch=fake).resolve(VendorInput(name="Acme Infotech Ltd"))
    assert r.status == EntityStatus.RESOLVED and r.lei == "LEI1"
    assert "Acme Infotech" in r.aliases


def test_ambiguous_names_need_a_human(isolated_data_dir):
    fake = lambda url, params: {"data": [_record("LEI1", "ACME INFOTECH LIMITED"),
                                         _record("LEI2", "ACME INFOTECH SERVICES LIMITED")]}
    r = EntityResolver(fetch=fake).resolve(VendorInput(name="Acme Infotech"))
    assert r.status in (EntityStatus.RESOLVED, EntityStatus.NEEDS_CONFIRMATION)
    fake2 = lambda url, params: {"data": [_record("LEI1", "ACME INFOTECH LIMITED"),
                                          _record("LEI3", "ACME INFOTECH LTD")]}
    r2 = EntityResolver(fetch=fake2).resolve(VendorInput(name="Acme Infotech"))
    assert r2.status == EntityStatus.NEEDS_CONFIRMATION and r2.lei is None


def test_registry_failure_does_not_crash(isolated_data_dir):
    def boom(url, params):
        raise ConnectionError("down")
    r = EntityResolver(fetch=boom).resolve(VendorInput(name="Acme Infotech"))
    assert r.status == EntityStatus.UNRESOLVED and r.confidence == 0


def test_user_confirmed_skips_registry(isolated_data_dir):
    r = EntityResolver(fetch=lambda *a: (_ for _ in ()).throw(AssertionError("should not call"))).resolve(
        VendorInput(name="ZZ-SYNTHETIC Test Ltd", confirmed=True))
    assert r.status == EntityStatus.USER_CONFIRMED
