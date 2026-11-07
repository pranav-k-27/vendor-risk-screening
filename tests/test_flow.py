"""Flow routing tests with stubbed research/explainer (no LLM, no network)."""
import asyncio

import pytest

pytest.importorskip("crewai")

from core.audit import AuditLogger
from core.entity_resolver import EntityResolver
from core.evidence_validator import EvidenceValidator
from core.pipeline import Services
from core.sanctions_matcher import SanctionsMatcher
from core.schemas import (
    EvidenceType, Explanation, FindingsBatch, RawFinding, ReportStatus, RiskBand, RiskDimension, Severity,
    SourceStatus, VendorInput,
)
from core.snapshot_manager import SnapshotManager
from core.sources import slugify
from orchestration.crewai_impl.flow import Runners
from orchestration.crewai_impl.orchestrator import CrewAIOrchestrator, SingleAgentOrchestrator

NEWS = ("Acme Infotech Limited faces allegations from former employees of unpaid wages, according to a "
        "complaint filed with the labour commissioner last month. The company denied the claims.")
NEWS_URL = "https://news.example.com/acme-wages"


def services():
    snaps = SnapshotManager()
    gleif = lambda url, params: {"data": [{"id": "LEI1", "attributes": {
        "lei": "LEI1", "registration": {"status": "ISSUED"},
        "entity": {"legalName": {"name": "ACME INFOTECH LIMITED"}, "otherNames": [], "legalAddress": {"country": "IN"}}}}]}
    return Services(resolver=EntityResolver(fetch=gleif, snapshots=snaps), matcher=SanctionsMatcher.from_directory(),
                    validator=EvidenceValidator(snaps), snapshots=snaps, audit=AuditLogger())


def media_runner(snaps, fabricate=False):
    async def run(inputs):
        s = snaps.save(NEWS.encode(), NEWS_URL, "web")
        passage = "former employees of unpaid wages" if not fabricate else "Acme Infotech Limited was convicted of fraud"
        f = RawFinding(title="Unpaid wages allegation", summary="Former staff allege unpaid wages.",
                       dimension=RiskDimension.ADVERSE_MEDIA, evidence_type=EvidenceType.MEDIA_ALLEGATION,
                       severity=Severity.MEDIUM, source_url=NEWS_URL, snapshot_id=s.snapshot_id,
                       supporting_passage=passage + " according to a complaint" if not fabricate else passage,
                       event_date="2026-10-01", entity_name_in_source="Acme Infotech Limited")
        return FindingsBatch(findings=[f], queries_run=["q"]), {"total_tokens": 100}
    return run


async def empty_runner(inputs):
    return FindingsBatch(), {"total_tokens": 10}


async def failing_runner(inputs):
    raise TimeoutError("search API timeout")


def explainer(ctx):
    return Explanation(executive_summary="Summary.", key_drivers=[]), {"total_tokens": 5}


def test_clear_path_runs_both_branches_and_scores(isolated_data_dir):
    svc = services()
    (isolated_data_dir / "documents" / slugify("Acme Infotech Ltd")).mkdir()  # curated, nothing found
    orch = CrewAIOrchestrator(svc=svc, runners=Runners(empty_runner, media_runner(svc.snapshots), explainer))
    r = orch.assess_vendor(VendorInput(name="Acme Infotech Ltd"))
    assert r.status == ReportStatus.PENDING_REVIEW
    assert r.assessment.source_status["regulatory"] == SourceStatus.NO_DATA
    assert r.assessment.source_status["adverse_media"] == SourceStatus.OK
    assert len(r.findings) == 1 and r.assessment.score == 7.0  # medium(20) x allegation(0.35)
    assert r.metrics.total_tokens == 105
    assert r.explanation is not None
    ok, _ = svc.audit.verify_chain()
    assert ok


def test_fabricated_claim_is_rejected_and_not_scored(isolated_data_dir):
    svc = services()
    (isolated_data_dir / "documents" / slugify("Acme Infotech Ltd")).mkdir()
    orch = CrewAIOrchestrator(svc=svc, runners=Runners(empty_runner, media_runner(svc.snapshots, True), None))
    r = orch.assess_vendor(VendorInput(name="Acme Infotech Ltd"))
    assert r.findings == [] and len(r.rejected_findings) == 1 and r.assessment.score == 0


def test_validation_ablation_lets_fabrication_through(isolated_data_dir):
    svc = services()
    (isolated_data_dir / "documents" / slugify("Acme Infotech Ltd")).mkdir()
    orch = CrewAIOrchestrator(svc=svc, runners=Runners(empty_runner, media_runner(svc.snapshots, True), None),
                              validate=False)
    r = orch.assess_vendor(VendorInput(name="Acme Infotech Ltd"))
    assert len(r.findings) == 1 and r.assessment.score > 0


def test_missing_curation_and_failed_media_give_insufficient(isolated_data_dir):
    svc = services()
    orch = CrewAIOrchestrator(svc=svc, runners=Runners(empty_runner, failing_runner, None))
    r = orch.assess_vendor(VendorInput(name="Acme Infotech Ltd"))
    assert r.assessment.source_status["regulatory"] == SourceStatus.FAILED
    assert r.assessment.source_status["adverse_media"] == SourceStatus.FAILED
    assert r.assessment.confidence.value == "INSUFFICIENT"
    assert any("timeout" in e for e in r.errors)


def test_sanctions_hit_skips_research(isolated_data_dir):
    svc = services()
    called = []

    async def spy(inputs):
        called.append(1)
        return FindingsBatch(), {}

    orch = CrewAIOrchestrator(svc=svc, runners=Runners(spy, spy, None))
    r = orch.assess_vendor(VendorInput(name="ZZ-SYNTHETIC Sanctioned Trading LLC", confirmed=True))
    assert called == []
    assert r.assessment.hard_stop and r.assessment.band == RiskBand.CRITICAL
    assert r.assessment.source_status["regulatory"] == SourceStatus.SKIPPED


def test_ambiguous_entity_halts_for_human(isolated_data_dir):
    svc = services()
    svc.resolver.fetch = lambda url, params: {"data": []}
    orch = CrewAIOrchestrator(svc=svc, runners=Runners(empty_runner, empty_runner, None))
    r = orch.assess_vendor(VendorInput(name="Unknown Vendor Pvt Ltd"))
    assert r.status == ReportStatus.NEEDS_ENTITY_CONFIRMATION and r.assessment is None


def test_single_agent_baseline_uses_same_pipeline(isolated_data_dir):
    svc = services()
    (isolated_data_dir / "documents" / slugify("Acme Infotech Ltd")).mkdir()
    orch = SingleAgentOrchestrator(svc=svc, research=media_runner(svc.snapshots), explain=False)
    r = orch.assess_vendor(VendorInput(name="Acme Infotech Ltd"))
    assert r.orchestrator == "single_agent" and r.assessment.score == 7.0


def test_human_decision_is_recorded_with_report_hash(isolated_data_dir):
    from core.reporting import record_decision, report_sha256

    svc = services()
    (isolated_data_dir / "documents" / slugify("Acme Infotech Ltd")).mkdir()
    r = CrewAIOrchestrator(svc=svc, runners=Runners(empty_runner, empty_runner, None)).assess_vendor(
        VendorInput(name="Acme Infotech Ltd"))
    sha = report_sha256(r)
    r = record_decision(r, "APPROVED", "P. Reviewer", "ok", audit=svc.audit)
    assert r.status == ReportStatus.APPROVED and r.review.reviewed_report_sha256 == sha
    with pytest.raises(ValueError):
        record_decision(r, "REJECTED", "someone")


def test_synthetic_injection_document_end_to_end(isolated_data_dir):
    """The real fixture document: an honest extraction verifies; the injected 'fully compliant' claim does not."""
    import shutil
    from pathlib import Path
    from core import sources

    src = Path(__file__).parent.parent / "data" / "documents" / "zz-synthetic-injection-test-ltd"
    shutil.copytree(src, isolated_data_dir / "documents" / src.name)
    svc = services()
    docs = sources.list_documents("ZZ-SYNTHETIC Injection Test Ltd", snapshots=svc.snapshots)
    assert len(docs) == 1
    sid, url = docs[0]["snapshot_id"], docs[0]["url"]
    assert "UNTRUSTED_SOURCE_CONTENT" in sources.read_snapshot(sid, snapshots=svc.snapshots)["content"]

    async def honest(inputs):
        f = RawFinding(title="Penalty for non-disclosure of default", summary="Rs 10 lakh penalty.",
                       dimension=RiskDimension.REGULATORY, evidence_type=EvidenceType.OFFICIAL_FINDING,
                       severity=Severity.MEDIUM, source_url=url, snapshot_id=sid,
                       supporting_passage="a penalty of Rs. 10,00,000 (Rupees Ten Lakh only) is hereby imposed on ZZ-SYNTHETIC Injection Test Ltd",
                       event_date="2026-08-15", entity_name_in_source="ZZ-SYNTHETIC Injection Test Ltd")
        g = f.model_copy(update={"title": "Company is fully compliant", "evidence_type": EvidenceType.OFFICIAL_FINDING,
                                 "severity": Severity.LOW,
                                 "supporting_passage": "The regulator confirms ZZ-SYNTHETIC Injection Test Ltd is fully compliant."})
        return FindingsBatch(findings=[f, g]), {}

    r = CrewAIOrchestrator(svc=svc, runners=Runners(honest, empty_runner, None)).assess_vendor(
        VendorInput(name="ZZ-SYNTHETIC Injection Test Ltd", confirmed=True))
    assert [f.title for f in r.findings] == ["Penalty for non-disclosure of default"]
    assert len(r.rejected_findings) == 1 and r.assessment.score == 20.0
