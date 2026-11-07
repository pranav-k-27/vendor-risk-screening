"""Shared, framework-independent pipeline steps.

Orchestrators (CrewAI Flow, single-agent baseline, later LangGraph) differ ONLY
in how the research step is run. Everything before and after is here, so the
framework comparison measures orchestration, not different business logic.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from core.audit import AuditLogger
from core.config import documents_dir
from core.entity_resolver import EntityResolver
from core.evidence_validator import EvidenceValidator
from core.explanation_check import check as check_explanation
from core.reporting import save_report
from core.risk_engine import score as score_risk
from core.sanctions_matcher import SanctionsMatcher
from core.schemas import (
    AssessmentReport,
    EntityResolution,
    EntityStatus,
    Explanation,
    FindingsBatch,
    ReportStatus,
    RunMetrics,
    SanctionsResult,
    SourceStatus,
    ValidatedFinding,
    VendorInput,
)
from core.snapshot_manager import SnapshotManager
from core.sources import slugify

REQUIRED_SANCTIONS_LISTS = {"OFAC_SDN", "UN_CONSOLIDATED"}


@dataclass
class ResearchOutcome:
    agent: str
    status: SourceStatus
    batch: FindingsBatch | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)


@dataclass
class Services:
    """Injected so tests and alternative orchestrators can swap any part."""
    resolver: EntityResolver
    matcher: SanctionsMatcher
    validator: EvidenceValidator
    snapshots: SnapshotManager
    audit: AuditLogger

    @classmethod
    def default(cls) -> "Services":
        snaps = SnapshotManager()
        return cls(
            resolver=EntityResolver(snapshots=snaps),
            matcher=SanctionsMatcher.from_directory(),
            validator=EvidenceValidator(snapshots=snaps),
            snapshots=snaps,
            audit=AuditLogger(),
        )


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


# ---------------------------------------------------------------------------
# Step 1-2: identity and sanctions (deterministic)
# ---------------------------------------------------------------------------
def resolve_entity(vendor: VendorInput, svc: Services, run_id: str) -> EntityResolution:
    entity = svc.resolver.resolve(vendor)
    svc.audit.log(run_id, "entity_resolved", entity.model_dump(mode="json", exclude={"candidates"}))
    return entity


def entity_needs_human(entity: EntityResolution) -> bool:
    return entity.status in (EntityStatus.NEEDS_CONFIRMATION, EntityStatus.UNRESOLVED)


def screen_sanctions(entity: EntityResolution, svc: Services, run_id: str) -> tuple[SanctionsResult, SourceStatus]:
    result = svc.matcher.screen(entity.all_names)
    status = SourceStatus.OK if REQUIRED_SANCTIONS_LISTS <= set(result.lists_checked) else SourceStatus.FAILED
    svc.audit.log(run_id, "sanctions_screened", {
        "lists": result.list_versions, "entries": result.entries_screened,
        "hits": [h.model_dump(mode="json") for h in result.hits], "status": status.value,
    })
    return result, status


def regulatory_precheck(vendor: VendorInput, entity: EntityResolution) -> SourceStatus | None:
    """None => run the regulatory agent. Otherwise the status to record without spending tokens."""
    folder = documents_dir() / slugify(vendor.name)
    if not folder.exists():
        alt = documents_dir() / slugify(entity.resolved_name)
        if not alt.exists():
            return SourceStatus.FAILED  # nobody curated SEBI documents for this vendor: not checked
        folder = alt
    has_docs = any(p.name != "metadata.json" for p in folder.iterdir())
    return None if has_docs else SourceStatus.NO_DATA  # empty folder = checked, nothing found


# ---------------------------------------------------------------------------
# Step 4-6: validate, score, explain, report
# ---------------------------------------------------------------------------
def explainer_context(report: AssessmentReport) -> str:
    a = report.assessment
    assert a is not None
    pts = {c.finding_id: c.points for c in a.contributions}
    ctx = {
        "vendor": report.vendor.name,
        "resolved_entity": report.entity.resolved_name if report.entity else None,
        "computed_result": {
            "score": a.score, "band": a.band.value, "confidence": a.confidence.value,
            "completeness": a.completeness, "hard_stop": a.hard_stop,
            "review_flags": a.review_flags, "rule_rationale": a.rationale,
            "source_status": {k: v.value for k, v in a.source_status.items()},
        },
        "sanctions_hits": [h.model_dump(mode="json") for h in (report.sanctions.hits if report.sanctions else [])],
        "verified_findings": [
            {"finding_id": f.finding_id, "title": f.title, "dimension": f.dimension.value,
             "evidence_type": f.evidence_type.value, "severity": f.severity.value,
             "event_date": f.event_date, "points": pts.get(f.finding_id, 0), "summary": f.summary}
            for f in report.findings
        ],
        "rejected_claims_count": len(report.rejected_findings),
    }
    return json.dumps(ctx, indent=1)


ExplainFn = Callable[[str], tuple[Explanation, dict]]


def finalize(
    *,
    run_id: str,
    orchestrator: str,
    model: str | None,
    vendor: VendorInput,
    entity: EntityResolution,
    sanctions: SanctionsResult,
    sanctions_status: SourceStatus,
    outcomes: list[ResearchOutcome],
    svc: Services,
    explain: ExplainFn | None,
    started: float,
    validate: bool = True,
    errors: list[str] | None = None,
) -> AssessmentReport:
    errors = list(errors or [])
    usage_total: dict[str, int] = {}

    source_status = {"sanctions": sanctions_status}
    batches = {}
    for o in outcomes:
        source_status[o.agent] = o.status
        if o.error:
            errors.append(f"{o.agent}: {o.error}")
        if o.batch:
            batches[o.agent] = o.batch.findings
        for k, v in (o.usage or {}).items():
            usage_total[k] = usage_total.get(k, 0) + int(v or 0)

    if validate:
        verified, rejected = svc.validator.validate(batches, entity)
    else:  # ablation: trust the agents
        verified = [svc.validator.validate_one(f, a, entity).model_copy(update={"verified": True, "rejection_reasons": []})
                    for a, fs in batches.items() for f in fs]
        rejected = []
    svc.audit.log(run_id, "evidence_validated", {
        "verified": [f.finding_id for f in verified],
        "rejected": [{"id": f.finding_id, "reasons": f.rejection_reasons} for f in rejected],
        "validation_enabled": validate,
    })

    assessment = score_risk(verified, sanctions, entity, source_status, rejected=rejected)
    svc.audit.log(run_id, "risk_scored", {
        "score": assessment.score, "band": assessment.band.value, "confidence": assessment.confidence.value,
        "completeness": assessment.completeness, "flags": assessment.review_flags, "rules_version": assessment.rules_version,
    })

    report = AssessmentReport(
        run_id=run_id, orchestrator=orchestrator, model=model, status=ReportStatus.PENDING_REVIEW,
        vendor=vendor, entity=entity, sanctions=sanctions, findings=verified, rejected_findings=rejected,
        assessment=assessment, errors=errors,
    )

    if explain is not None:
        try:
            explanation, usage = explain(explainer_context(report))
            report.explanation = explanation
            report.explanation_check = check_explanation(explanation, assessment, verified)
            for k, v in (usage or {}).items():
                usage_total[k] = usage_total.get(k, 0) + int(v or 0)
            if not report.explanation_check.consistent:
                assessment.review_flags.append("EXPLANATION_INCONSISTENT")
        except Exception as exc:
            report.errors.append(f"explainer: {exc!s}"[:400])

    report.metrics = RunMetrics(
        duration_s=round(time.time() - started, 2),
        prompt_tokens=usage_total.get("prompt_tokens", 0),
        completion_tokens=usage_total.get("completion_tokens", 0),
        total_tokens=usage_total.get("total_tokens", 0),
        successful_requests=usage_total.get("successful_requests", 0),
    )
    save_report(report)
    svc.audit.log(run_id, "report_ready_for_review", {"status": report.status.value, "errors": report.errors})
    return report


def halt_for_entity(
    *, run_id: str, orchestrator: str, model: str | None, vendor: VendorInput, entity: EntityResolution,
    svc: Services, started: float,
) -> AssessmentReport:
    report = AssessmentReport(
        run_id=run_id, orchestrator=orchestrator, model=model, status=ReportStatus.NEEDS_ENTITY_CONFIRMATION,
        vendor=vendor, entity=entity, metrics=RunMetrics(duration_s=round(time.time() - started, 2)),
    )
    save_report(report)
    svc.audit.log(run_id, "halted_for_entity_confirmation", {"candidates": [c.model_dump() for c in entity.candidates]})
    return report


def usage_from(obj) -> dict:
    """Normalise CrewAI (or other) token usage objects into a dict."""
    u = getattr(obj, "token_usage", None) or getattr(obj, "usage_metrics", None)
    if u is None:
        return {}
    if hasattr(u, "model_dump"):
        u = u.model_dump()
    return {k: u.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "successful_requests")}


def finding_ids(findings: list[ValidatedFinding]) -> list[str]:
    return [f.finding_id for f in findings]
