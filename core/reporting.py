"""Persist reports, render Markdown, and record the human decision."""
from __future__ import annotations

import hashlib
from pathlib import Path

from core.audit import AuditLogger
from core.config import runs_dir
from core.schemas import AssessmentReport, ReportStatus, ReviewDecision


def report_sha256(report: AssessmentReport) -> str:
    """Hash of the report as presented for review (review field excluded)."""
    return hashlib.sha256(report.model_dump_json(exclude={"review"}).encode()).hexdigest()


def save_report(report: AssessmentReport, root: Path | None = None) -> Path:
    d = (Path(root) if root else runs_dir()) / report.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (d / "report.md").write_text(to_markdown(report), encoding="utf-8")
    return d


def load_report(run_id: str, root: Path | None = None) -> AssessmentReport:
    p = (Path(root) if root else runs_dir()) / run_id / "report.json"
    return AssessmentReport.model_validate_json(p.read_text(encoding="utf-8"))


def list_runs(root: Path | None = None) -> list[str]:
    r = Path(root) if root else runs_dir()
    if not r.exists():
        return []
    return sorted((p.name for p in r.iterdir() if (p / "report.json").exists()), reverse=True)


def record_decision(
    report: AssessmentReport, decision: str, reviewer: str, comment: str = "", audit: AuditLogger | None = None
) -> AssessmentReport:
    if report.status not in (ReportStatus.PENDING_REVIEW, ReportStatus.ESCALATED):
        raise ValueError(f"Report is {report.status.value}; only PENDING_REVIEW/ESCALATED can be decided.")
    if not reviewer.strip():
        raise ValueError("Reviewer name is required.")
    sha = report_sha256(report)
    report.review = ReviewDecision(decision=decision, reviewer=reviewer.strip(), comment=comment, reviewed_report_sha256=sha)
    report.status = ReportStatus(decision)
    (audit or AuditLogger()).log(
        report.run_id, "human_decision",
        {"decision": decision, "reviewer": reviewer, "comment": comment, "reviewed_report_sha256": sha},
    )
    save_report(report)
    return report


def to_markdown(r: AssessmentReport) -> str:
    L: list[str] = [f"# Vendor risk screening: {r.vendor.name}", ""]
    L.append(f"- **Run:** `{r.run_id}` · **Status:** {r.status.value} · **Orchestrator:** {r.orchestrator} · **Model:** {r.model}")
    L.append(f"- **Generated:** {r.created_at.isoformat()}")
    if r.entity:
        L.append(f"- **Entity:** {r.entity.resolved_name} (LEI {r.entity.lei or 'n/a'}, {r.entity.status.value}, "
                 f"confidence {r.entity.confidence})")
    L.append("")
    a = r.assessment
    if a:
        L += ["## Result (deterministic)", "",
              f"| Score | Band | Confidence | Completeness | Hard stop |", "|---|---|---|---|---|",
              f"| {a.score} | {a.band.value} | {a.confidence.value} | {a.completeness:.0%} | {'YES' if a.hard_stop else 'no'} |", ""]
        if a.review_flags:
            L += ["**Review flags:** " + ", ".join(f"`{f}`" for f in a.review_flags), ""]
        L += ["**Rule-based rationale**", ""] + [f"- {x}" for x in a.rationale] + [""]
        L += ["**Sources checked:** " + ", ".join(f"{k}={v.value}" for k, v in a.source_status.items()), ""]
    if r.sanctions:
        L += ["## Sanctions screening", "",
              f"Lists: {', '.join(r.sanctions.lists_checked) or 'NONE LOADED'} · entries screened: {r.sanctions.entries_screened}", ""]
        if r.sanctions.hits:
            L += ["| List | Entry | Listed name | Matched on | Score | Strength |", "|---|---|---|---|---|---|"]
            L += [f"| {h.list_name} | {h.entry_id} | {h.listed_name} | {h.matched_on} | {h.score} | {h.strength.value} |"
                  for h in r.sanctions.hits]
        else:
            L.append("No matches at or above the potential-match threshold.")
        L.append("")
    if r.findings:
        pts = {c.finding_id: c.points for c in (a.contributions if a else [])}
        L += ["## Verified findings", "", "| ID | Dimension | Evidence type | Severity | Date | Points | Title |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {f.finding_id} | {f.dimension.value} | {f.evidence_type.value} | {f.severity.value} | "
              f"{f.event_date or '—'} | {pts.get(f.finding_id, 0)} | {f.title} |" for f in r.findings]
        L.append("")
        for f in r.findings:
            L += [f"### {f.finding_id}: {f.title}", "", f.summary, "",
                  f"> {f.supporting_passage}", "",
                  f"Source: {f.source_url} · snapshot `{f.snapshot_id}`", ""]
            L += [f"**Warning:** {w}" for w in f.warnings] + ([""] if f.warnings else [])
            if f.merged_finding_ids:
                L += [f"Same event as {', '.join(f.merged_finding_ids)}: counted once.", ""]
    if r.rejected_findings:
        L += ["## Rejected agent claims (not scored)", ""]
        L += [f"- {f.title} — {'; '.join(f.rejection_reasons)}" for f in r.rejected_findings]
        L.append("")
    if r.explanation:
        e = r.explanation
        L += ["## Analyst explanation (LLM, constrained to the result above)", "", e.executive_summary, ""]
        if e.key_drivers:
            L += ["**Key drivers**", ""] + [f"- {x}" for x in e.key_drivers] + [""]
        if e.evidence_gaps:
            L += ["**Evidence gaps**", ""] + [f"- {x}" for x in e.evidence_gaps] + [""]
        if e.recommended_actions:
            L += ["**Recommended actions**", ""] + [f"- {x}" for x in e.recommended_actions] + [""]
    if r.explanation_check and not r.explanation_check.consistent:
        L += ["> **Explanation consistency check FAILED:** " + "; ".join(r.explanation_check.issues), ""]
    if r.errors:
        L += ["## Errors", ""] + [f"- {x}" for x in r.errors] + [""]
    if r.review:
        L += ["## Human decision", "",
              f"{r.review.decision} by {r.review.reviewer} at {r.review.decided_at.isoformat()}", "",
              r.review.comment or "", "", f"Reviewed report sha256: `{r.review.reviewed_report_sha256}`", ""]
    L += ["---", "*Decision-support output. Allegations are not findings of fact. A human makes the onboarding decision.*"]
    return "\n".join(L)
