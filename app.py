"""Streamlit UI: run an assessment, confirm identity, review evidence, record the human decision.

streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.reporting import list_runs, load_report, record_decision, to_markdown
from core.schemas import AssessmentReport, ReportStatus, VendorInput

load_dotenv()
st.set_page_config(page_title="Vendor Risk Screening", layout="wide")

@st.cache_resource(show_spinner=False)
def get_orchestrator(kind: str, validate: bool):
    from orchestration.crewai_impl.orchestrator import CrewAIOrchestrator, SingleAgentOrchestrator

    return (CrewAIOrchestrator if kind == "Multi-agent (CrewAI Flow)" else SingleAgentOrchestrator)(validate=validate)


def run(vendor: VendorInput, kind: str, validate: bool) -> None:
    with st.spinner("Resolving entity, screening sanctions, running research agents…"):
        try:
            st.session_state.report = get_orchestrator(kind, validate).assess_vendor(vendor)
        except Exception as exc:
            st.error(f"Run failed: {exc}")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    kind = st.radio("Orchestrator", ["Multi-agent (CrewAI Flow)", "Single-agent baseline"])
    validate = st.toggle("Evidence validation", value=True, help="Off = ablation: agent claims are scored unchecked")
    st.divider()
    runs = list_runs()
    pick = st.selectbox("Open a previous run", ["—"] + runs)
    if pick != "—" and st.button("Load run"):
        st.session_state.report = load_report(pick)

st.title("Vendor Risk Screening")
st.caption("Decision support for vendor onboarding. Agents investigate, code verifies and scores, an LLM explains, "
           "a human decides.")

# ---------------------------------------------------------------- input
with st.form("vendor"):
    c1, c2, c3 = st.columns([3, 1, 2])
    name = c1.text_input("Vendor legal name", placeholder="e.g. Acme Infotech Limited")
    country = c2.text_input("Country", value="IN", max_chars=2)
    lei = c3.text_input("LEI (optional)")
    website = st.text_input("Website (optional)")
    submitted = st.form_submit_button("Run assessment", type="primary")
if submitted and name.strip():
    run(VendorInput(name=name.strip(), country=country.strip() or "IN", lei=lei.strip() or None,
                    website=website.strip() or None), kind, validate)

r: AssessmentReport | None = st.session_state.get("report")
if r is None:
    st.stop()

st.divider()
st.subheader(f"{r.vendor.name}  ·  run `{r.run_id}`")
st.caption(f"Status: **{r.status.value}** · orchestrator: {r.orchestrator} · model: {r.model} · "
           f"{r.metrics.duration_s}s · {r.metrics.total_tokens} tokens")

# ---------------------------------------------------------------- identity confirmation
if r.status == ReportStatus.NEEDS_ENTITY_CONFIRMATION:
    st.warning("Identity is ambiguous. Confirm the correct legal entity before any research is run.")
    for n in r.entity.notes:
        st.caption(n)
    options = {f"{c.legal_name} — LEI {c.lei} ({c.city}, {c.country}) · match {c.match_score}": c for c in r.entity.candidates}
    choice = st.radio("Registry candidates", list(options) + ["None of these: proceed with the name as entered"])
    if st.button("Confirm and continue", type="primary"):
        if choice in options:
            v = r.vendor.model_copy(update={"lei": options[choice].lei})
        else:
            v = r.vendor.model_copy(update={"confirmed": True})
        run(v, kind, validate)
        st.rerun()
    st.stop()

a = r.assessment
if a is None:
    st.error("No assessment produced. Errors: " + "; ".join(r.errors))
    st.stop()

# ---------------------------------------------------------------- headline
if a.hard_stop:
    st.error("HARD STOP: strong sanctions match. Research was not run. Compliance must clear this before anything else.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Risk score (rules)", f"{a.score:.1f} / 100")
m2.metric("Risk band", a.band.value)
m3.metric("Evidence confidence", a.confidence.value)
m4.metric("Source completeness", f"{a.completeness:.0%}")
if a.review_flags:
    st.warning("Review flags: " + ", ".join(f"`{f}`" for f in a.review_flags))

tab_find, tab_score, tab_sanc, tab_expl, tab_rej, tab_audit = st.tabs(
    ["Evidence", "Score breakdown", "Sanctions", "Explanation", f"Rejected claims ({len(r.rejected_findings)})", "Audit"])

with tab_find:
    if not r.findings:
        st.info("No verified findings." + (" Evidence is INSUFFICIENT: this is not a clean result."
                                           if a.confidence.value == "INSUFFICIENT" else ""))
    for f in r.findings:
        with st.expander(f"[{f.finding_id}] {f.title}  ·  {f.evidence_type.value}  ·  {f.severity.value}"):
            st.write(f.summary)
            st.markdown(f"> {f.supporting_passage}")
            st.caption(f"{f.dimension.value} · date {f.event_date or 'unknown'} · source {f.source_url} · "
                       f"snapshot `{f.snapshot_id}` · passage match {f.passage_match_score}")
            for w in f.warnings:
                st.warning(w)
            if f.merged_finding_ids:
                st.caption(f"Same event as {', '.join(f.merged_finding_ids)}: counted once")

with tab_score:
    st.write("Every point comes from a rule in `config/rules.yaml` (version " + a.rules_version + ").")
    if a.contributions:
        st.dataframe(pd.DataFrame([c.model_dump() for c in a.contributions]), width="stretch", hide_index=True)
    st.json(a.dimension_scores)
    for x in a.rationale:
        st.write("• " + x)
    st.write("Sources: " + ", ".join(f"{k} = {v.value}" for k, v in a.source_status.items()))

with tab_sanc:
    s = r.sanctions
    if s:
        st.write(f"Lists: {', '.join(s.lists_checked) or 'none loaded'} · {s.entries_screened} names screened")
        if s.hits:
            st.dataframe(pd.DataFrame([h.model_dump() for h in s.hits]), width="stretch", hide_index=True)
        else:
            st.success("No matches at or above the potential-match threshold.")

with tab_expl:
    if r.explanation:
        if r.explanation_check and not r.explanation_check.consistent:
            st.error("Consistency check failed: " + "; ".join(r.explanation_check.issues))
        st.write(r.explanation.executive_summary)
        for title, items in (("Key drivers", r.explanation.key_drivers), ("Evidence gaps", r.explanation.evidence_gaps),
                             ("Recommended actions", r.explanation.recommended_actions)):
            if items:
                st.markdown(f"**{title}**")
                for i in items:
                    st.write("• " + i)
    else:
        st.info("No explanation generated.")

with tab_rej:
    st.caption("Claims the agents made that failed deterministic validation. They were not scored.")
    for f in r.rejected_findings:
        st.write(f"**{f.title}** ({f.agent}): " + "; ".join(f.rejection_reasons))

with tab_audit:
    from core.audit import AuditLogger

    log = AuditLogger()
    ok, bad = log.verify_chain()
    st.write("Audit chain intact ✅" if ok else f"Audit chain BROKEN at line {bad} ❌")
    st.dataframe(pd.DataFrame([{k: e[k] for k in ("ts", "event")} for e in log.events(r.run_id)]),
                 width="stretch", hide_index=True)

c1, c2 = st.columns(2)
c1.download_button("Download report (Markdown)", to_markdown(r), file_name=f"{r.run_id}.md")
c2.download_button("Download report (JSON)", r.model_dump_json(indent=2), file_name=f"{r.run_id}.json",
                   mime="application/json")

# ---------------------------------------------------------------- human decision
st.divider()
st.subheader("Reviewer decision")
if r.review:
    st.success(f"{r.review.decision} by {r.review.reviewer} at {r.review.decided_at:%Y-%m-%d %H:%M UTC}. "
               f"Report hash `{r.review.reviewed_report_sha256[:16]}…`")
    if r.review.comment:
        st.write(r.review.comment)
elif r.status == ReportStatus.PENDING_REVIEW:
    with st.form("decision"):
        reviewer = st.text_input("Reviewer name")
        decision = st.radio("Decision", ["APPROVED", "REJECTED", "ESCALATED"], horizontal=True)
        comment = st.text_area("Comment (required for APPROVED with flags)")
        if st.form_submit_button("Record decision", type="primary"):
            if decision == "APPROVED" and a.review_flags and not comment.strip():
                st.error("Approving a report with review flags requires a comment.")
            else:
                try:
                    st.session_state.report = record_decision(r, decision, reviewer, comment)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
