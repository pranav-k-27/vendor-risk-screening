"""CrewAI implementations of the Orchestrator contract.

CrewAIOrchestrator      multi-agent Flow (2 parallel research crews + explainer)
SingleAgentOrchestrator baseline: one agent with all tools, SAME deterministic pre/post steps
Both accept validate=False for the "trust the agents" ablation.
"""
from __future__ import annotations

import asyncio
import os
import time

from core import pipeline
from core.pipeline import ResearchOutcome, Services
from core.schemas import AssessmentReport, Explanation, FindingsBatch, SourceStatus, VendorInput
from orchestration.crewai_impl import agents
from orchestration.crewai_impl.flow import Runners, VendorAssessmentFlow
from orchestration.crewai_impl.tools import ToolCallCounter


def _crew_runner(build, agent_label: str | None = None):
    """build(inputs, tool_calls) -> Crew. Usage returned includes per-tool call counts ('tool_calls')
    and, if given, an 'agent' label used for per-agent metrics."""
    async def run(inputs: dict) -> tuple[FindingsBatch, dict]:
        calls = ToolCallCounter()
        out = await build(inputs, calls).kickoff_async(inputs=inputs)  # guardrail needs the run inputs
        batch = out.pydantic if isinstance(out.pydantic, FindingsBatch) else FindingsBatch()
        usage = pipeline.usage_from(out) | {"tool_calls": dict(calls.counts)}
        if agent_label:
            usage["agent"] = agent_label
        return batch, usage
    return run


def _explain_runner(llm):
    def run(context: str) -> tuple[Explanation, dict]:
        out = agents.explainer_crew(llm).kickoff(inputs={"context": context})
        if not isinstance(out.pydantic, Explanation):
            raise ValueError("explainer returned no structured output")
        return out.pydantic, pipeline.usage_from(out)
    return run


def default_runners(explain: bool = True, guard_validate: bool = True) -> Runners:
    """guard_validate=False: task guardrails run structural checks only (no validator feedback loop)."""
    llm = agents.build_llm()
    return Runners(
        regulatory=_crew_runner(lambda inputs, calls: agents.regulatory_crew(llm, inputs, guard_validate, calls)),
        media=_crew_runner(lambda inputs, calls: agents.media_crew(llm, inputs, guard_validate, calls)),
        explain=_explain_runner(llm) if explain else None,
    )


class CrewAIOrchestrator:
    name = "crewai_flow"

    def __init__(self, svc: Services | None = None, runners: Runners | None = None, validate: bool = True,
                 explain: bool = True, guard_validate: bool = True):
        self.svc = svc or Services.default()
        self.runners = runners or default_runners(explain, guard_validate)
        self.validate = validate
        self.model = os.getenv("MODEL", "gpt-4o-mini")

    def assess_vendor(self, vendor: VendorInput) -> AssessmentReport:
        flow = VendorAssessmentFlow(vendor=vendor, svc=self.svc, runners=self.runners, model=self.model,
                                    validate=self.validate)
        report = flow.kickoff()
        if not isinstance(report, AssessmentReport):
            raise RuntimeError(f"Flow ended without a report (got {type(report).__name__})")
        return report


class SingleAgentOrchestrator:
    """Same gates and scoring; the research step is one generalist agent instead of two specialists."""

    name = "single_agent"

    def __init__(self, svc: Services | None = None, validate: bool = True, explain: bool = True,
                 research=None, explain_fn=None):
        self.svc = svc or Services.default()
        self.validate = validate
        llm = agents.build_llm() if (research is None or (explain and explain_fn is None)) else None
        self.research = research or _crew_runner(
            lambda inputs, calls: agents.single_agent_crew(llm, inputs, tool_calls=calls), agent_label="single_agent")
        self.explain = explain_fn or (_explain_runner(llm) if explain else None)
        self.model = os.getenv("MODEL", "gpt-4o-mini")

    def assess_vendor(self, vendor: VendorInput) -> AssessmentReport:
        run_id, started, svc = pipeline.new_run_id(), time.time(), self.svc
        svc.audit.log(run_id, "run_started", {"vendor": vendor.model_dump(), "orchestrator": self.name,
                                              "model": self.model, "validation": self.validate})
        entity = pipeline.resolve_entity(vendor, svc, run_id)
        if pipeline.entity_needs_human(entity):
            return pipeline.halt_for_entity(run_id=run_id, orchestrator=self.name, model=self.model,
                                            vendor=vendor, entity=entity, svc=svc, started=started)
        sanctions, s_status = pipeline.screen_sanctions(entity, svc, run_id)
        common = dict(run_id=run_id, orchestrator=self.name, model=self.model, vendor=vendor, entity=entity,
                      sanctions=sanctions, sanctions_status=s_status, svc=svc, explain=self.explain,
                      started=started, validate=self.validate)
        if sanctions.has_strong_hit:
            return pipeline.finalize(outcomes=[ResearchOutcome("regulatory", SourceStatus.SKIPPED),
                                               ResearchOutcome("adverse_media", SourceStatus.SKIPPED)], **common)

        inputs = {"run_id": run_id, "vendor_name": vendor.name, "resolved_name": entity.resolved_name,
                  "aliases": ", ".join(entity.aliases) or "none", "country": vendor.country}
        try:
            batch, usage = asyncio.run(self.research(inputs))
            outcomes = _split_by_dimension(batch, usage, pipeline.regulatory_precheck(vendor, entity))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:400]
            outcomes = [ResearchOutcome("regulatory", SourceStatus.FAILED, error=err),
                        ResearchOutcome("adverse_media", SourceStatus.FAILED, error=err)]
        return pipeline.finalize(outcomes=outcomes, **common)


def _split_by_dimension(batch: FindingsBatch, usage: dict, reg_pre: SourceStatus | None) -> list[ResearchOutcome]:
    reg = [f for f in batch.findings if f.dimension.value == "regulatory"]
    med = [f for f in batch.findings if f.dimension.value == "adverse_media"]
    reg_status = reg_pre if reg_pre is not None and not reg else (SourceStatus.OK if reg else SourceStatus.NO_DATA)
    return [
        ResearchOutcome("regulatory", reg_status, FindingsBatch(findings=reg), usage=usage),
        ResearchOutcome("adverse_media", SourceStatus.OK if med else SourceStatus.NO_DATA,
                        FindingsBatch(findings=med, queries_run=batch.queries_run)),
    ]
