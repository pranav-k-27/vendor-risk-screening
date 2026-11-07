"""CrewAI Flow: deterministic gates around two parallel research crews.

  resolve_entity ──router──► needs_entity_confirmation ─► halt (status NEEDS_ENTITY_CONFIRMATION)
        │
        └─► entity_ok ─► screen_sanctions ──router──► sanctions_hit ─► finalize_hard_stop ─┐
                                   │                                                     ├─► report
                                   └─► sanctions_clear ─┬─► research_regulatory ─┐       │
                                                        └─► research_media ──────┴─and_─► finalize
Human review is NOT inside the Flow: the Flow ends at PENDING_REVIEW and the app
records the decision (see core.reporting.record_decision).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from crewai.flow.flow import Flow, and_, listen, or_, router, start

from core import pipeline
from core.pipeline import ResearchOutcome, Services
from core.schemas import AssessmentReport, Explanation, FindingsBatch, SourceStatus, VendorInput

ResearchFn = Callable[[dict], Awaitable[tuple[FindingsBatch, dict]]]
ExplainFn = Callable[[str], tuple[Explanation, dict]]


@dataclass
class Runners:
    regulatory: ResearchFn
    media: ResearchFn
    explain: ExplainFn | None


class VendorAssessmentFlow(Flow):
    def __init__(self, vendor: VendorInput, svc: Services, runners: Runners, model: str | None,
                 validate: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.vendor = vendor
        self.svc = svc
        self.runners = runners
        self.model = model
        self.validate = validate
        self.run_id = pipeline.new_run_id()
        self.started = time.time()
        self.entity = None
        self.sanctions = None
        self.sanctions_status = SourceStatus.FAILED
        self.outcomes: dict[str, ResearchOutcome] = {}
        self.svc.audit.log(self.run_id, "run_started", {"vendor": vendor.model_dump(), "orchestrator": "crewai_flow",
                                                         "model": model, "validation": validate})

    # ---- 1. identity -------------------------------------------------------
    @start()
    def resolve_entity(self):
        self.entity = pipeline.resolve_entity(self.vendor, self.svc, self.run_id)

    @router(resolve_entity)
    def entity_gate(self):
        return "needs_entity_confirmation" if pipeline.entity_needs_human(self.entity) else "entity_ok"

    @listen("needs_entity_confirmation")
    def halt_for_entity(self) -> AssessmentReport:
        return pipeline.halt_for_entity(run_id=self.run_id, orchestrator="crewai_flow", model=self.model,
                                        vendor=self.vendor, entity=self.entity, svc=self.svc, started=self.started)

    # ---- 2. sanctions gate ---------------------------------------------------
    @listen("entity_ok")
    def screen_sanctions(self):
        self.sanctions, self.sanctions_status = pipeline.screen_sanctions(self.entity, self.svc, self.run_id)

    @router(screen_sanctions)
    def sanctions_gate(self):
        return "sanctions_hit" if self.sanctions.has_strong_hit else "sanctions_clear"

    # ---- 3. research, in parallel -------------------------------------------
    def _inputs(self) -> dict:
        return {
            "run_id": self.run_id,  # lets the task guardrail write to the audit log
            "vendor_name": self.vendor.name,
            "resolved_name": self.entity.resolved_name,
            "aliases": ", ".join(self.entity.aliases) or "none",
            "country": self.vendor.country,
        }

    async def _run(self, agent: str, fn: ResearchFn) -> ResearchOutcome:
        self.svc.audit.log(self.run_id, f"{agent}_started", {})
        try:
            batch, usage = await fn(self._inputs())
            status = SourceStatus.OK if batch.findings else SourceStatus.NO_DATA
            out = ResearchOutcome(agent=agent, status=status, batch=batch, usage=usage)
        except Exception as exc:  # one failed source must not kill the run
            out = ResearchOutcome(agent=agent, status=SourceStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:400])
        self.svc.audit.log(self.run_id, f"{agent}_finished", {
            "status": out.status.value, "error": out.error,
            "findings": len(out.batch.findings) if out.batch else 0,
            "queries": out.batch.queries_run if out.batch else [],
        })
        return out

    @listen("sanctions_clear")
    async def research_regulatory(self):
        pre = pipeline.regulatory_precheck(self.vendor, self.entity)
        if pre is not None:
            err = "no curated regulatory documents folder for this vendor" if pre == SourceStatus.FAILED else None
            self.outcomes["regulatory"] = ResearchOutcome(agent="regulatory", status=pre, error=err)
        else:
            self.outcomes["regulatory"] = await self._run("regulatory", self.runners.regulatory)

    @listen("sanctions_clear")
    async def research_media(self):
        self.outcomes["adverse_media"] = await self._run("adverse_media", self.runners.media)

    # ---- 4. validate, score, explain ----------------------------------------
    @listen(and_(research_regulatory, research_media))
    def finalize(self) -> AssessmentReport:
        return pipeline.finalize(
            run_id=self.run_id, orchestrator="crewai_flow", model=self.model, vendor=self.vendor,
            entity=self.entity, sanctions=self.sanctions, sanctions_status=self.sanctions_status,
            outcomes=list(self.outcomes.values()), svc=self.svc, explain=self.runners.explain,
            started=self.started, validate=self.validate,
        )

    @listen("sanctions_hit")
    def finalize_hard_stop(self) -> AssessmentReport:
        skipped = [ResearchOutcome(agent=a, status=SourceStatus.SKIPPED) for a in ("regulatory", "adverse_media")]
        return pipeline.finalize(
            run_id=self.run_id, orchestrator="crewai_flow", model=self.model, vendor=self.vendor,
            entity=self.entity, sanctions=self.sanctions, sanctions_status=self.sanctions_status,
            outcomes=skipped, svc=self.svc, explain=self.runners.explain, started=self.started,
        )

    @listen(or_(finalize, finalize_hard_stop, halt_for_entity))
    def done(self, report: AssessmentReport) -> AssessmentReport:
        return report
