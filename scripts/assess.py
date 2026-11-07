"""CLI: run one assessment.

python -m scripts.assess "Acme Infotech Ltd" [--lei LEI] [--confirmed] [--orchestrator crewai|single] [--no-validate]
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from core.schemas import VendorInput


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--country", default="IN")
    ap.add_argument("--website")
    ap.add_argument("--lei")
    ap.add_argument("--confirmed", action="store_true", help="identity already confirmed by a human")
    ap.add_argument("--orchestrator", choices=["crewai", "single"], default="crewai")
    ap.add_argument("--no-validate", action="store_true", help="ablation: score unvalidated agent claims")
    args = ap.parse_args()

    from orchestration.crewai_impl.orchestrator import CrewAIOrchestrator, SingleAgentOrchestrator

    cls = CrewAIOrchestrator if args.orchestrator == "crewai" else SingleAgentOrchestrator
    orch = cls(validate=not args.no_validate)
    vendor = VendorInput(name=args.name, country=args.country, website=args.website, lei=args.lei,
                         confirmed=args.confirmed)
    r = orch.assess_vendor(vendor)
    print(f"\nrun_id={r.run_id} status={r.status.value}")
    if r.assessment:
        a = r.assessment
        print(f"score={a.score} band={a.band.value} confidence={a.confidence.value} completeness={a.completeness:.0%}")
        print(f"flags={a.review_flags}")
        print(f"verified={len(r.findings)} rejected={len(r.rejected_findings)} tokens={r.metrics.total_tokens} "
              f"time={r.metrics.duration_s}s")
    elif r.entity:
        print("Entity needs confirmation. Candidates:")
        for c in r.entity.candidates:
            print(f"  {c.lei}  {c.legal_name}  ({c.city}, {c.country})  score={c.match_score}")
    print(f"report: data/runs/{r.run_id}/report.md")


if __name__ == "__main__":
    main()
