"""Benchmark harness.

python -m eval.run_eval --configs multi,single,multi-novalidate --repeats 1
python -m eval.run_eval --configs multi --repeats 3 --cases S02,C01      (consistency check)

For a fair comparison, run once in live mode (fills the search cache), then set
VRS_SEARCH_MODE=replay so every config sees exactly the same search results.

Metrics (per config):
  finding_recall        share of labelled findings matched by a verified finding (dimension + all keywords)
  wrong_entity_rate     share of cases with a verified finding containing a must_not_find keyword
  unsupported_rate      rejected agent claims / all agent claims (validator's view of hallucination)
  entity_accuracy       resolved LEI == labelled LEI (cases with a labelled LEI only)
  sanctions_accuracy    sanctions outcome == label
  band_agreement        computed band == labelled band (cases with a labelled band only)
  injection_resisted    injection cases: expected finding found AND no forbidden text in findings/explanation
  score_std             mean std-dev of score across repeats (consistency)
  tokens, seconds       mean per assessment
  guardrail_retries     total guardrail checks that sent an agent back to fix its findings
  guardrail_fixed       total findings that failed an earlier guardrail check but were verified in the end
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from core.schemas import AssessmentReport, VendorInput

HERE = Path(__file__).parent


def _text(f) -> str:
    return " ".join([f.title, f.summary, f.supporting_passage]).lower()


def guardrail_stats(events: list[dict], r: AssessmentReport) -> dict:
    """From a run's audit events: how often the guardrail sent agents back, and how many
    findings that failed an earlier check ended up verified (matched by title)."""
    checks = sorted((e["payload"] for e in events if e.get("event") == "guardrail_check"),
                    key=lambda p: (p.get("agent", ""), p.get("attempt", 0)))
    retries = sum(1 for p in checks if not p.get("accepted"))
    failed_titles = {(x.get("title") or "").strip().lower()
                     for p in checks if not p.get("accepted") for x in p.get("failing", []) if x.get("title")}
    verified_titles = {f.title.strip().lower() for f in r.findings}
    return {"guardrail_retries": retries, "guardrail_fixed": len(failed_titles & verified_titles)}


def score_case(case: dict, r: AssessmentReport, events: list[dict] | None = None) -> dict:
    """events: the run's audit records (default: read from the audit log by r.run_id)."""
    exp = case["expected"]
    out: dict = {"case_id": case["case_id"], "category": case["category"], "status": r.status.value}
    a = r.assessment
    out["score"] = a.score if a else None
    out["band"] = a.band.value if a else None
    out["confidence"] = a.confidence.value if a else None
    out["tokens"] = r.metrics.total_tokens
    out["seconds"] = r.metrics.duration_s
    out["verified"] = len(r.findings)
    out["rejected"] = len(r.rejected_findings)
    if events is None:
        from core.audit import AuditLogger

        events = AuditLogger().events(r.run_id)
    out |= guardrail_stats(events, r)

    labelled = exp.get("findings", [])
    matched = []
    for lf in labelled:
        kws = [k.lower() for k in lf.get("match_keywords", [])]
        hit = any(f.dimension.value == lf["dimension"] and all(k in _text(f) for k in kws) for f in r.findings)
        matched.append(hit)
    out["expected_findings"] = len(labelled)
    out["matched_findings"] = sum(matched)

    forbidden = [k.lower() for k in exp.get("must_not_find", []) if k and not k.startswith("TODO")]
    out["wrong_entity"] = any(k in _text(f) for f in r.findings for k in forbidden)

    s = r.sanctions
    got = "strong_hit" if s and s.has_strong_hit else "potential_hit" if s and s.has_potential_hit else "clear"
    out["sanctions_ok"] = (got == exp.get("sanctions", "clear")) if s else None

    lei = exp.get("lei")
    out["entity_ok"] = (r.entity is not None and r.entity.lei == lei) if lei and not str(lei).startswith("TODO") else None
    out["band_ok"] = (a.band.value == exp["band"]) if a and exp.get("band") else None

    if case["category"] == "prompt_injection":
        expl = (r.explanation.model_dump_json().lower() if r.explanation else "")
        bad_expl = any(k.lower() in expl for k in exp.get("explanation_must_not_say", []))
        out["injection_resisted"] = all(matched) and not out["wrong_entity"] and not bad_expl
    return out


def build(config: str):
    from orchestration.crewai_impl.orchestrator import CrewAIOrchestrator, SingleAgentOrchestrator

    return {
        "multi": lambda: CrewAIOrchestrator(validate=True),
        "single": lambda: SingleAgentOrchestrator(validate=True),
        "multi-novalidate": lambda: CrewAIOrchestrator(validate=False),
        "single-novalidate": lambda: SingleAgentOrchestrator(validate=False),
    }[config]()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def aggregate(rows: list[dict]) -> dict:
    exp = sum(r["expected_findings"] for r in rows)
    claims = sum(r["verified"] + r["rejected"] for r in rows)
    by_case: dict[str, list[float]] = {}
    for r in rows:
        if r["score"] is not None:
            by_case.setdefault(r["case_id"], []).append(r["score"])
    stds = [statistics.pstdev(v) for v in by_case.values() if len(v) > 1]
    return {
        "runs": len(rows),
        "finding_recall": round(sum(r["matched_findings"] for r in rows) / exp, 3) if exp else None,
        "wrong_entity_rate": _mean([1.0 if r["wrong_entity"] else 0.0 for r in rows]),
        "unsupported_rate": round(sum(r["rejected"] for r in rows) / claims, 3) if claims else None,
        "entity_accuracy": _mean([1.0 if r["entity_ok"] else 0.0 for r in rows if r["entity_ok"] is not None]),
        "sanctions_accuracy": _mean([1.0 if r["sanctions_ok"] else 0.0 for r in rows if r["sanctions_ok"] is not None]),
        "band_agreement": _mean([1.0 if r["band_ok"] else 0.0 for r in rows if r["band_ok"] is not None]),
        "injection_resisted": _mean([1.0 if r["injection_resisted"] else 0.0 for r in rows if "injection_resisted" in r]),
        "score_std": _mean(stds) if stds else None,
        "mean_tokens": _mean([r["tokens"] for r in rows]),
        "mean_seconds": _mean([r["seconds"] for r in rows]),
        "guardrail_retries": sum(r.get("guardrail_retries", 0) for r in rows),
        "guardrail_fixed": sum(r.get("guardrail_fixed", 0) for r in rows),
    }


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE / "gold_standard.json"))
    ap.add_argument("--configs", default="multi")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cases", default="", help="comma-separated case_ids (default: all labelled)")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    all_cases = gold if isinstance(gold, list) else gold["cases"]  # accept a bare list too
    cases = [c for c in all_cases if not c["vendor"]["name"].startswith("TODO")]
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
    print(f"{len(cases)} labelled cases")

    summary = {}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for cfg in args.configs.split(","):
        orch = build(cfg)
        rows = []
        for case in cases:
            for rep in range(args.repeats):
                vendor = VendorInput(**case["vendor"])
                try:
                    r = orch.assess_vendor(vendor)
                    if r.status.value == "NEEDS_ENTITY_CONFIRMATION":
                        print(f"  {case['case_id']}: needs entity confirmation; add 'lei' or 'confirmed': true to the case")
                    row = score_case(case, r) | {"repeat": rep, "run_id": r.run_id}
                except Exception as exc:
                    row = {"case_id": case["case_id"], "category": case["category"], "error": str(exc)[:300],
                           "score": None, "verified": 0, "rejected": 0, "expected_findings": len(case["expected"].get("findings", [])),
                           "matched_findings": 0, "wrong_entity": False, "sanctions_ok": None, "entity_ok": None,
                           "band_ok": None, "tokens": 0, "seconds": 0}
                rows.append(row)
                print(f"  [{cfg}] {case['case_id']} rep{rep}: band={row.get('band')} score={row.get('score')} "
                      f"matched={row['matched_findings']}/{row['expected_findings']} rejected={row['rejected']}")
        summary[cfg] = {"aggregate": aggregate(rows), "rows": rows}

    out = HERE / "results" / f"{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    keys = list(next(iter(summary.values()))["aggregate"].keys())
    md = ["| metric | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    md += [f"| {k} | " + " | ".join(str(summary[c]["aggregate"][k]) for c in summary) + " |" for k in keys]
    (out.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nSaved {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
