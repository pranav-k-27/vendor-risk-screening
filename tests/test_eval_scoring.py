from eval.run_eval import aggregate, score_case
from tests.test_flow import CrewAIOrchestrator, Runners, empty_runner, media_runner, services
from core.schemas import VendorInput
from core.sources import slugify


def test_eval_scoring_on_stubbed_run(isolated_data_dir):
    svc = services()
    (isolated_data_dir / "documents" / slugify("Acme Infotech Ltd")).mkdir()
    r = CrewAIOrchestrator(svc=svc, runners=Runners(empty_runner, media_runner(svc.snapshots), None)).assess_vendor(
        VendorInput(name="Acme Infotech Ltd"))
    case = {"case_id": "T1", "category": "media_allegation", "vendor": {"name": "Acme Infotech Ltd"},
            "expected": {"lei": "LEI1", "sanctions": "clear", "band": "LOW",
                         "findings": [{"dimension": "adverse_media", "match_keywords": ["unpaid wages"]},
                                      {"dimension": "regulatory", "match_keywords": ["penalty"]}],
                         "must_not_find": ["convicted"]}}
    row = score_case(case, r)
    assert row["matched_findings"] == 1 and row["expected_findings"] == 2
    assert row["entity_ok"] and row["sanctions_ok"] and row["band_ok"] and not row["wrong_entity"]
    agg = aggregate([row])
    assert agg["finding_recall"] == 0.5 and agg["unsupported_rate"] == 0.0
