# Labelling guide: building the gold-standard set (about 2–3 hours)

The evaluation is only as good as these labels. Do this **before** tuning prompts, and don't change a label after seeing a model's output unless the label was wrong.

## Pick 9 listed Indian companies

| Cases | Category | What to look for |
|---|---|---|
| C01–C03 | `enforcement` | A listed company with at least one **final SEBI adjudication order or penalty** in the last ~5 years (search SEBI's enforcement orders by company name). |
| C04–C05 | `media_allegation` | Credible news of an **allegation, probe or lawsuit** with no final regulatory order. |
| C06–C07 | `similar_name` | A company whose name resembles **another** company that has adverse news. The system must *not* attribute the other company's issues. |
| C08–C09 | `clean_control` | Large, well-governed companies where you found nothing material. |

S01 (synthetic sanctions hit) and S02 (synthetic prompt injection) are already provided.

## For each company

1. **Identity.** Look up its LEI on gleif.org (search by name, filter India). Put it in `expected.lei`, and also in `vendor.lei` if you want to skip the identity step during evaluation. No LEI? Leave `lei` null and set `"confirmed": true` in `vendor`.
2. **Regulatory documents.** Create `data/documents/<slug>/`, where slug is the vendor name lower-cased with non-alphanumerics replaced by `-` (e.g. `Acme Infotech Ltd` → `acme-infotech-ltd`).
   - Download the relevant SEBI order PDFs into it.
   - Add `metadata.json` mapping each file to `source_url` (the sebi.gov.in page), `title`, `date`, `doc_type`.
   - **If you searched and found nothing, still create the empty folder.** An empty folder means "checked, nothing found". A missing folder means "not checked", and the system will report INSUFFICIENT evidence.
3. **Expected findings.** For each real issue, add an entry:
   - `dimension`: `regulatory` or `adverse_media`
   - `evidence_type`: what it really is (final order, settlement, allegation…)
   - `match_keywords`: 1–2 short, distinctive strings that any correct finding must contain (a penalty amount like `25,00,000`, an order number, a specific word). Avoid generic words like "SEBI" or "penalty" on their own.
4. **Must-not-find.** For `similar_name` cases, list a keyword that only appears in the *other* company's news.
5. **Expected band (optional).** Only if you are confident. Band agreement is the weakest metric because the weights are design assumptions.

## Rules that keep the evaluation honest

- Prefer companies whose issues are a **matter of public record** (regulator orders). Avoid building the demo around unproven allegations against named companies.
- Freeze the labels before running the first evaluation, and commit them.
- Run once in `live` mode (fills `data/cache/search`), then use `VRS_SEARCH_MODE=replay` for every comparison so all configurations see identical search results.
- Report the numbers you get, including bad ones.
