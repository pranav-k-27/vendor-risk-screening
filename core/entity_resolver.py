"""Deterministic entity resolution using the GLEIF LEI registry.

Order of precedence:
  1. LEI supplied           -> direct registry lookup, confidence 1.0
  2. Human already confirmed -> USER_CONFIRMED (e.g. synthetic or no-LEI vendors)
  3. Full-text registry search, scored locally with rapidfuzz.
     Auto-accept only if the best candidate is strong AND clearly ahead of the
     runner-up; otherwise ask a human (NEEDS_CONFIRMATION).
Registry responses are snapshotted so the identity decision is auditable.
"""
from __future__ import annotations

import json
from typing import Callable

from rapidfuzz import fuzz

from core import http
from core.config import load_rules
from core.schemas import EntityCandidate, EntityResolution, EntityStatus, VendorInput
from core.snapshot_manager import SnapshotManager
from core.text import normalize_name

GLEIF_BASE = "https://api.gleif.org/api/v1/lei-records"

Fetcher = Callable[[str, dict | None], dict]


def _default_fetch(url: str, params: dict | None) -> dict:
    return http.get(url, params=params).json()


def _candidate_from_record(rec: dict) -> EntityCandidate:
    attrs = rec.get("attributes", {})
    ent = attrs.get("entity", {})
    addr = ent.get("legalAddress", {}) or {}
    return EntityCandidate(
        lei=attrs.get("lei") or rec.get("id"),
        legal_name=(ent.get("legalName") or {}).get("name", ""),
        other_names=[o.get("name", "") for o in ent.get("otherNames", []) or [] if o.get("name")],
        country=addr.get("country"),
        city=addr.get("city"),
        registration_status=(attrs.get("registration") or {}).get("status"),
    )


def _score(query: str, cand: EntityCandidate) -> float:
    qn = normalize_name(query)
    names = [cand.legal_name, *cand.other_names]
    return max((fuzz.token_sort_ratio(qn, normalize_name(n)) for n in names if n), default=0.0)


class EntityResolver:
    def __init__(self, fetch: Fetcher | None = None, snapshots: SnapshotManager | None = None, rules: dict | None = None):
        self.fetch = fetch or _default_fetch
        self.snapshots = snapshots or SnapshotManager()
        cfg = (rules or load_rules())["matching"]["entity"]
        self.accept = float(cfg["auto_accept_score"])
        self.margin = float(cfg["auto_accept_margin"])

    def _snapshot(self, url: str, params: dict | None, payload: dict) -> str:
        raw = json.dumps({"url": url, "params": params, "response": payload}, sort_keys=True).encode()
        return self.snapshots.save(raw, source_url=url, source_type="registry", suffix=".json").snapshot_id

    def resolve(self, vendor: VendorInput) -> EntityResolution:
        aliases = list(vendor.aliases)

        # 1. LEI supplied
        if vendor.lei:
            url = f"{GLEIF_BASE}/{vendor.lei.strip().upper()}"
            try:
                payload = self.fetch(url, None)
                sid = self._snapshot(url, None, payload)
                cand = _candidate_from_record(payload.get("data", {}))
                return EntityResolution(
                    status=EntityStatus.RESOLVED,
                    input_name=vendor.name,
                    resolved_name=cand.legal_name or vendor.name,
                    lei=cand.lei,
                    aliases=list(dict.fromkeys(aliases + cand.other_names)),
                    confidence=1.0,
                    method="gleif_lei_lookup",
                    candidates=[cand],
                    notes=[f"registry snapshot {sid}"],
                )
            except Exception as exc:  # registry down or bad LEI: fall through to a human
                return EntityResolution(
                    status=EntityStatus.NEEDS_CONFIRMATION,
                    input_name=vendor.name,
                    resolved_name=vendor.name,
                    lei=vendor.lei,
                    aliases=aliases,
                    confidence=0.0,
                    method="gleif_lei_lookup_failed",
                    notes=[f"LEI lookup failed: {exc!s}"[:300]],
                )

        # 2. Human already confirmed identity
        if vendor.confirmed:
            return EntityResolution(
                status=EntityStatus.USER_CONFIRMED,
                input_name=vendor.name,
                resolved_name=vendor.name,
                aliases=aliases,
                confidence=0.95,
                method="user_confirmed",
            )

        # 3. Registry search
        params = {"filter[fulltext]": vendor.name, "page[size]": 10}
        if vendor.country:
            params["filter[entity.legalAddress.country]"] = vendor.country.upper()
        try:
            payload = self.fetch(GLEIF_BASE, params)
        except Exception as exc:
            return EntityResolution(
                status=EntityStatus.UNRESOLVED,
                input_name=vendor.name,
                resolved_name=vendor.name,
                aliases=aliases,
                confidence=0.0,
                method="gleif_search_failed",
                notes=[f"Registry search failed: {exc!s}"[:300]],
            )
        sid = self._snapshot(GLEIF_BASE, params, payload)

        cands = [_candidate_from_record(r) for r in payload.get("data", []) or []]
        for c in cands:
            c.match_score = round(_score(vendor.name, c), 1)
        cands.sort(key=lambda c: -c.match_score)

        if not cands:
            return EntityResolution(
                status=EntityStatus.UNRESOLVED,
                input_name=vendor.name,
                resolved_name=vendor.name,
                aliases=aliases,
                confidence=0.0,
                method="gleif_search",
                notes=[f"No registry candidates (snapshot {sid}). Confirm identity manually."],
            )

        best = cands[0]
        runner_up = cands[1].match_score if len(cands) > 1 else 0.0
        clear_winner = best.match_score >= self.accept and (best.match_score - runner_up) >= self.margin
        active = (best.registration_status or "").upper() in {"ISSUED", "PENDING_TRANSFER", "PENDING_ARCHIVAL", ""}
        notes = [f"registry snapshot {sid}", f"best={best.match_score} runner_up={runner_up}"]
        if not active:
            notes.append(f"LEI registration status is {best.registration_status}")

        return EntityResolution(
            status=EntityStatus.RESOLVED if clear_winner else EntityStatus.NEEDS_CONFIRMATION,
            input_name=vendor.name,
            resolved_name=best.legal_name if clear_winner else vendor.name,
            lei=best.lei if clear_winner else None,
            aliases=list(dict.fromkeys(aliases + (best.other_names if clear_winner else []))),
            confidence=round(best.match_score / 100.0, 3) if clear_winner else round(best.match_score / 200.0, 3),
            method="gleif_search",
            candidates=cands[:5],
            notes=notes,
        )
