"""Deterministic claim-to-source validation.

An agent's finding is VERIFIED only if:
  1. its snapshot_id exists in the snapshot store and the bytes still hash correctly
  2. its source_url matches the URL recorded in that snapshot
  3. its supporting_passage actually appears in the snapshot text (fuzzy, >= threshold)
  4. the snapshot text mentions the resolved entity (any known name)
  5. passage length is within bounds
Rejected findings are kept (for the reviewer and the unsupported-claim metric)
but never scored.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from rapidfuzz import fuzz

from core.config import load_rules
from core.schemas import EntityResolution, EvidenceType, RawFinding, ValidatedFinding
from core.snapshot_manager import SnapshotManager
from core.text import normalize_name, normalize_text


def _finding_id(agent: str, f: RawFinding) -> str:
    key = f"{agent}|{f.snapshot_id}|{normalize_text(f.supporting_passage)[:200]}"
    return "F-" + hashlib.sha256(key.encode()).hexdigest()[:8]


def _url_key(url: str) -> str:
    return url.strip().lower().rstrip("/").replace("https://", "").replace("http://", "").replace("www.", "")


OFFICIAL_EVIDENCE = {EvidenceType.OFFICIAL_FINDING, EvidenceType.SETTLEMENT, EvidenceType.UNDER_APPEAL}

_MONEY = re.compile(
    r"(?:rs\.?|inr|₹|\$|usd)?\s*(\d[\d,]*(?:\.\d+)?)\s*(crores?|cr\b|lakhs?|lacs?|million|mn\b|billion|bn\b)",
    re.IGNORECASE,
)
_UNIT = {"crore": "crore", "crores": "crore", "cr": "crore", "lakh": "lakh", "lakhs": "lakh", "lac": "lakh",
         "lacs": "lakh", "million": "million", "mn": "million", "billion": "billion", "bn": "billion"}


def _norm_amount(num: str, unit: str) -> str:
    return f"{num.replace(',', '')}{_UNIT[unit.lower()]}"


def unsupported_figures(claim_text: str, source_norm: str) -> list[str]:
    """Money amounts stated in the claim that do not appear in the source.

    Both sides are compared with commas and spaces removed and units normalised,
    so "Rs. 25 crores" in a claim is supported by "rs. 25 crore" in the source.
    """
    compact = source_norm.lower().replace(",", "").replace(" ", "")
    for m in _MONEY.finditer(source_norm):
        compact += " " + _norm_amount(m.group(1), m.group(2))  # source units normalised too ("lacs" -> "lakh")
    out: list[str] = []
    for m in _MONEY.finditer(claim_text):
        key = _norm_amount(m.group(1), m.group(2))
        if key not in compact and m.group(0).strip() not in out:
            out.append(m.group(0).strip())
    return out


class EvidenceValidator:
    def __init__(self, snapshots: SnapshotManager | None = None, rules: dict | None = None):
        self.snapshots = snapshots or SnapshotManager()
        cfg = (rules or load_rules())["matching"]["validation"]
        self.passage_min = float(cfg["passage_min_score"])
        self.entity_min = float(cfg["entity_min_score"])
        self.min_chars = int(cfg["min_passage_chars"])
        self.max_chars = int(cfg["max_passage_chars"])
        self.regulator_domains = [d.lower().lstrip(".") for d in cfg.get("regulator_domains", [])]
        hedges = [w.strip() for w in cfg.get("hedge_words", []) if w.strip()]
        # longest first so "under investigation" is reported rather than a shorter overlap
        self.hedge_re = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in sorted(hedges, key=len, reverse=True)) + r")\b", re.IGNORECASE
        ) if hedges else None

    def _is_regulator_url(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return any(host == d or host.endswith("." + d) for d in self.regulator_domains)

    def _classification_guards(self, data: dict, snap, warnings: list[str]) -> None:
        """Deterministic corrections of the agent's evidence_type / event_date (mutates data)."""
        old = EvidenceType(data["evidence_type"])
        if old in OFFICIAL_EVIDENCE and snap.source_type != "regulatory_document" \
                and not self._is_regulator_url(snap.source_url):
            data["evidence_type"] = EvidenceType.MEDIA_CONFIRMED
            warnings.append(f"evidence_type downgraded from {old.value} to media_confirmed: source is not a regulator")
        if EvidenceType(data["evidence_type"]) == EvidenceType.MEDIA_CONFIRMED and self.hedge_re:
            m = self.hedge_re.search(data["supporting_passage"])
            if m:
                data["evidence_type"] = EvidenceType.MEDIA_ALLEGATION
                warnings.append(f"evidence_type downgraded to media_allegation: passage contains '{m.group(1).lower()}'")
        if not data.get("event_date") and snap.published_date:
            data["event_date"] = snap.published_date[:10]
            warnings.append("event_date taken from source publication date")

    def _entity_mentioned(self, text_norm: str, entity: EntityResolution) -> bool:
        for name in entity.all_names:
            core_name = normalize_name(name).lower()
            if not core_name:
                continue
            if core_name in text_norm:
                return True
            if len(core_name) >= 6 and fuzz.partial_ratio(core_name, text_norm) >= self.entity_min:
                return True
        return False

    def validate_one(self, f: RawFinding, agent: str, entity: EntityResolution) -> ValidatedFinding:
        reasons: list[str] = []
        score = 0.0
        plen = len(f.supporting_passage.strip())
        if plen < self.min_chars or plen > self.max_chars:
            reasons.append(f"passage length {plen} outside [{self.min_chars},{self.max_chars}]")

        snap = self.snapshots.get(f.snapshot_id)
        if snap is None:
            reasons.append("snapshot_id not found (source never fetched by a tool)")
        else:
            if not self.snapshots.verify(f.snapshot_id):
                reasons.append("snapshot hash mismatch (content altered)")
            if _url_key(snap.source_url) != _url_key(f.source_url):
                reasons.append("source_url does not match snapshot")
            text_norm = normalize_text(self.snapshots.read_text(f.snapshot_id) or "")
            passage_norm = normalize_text(f.supporting_passage)
            if passage_norm and passage_norm in text_norm:
                score = 100.0
            elif passage_norm:
                score = float(fuzz.partial_ratio(passage_norm, text_norm))
            if score < self.passage_min:
                reasons.append(f"supporting_passage not found in source (match {score:.0f})")
            if not self._entity_mentioned(text_norm, entity):
                reasons.append("source does not mention the resolved entity")

        warnings: list[str] = []
        data = f.model_dump()
        if not reasons and snap is not None:
            missing = unsupported_figures(f"{f.title} {f.summary}", text_norm)
            if missing:
                warnings.append("amounts in title/summary not found in source: " + ", ".join(missing))
            self._classification_guards(data, snap, warnings)

        return ValidatedFinding(
            **data,
            warnings=warnings,
            finding_id=_finding_id(agent, f),  # id from the ORIGINAL agent output
            agent=agent,
            verified=not reasons,
            rejection_reasons=reasons,
            passage_match_score=round(score, 1),
        )

    def validate(
        self, batches: dict[str, list[RawFinding]], entity: EntityResolution
    ) -> tuple[list[ValidatedFinding], list[ValidatedFinding]]:
        verified: list[ValidatedFinding] = []
        rejected: list[ValidatedFinding] = []
        seen: set[str] = set()
        for agent, findings in batches.items():
            for f in findings:
                vf = self.validate_one(f, agent, entity)
                if vf.finding_id in seen:
                    continue  # exact duplicate (same agent, source, passage)
                seen.add(vf.finding_id)
                (verified if vf.verified else rejected).append(vf)
        return self._dedupe_events(verified), rejected

    @staticmethod
    def _dedupe_events(findings: list[ValidatedFinding]) -> list[ValidatedFinding]:
        """Same event reported several times: keep the strongest evidence once.

        A finding is merged into an already kept one if
          - it cites the same snapshot and the dates match (or either is missing):
            several findings extracted from one order are one event; or
          - the dates match and the titles are near-identical, even across dimensions
            (a SEBI order and the news story about it must not be counted twice).
        Merged ids are recorded on the kept finding, never dropped silently.
        """
        rank = {"official_finding": 0, "under_appeal": 1, "settlement": 2, "media_confirmed": 3,
                "media_allegation": 4, "unverified": 5}
        sev = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        kept: list[ValidatedFinding] = []
        for f in sorted(findings, key=lambda x: (rank[x.evidence_type.value], sev[x.severity.value])):
            target = None
            for k in kept:
                same_source = k.snapshot_id == f.snapshot_id and (
                    not k.event_date or not f.event_date or k.event_date == f.event_date)
                same_story = bool(k.event_date) and k.event_date == f.event_date and fuzz.token_set_ratio(
                    normalize_text(k.title), normalize_text(f.title)) >= 85
                if same_source or same_story:
                    target = k
                    break
            if target is None:
                kept.append(f)
            else:
                target.merged_finding_ids.append(f.finding_id)
        return kept
