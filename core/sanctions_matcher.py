"""Deterministic sanctions screening against full official lists.

Lists (download with scripts/download_sanctions.py):
  data/sanctions/ofac_sdn.csv   OFAC SDN primary names (no header row)
  data/sanctions/ofac_alt.csv   OFAC SDN aliases (no header row)
  data/sanctions/un_consolidated.xml  UN Security Council consolidated list
  data/sanctions/synthetic.csv  clearly-fake test entries (header row)

Only non-individual entries are screened: vendors are organisations.
Matching is on normalised names (legal suffixes removed) with rapidfuzz
token_sort_ratio. One-token names require an exact normalised match, because
fuzzy matching short generic names is where false positives come from.
No LLM is involved anywhere in this module.
"""
from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from core.config import load_rules, sanctions_dir
from core.schemas import MatchStrength, SanctionsHit, SanctionsResult
from core.text import normalize_name


@dataclass(frozen=True)
class ListEntry:
    list_name: str
    entry_id: str
    primary_name: str
    name: str          # the specific name/alias this row represents
    norm: str
    program: str | None


def _file_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_ofac(sdn_path: Path, alt_path: Path | None = None) -> list[ListEntry]:
    entries: list[ListEntry] = []
    primary: dict[str, tuple[str, str]] = {}
    with open(sdn_path, encoding="latin-1", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            ent_num, name, sdn_type, program = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            if not ent_num.isdigit() or sdn_type.lower() == "individual":
                continue
            primary[ent_num] = (name, program)
            entries.append(ListEntry("OFAC_SDN", ent_num, name, name, normalize_name(name), program))
    if alt_path and alt_path.exists():
        with open(alt_path, encoding="latin-1", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 4:
                    continue
                ent_num, alt_name = row[0].strip(), row[3].strip()
                if ent_num in primary and alt_name and alt_name != "-0-":
                    pname, program = primary[ent_num]
                    entries.append(ListEntry("OFAC_SDN", ent_num, pname, alt_name, normalize_name(alt_name), program))
    return entries


def load_un(xml_path: Path) -> list[ListEntry]:
    entries: list[ListEntry] = []
    root = ET.parse(xml_path).getroot()
    for ent in root.iter("ENTITY"):
        ref = (ent.findtext("REFERENCE_NUMBER") or ent.findtext("DATAID") or "").strip()
        name = (ent.findtext("FIRST_NAME") or "").strip()
        program = (ent.findtext("UN_LIST_TYPE") or "").strip() or None
        if not name:
            continue
        entries.append(ListEntry("UN_CONSOLIDATED", ref, name, name, normalize_name(name), program))
        for alias in ent.iter("ENTITY_ALIAS"):
            an = (alias.findtext("ALIAS_NAME") or "").strip()
            if an:
                entries.append(ListEntry("UN_CONSOLIDATED", ref, name, an, normalize_name(an), program))
    return entries


def load_synthetic(path: Path) -> list[ListEntry]:
    entries: list[ListEntry] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            names = [row["name"], *[a for a in (row.get("aliases") or "").split("|") if a]]
            for n in names:
                entries.append(
                    ListEntry("SYNTHETIC_TEST", row["entry_id"], row["name"], n, normalize_name(n), row.get("program"))
                )
    return entries


class SanctionsMatcher:
    def __init__(self, entries: list[ListEntry], versions: dict[str, str], rules: dict | None = None):
        self.entries = [e for e in entries if e.norm]
        self.versions = versions
        cfg = (rules or load_rules())["matching"]["sanctions"]
        self.strong = float(cfg["strong_threshold"])
        self.potential = float(cfg["potential_threshold"])
        self.min_tokens = int(cfg["short_name_min_tokens"])
        self._choices = [e.norm for e in self.entries]
        self._exact: dict[str, list[int]] = {}
        for i, e in enumerate(self.entries):
            self._exact.setdefault(e.norm, []).append(i)

    @classmethod
    def from_directory(cls, root: Path | None = None, include_synthetic: bool = True) -> "SanctionsMatcher":
        root = Path(root) if root else sanctions_dir()
        entries: list[ListEntry] = []
        versions: dict[str, str] = {}
        sdn, alt = root / "ofac_sdn.csv", root / "ofac_alt.csv"
        if sdn.exists():
            entries += load_ofac(sdn, alt)
            versions["OFAC_SDN"] = _file_version(sdn)
        un = root / "un_consolidated.xml"
        if un.exists():
            entries += load_un(un)
            versions["UN_CONSOLIDATED"] = _file_version(un)
        syn = root / "synthetic.csv"
        if include_synthetic and syn.exists():
            entries += load_synthetic(syn)
            versions["SYNTHETIC_TEST"] = _file_version(syn)
        return cls(entries, versions)

    @property
    def lists_loaded(self) -> list[str]:
        return sorted(self.versions)

    def screen(self, names: list[str]) -> SanctionsResult:
        hits: dict[tuple[str, str], SanctionsHit] = {}
        for q in names:
            qn = normalize_name(q)
            if not qn:
                continue
            candidates: list[tuple[int, float]] = [(i, 100.0) for i in self._exact.get(qn, [])]
            if len(qn.split()) >= self.min_tokens:
                for _, score, idx in process.extract(
                    qn, self._choices, scorer=fuzz.token_sort_ratio, score_cutoff=self.potential, limit=25
                ):
                    if len(self.entries[idx].norm.split()) >= self.min_tokens:
                        candidates.append((idx, float(score)))
            for idx, score in candidates:
                e = self.entries[idx]
                strength = MatchStrength.STRONG if score >= self.strong else MatchStrength.POTENTIAL
                key = (e.list_name, e.entry_id)
                if key not in hits or hits[key].score < score:
                    hits[key] = SanctionsHit(
                        list_name=e.list_name,
                        entry_id=e.entry_id,
                        listed_name=e.primary_name,
                        matched_on=e.name,
                        query_name=q,
                        score=round(score, 1),
                        strength=strength,
                        program=e.program,
                    )
        return SanctionsResult(
            screened_names=names,
            lists_checked=self.lists_loaded,
            list_versions=self.versions,
            entries_screened=len(self.entries),
            hits=sorted(hits.values(), key=lambda h: -h.score),
        )
