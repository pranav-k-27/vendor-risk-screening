"""Name and text normalisation shared by matching and validation."""
from __future__ import annotations

import re
import unicodedata

LEGAL_SUFFIXES = {
    "PVT", "PRIVATE", "LTD", "LIMITED", "LLC", "LLP", "INC", "INCORPORATED", "CO", "COMPANY",
    "CORP", "CORPORATION", "PLC", "GMBH", "AG", "SA", "SAS", "SPA", "BV", "NV", "FZE", "FZCO",
    "FZ", "JSC", "OJSC", "PJSC", "CJSC", "OOO", "TOO", "PTE", "PTY", "SDN", "BHD", "KG", "THE",
}

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_name(name: str) -> str:
    """Upper-case, strip accents/punctuation and legal suffixes. '&' -> 'AND'."""
    s = strip_accents(name or "").upper().replace("&", " AND ")
    s = _PUNCT.sub(" ", s)
    tokens = [t for t in _WS.split(s) if t and t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_text(text: str) -> str:
    """For passage matching: lower-case, collapse whitespace, unify quotes/dashes."""
    s = strip_accents(text or "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace(" ", " ")
    return _WS.sub(" ", s).strip().lower()
