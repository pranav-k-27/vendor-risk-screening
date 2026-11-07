"""Paths and rule loading. Override locations with environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return Path(os.getenv("VRS_DATA_DIR", REPO_ROOT / "data"))


def snapshots_dir() -> Path:
    return data_dir() / "snapshots"


def sanctions_dir() -> Path:
    return data_dir() / "sanctions"


def documents_dir() -> Path:
    return data_dir() / "documents"


def runs_dir() -> Path:
    return data_dir() / "runs"


def cache_dir() -> Path:
    return data_dir() / "cache"


def audit_log_path() -> Path:
    return data_dir() / "audit" / "audit_log.jsonl"


def rules_path() -> Path:
    return Path(os.getenv("VRS_RULES_PATH", REPO_ROOT / "config" / "rules.yaml"))


@lru_cache(maxsize=4)
def _load_rules(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_rules() -> dict:
    return _load_rules(str(rules_path()))


def search_mode() -> str:
    """'live' calls the search API and caches; 'replay' only reads the cache (reproducible evals)."""
    return os.getenv("VRS_SEARCH_MODE", "live").lower()
