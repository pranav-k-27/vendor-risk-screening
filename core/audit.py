"""Append-only, hash-chained audit log (JSON Lines).

Each record stores the sha256 of the previous record, so editing or deleting
any line breaks the chain and `verify_chain()` reports where.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.config import audit_log_path
from core.schemas import utcnow

GENESIS = "0" * 64


def _hash(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


class AuditLogger:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else audit_log_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = None
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        return json.loads(last)["record_hash"] if last else GENESIS

    def log(self, run_id: str, event: str, payload: dict[str, Any] | None = None) -> dict:
        body = {
            "ts": utcnow().isoformat(),
            "run_id": run_id,
            "event": event,
            "payload": payload or {},
            "prev_hash": self._last_hash(),
        }
        body["record_hash"] = _hash(body)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(body, default=str) + "\n")
        return body

    def verify_chain(self) -> tuple[bool, int | None]:
        """Returns (ok, first_bad_line_number)."""
        if not self.path.exists():
            return True, None
        prev = GENESIS
        with open(self.path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                claimed = rec.pop("record_hash")
                if rec["prev_hash"] != prev or _hash(rec) != claimed:
                    return False, n
                prev = claimed
        return True, None

    def events(self, run_id: str) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [r for r in (json.loads(l) for l in fh if l.strip()) if r["run_id"] == run_id]
