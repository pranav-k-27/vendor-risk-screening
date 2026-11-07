"""Content-addressed source snapshots.

Every byte any agent reasons over is stored here first. The full sha256 is kept
in the metadata (tamper evidence); the first 16 hex chars are the snapshot_id
and file name. Re-fetching identical bytes yields the same snapshot_id.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.config import snapshots_dir
from core.schemas import Snapshot


class SnapshotManager:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else snapshots_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- write -------------------------------------------------------------
    def save(
        self,
        raw: bytes,
        source_url: str,
        source_type: str,
        text: str | None = None,
        title: str | None = None,
        published_date: str | None = None,
        suffix: str = ".bin",
    ) -> Snapshot:
        """Store raw bytes (hashed) plus extracted text (what agents read)."""
        sha = hashlib.sha256(raw).hexdigest()
        sid = sha[:16]
        meta_path = self.root / f"{sid}.json"
        if meta_path.exists():
            existing = Snapshot.model_validate_json(meta_path.read_text(encoding="utf-8"))
            if Path(existing.content_path).exists() and Path(existing.text_path).exists():
                return existing
            # metadata without its files (partial write / old layout): re-write them below

        content_path = self.root / f"{sid}{suffix}"
        # Separate name so a ".txt" source never shares a path with its extracted text
        # (which would overwrite the hashed raw bytes).
        text_path = self.root / f"{sid}.extracted.txt"
        content_path.write_bytes(raw)
        if text is None:
            text = raw.decode("utf-8", errors="replace")
        # write_bytes, not write_text: write_text turns LF into CRLF on Windows
        text_path.write_bytes(text.encode("utf-8"))

        snap = Snapshot(
            snapshot_id=sid,
            sha256=sha,
            source_url=source_url,
            source_type=source_type,  # type: ignore[arg-type]
            title=title,
            published_date=published_date,
            content_path=str(content_path),
            text_path=str(text_path),
            byte_length=len(raw),
        )
        meta_path.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
        return snap

    # -- read --------------------------------------------------------------
    def get(self, snapshot_id: str) -> Snapshot | None:
        p = self.root / f"{snapshot_id}.json"
        if not p.exists():
            return None
        return Snapshot.model_validate_json(p.read_text(encoding="utf-8"))

    def read_text(self, snapshot_id: str) -> str | None:
        snap = self.get(snapshot_id)
        if snap is None:
            return None
        return Path(snap.text_path).read_bytes().decode("utf-8")

    def verify(self, snapshot_id: str) -> bool:
        """True if stored bytes still hash to the recorded sha256."""
        snap = self.get(snapshot_id)
        if snap is None:
            return False
        return hashlib.sha256(Path(snap.content_path).read_bytes()).hexdigest() == snap.sha256

    def manifest(self, snapshot_ids: list[str]) -> list[dict]:
        out = []
        for sid in sorted(set(snapshot_ids)):
            snap = self.get(sid)
            if snap:
                out.append(json.loads(snap.model_dump_json(include={"snapshot_id", "sha256", "source_url", "fetched_at"})))
        return out
