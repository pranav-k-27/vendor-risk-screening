"""Thin CrewAI wrappers around core.sources. No business logic lives here."""
from __future__ import annotations

import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from core import sources
from core.sources import UNTRUSTED_NOTE


class ToolCallCounter:
    """Shared by all tools of one crew. A plain object (not a dict) so pydantic keeps the reference
    instead of copying it on validation."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def add(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1


class CountedTool(BaseTool):
    """Counts its own executions (for per-agent cost metrics). Calls answered from CrewAI's tool
    cache do not reach _run and are not counted."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    call_counter: ToolCallCounter | None = Field(default=None, exclude=True)

    def _count(self) -> None:
        if self.call_counter is not None:
            self.call_counter.add(self.name)


class _SearchArgs(BaseModel):
    query: str = Field(..., description="Search query, e.g. '\"Acme Infotech\" SEBI penalty'")
    max_results: int = Field(5, ge=1, le=8)


class AdverseMediaSearchTool(CountedTool):
    name: str = "adverse_media_search"
    description: str = (
        "Search the web/news. Every result is stored as a snapshot first. Returns snapshot_id, url, title, "
        "published_date and a short excerpt. Use read_source with the snapshot_id to read the full text "
        "before citing it. " + UNTRUSTED_NOTE
    )
    args_schema: Type[BaseModel] = _SearchArgs

    def _run(self, query: str, max_results: int = 5) -> str:
        self._count()
        try:
            res = sources.web_search(query, max_results=max_results)
        except sources.SourceError as exc:
            return json.dumps({"error": str(exc)})
        for r in res:
            r["excerpt"] = sources.wrap_untrusted(r["snapshot_id"], r["excerpt"])
        return json.dumps(res, ensure_ascii=False)


class _ListDocsArgs(BaseModel):
    company_name: str = Field(..., description="Vendor name exactly as given in the task")


class RegulatoryDocumentsTool(CountedTool):
    name: str = "list_regulatory_documents"
    description: str = (
        "List curated regulatory documents (e.g. SEBI orders) collected for this vendor. Returns snapshot_id, "
        "url, title, date and size (chars) for each. Find passages with search_source, then read_source "
        "around the offsets it returns."
    )
    args_schema: Type[BaseModel] = _ListDocsArgs

    def _run(self, company_name: str) -> str:
        self._count()
        return json.dumps(sources.list_documents(company_name), ensure_ascii=False)


class _ReadArgs(BaseModel):
    snapshot_id: str = Field(..., description="snapshot_id returned by a search/list tool")
    start: int = Field(0, ge=0, description="Character offset; use the 'end' value of the previous call to continue")
    max_chars: int = Field(12000, ge=500, le=12000, description="Characters to return (default 12000)")


class ReadSourceTool(CountedTool):
    name: str = "read_source"
    description: str = (
        "Read the stored text of a snapshot from character offset 'start' (up to 12,000 characters per call; "
        "set max_chars lower to read just around a passage; 'more'=true means there is text after 'end'). "
        "Copy supporting passages VERBATIM from this text. " + UNTRUSTED_NOTE
    )
    args_schema: Type[BaseModel] = _ReadArgs

    def _run(self, snapshot_id: str, start: int = 0, max_chars: int = 12000) -> str:
        self._count()
        try:
            return json.dumps(sources.read_snapshot(snapshot_id, start=start, max_chars=max_chars), ensure_ascii=False)
        except sources.SourceError as exc:
            return json.dumps({"error": str(exc)})


class _SearchSourceArgs(BaseModel):
    snapshot_id: str = Field(..., description="snapshot_id returned by list_regulatory_documents")
    query: str = Field(..., description="Keywords, e.g. 'penalty imposed', 'hereby order', 'directions'")
    top_k: int = Field(5, ge=1, le=8, description="Number of passages to return")


class SearchSourceTool(CountedTool):
    name: str = "search_source"
    description: str = (
        "Search inside ONE stored document for passages matching keywords. Returns the best ~1500-character "
        "passages with their 'start'/'end' character offsets. Use read_source(start=...) to read around a hit "
        "instead of reading the whole document. " + UNTRUSTED_NOTE
    )
    args_schema: Type[BaseModel] = _SearchSourceArgs

    def _run(self, snapshot_id: str, query: str, top_k: int = 5) -> str:
        self._count()
        try:
            return json.dumps(sources.search_source(snapshot_id, query, top_k=top_k), ensure_ascii=False)
        except sources.SourceError as exc:
            return json.dumps({"error": str(exc)})
