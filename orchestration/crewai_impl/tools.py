"""Thin CrewAI wrappers around core.sources. No business logic lives here."""
from __future__ import annotations

import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from core import sources
from core.sources import UNTRUSTED_NOTE


class _SearchArgs(BaseModel):
    query: str = Field(..., description="Search query, e.g. '\"Acme Infotech\" SEBI penalty'")
    max_results: int = Field(5, ge=1, le=8)


class AdverseMediaSearchTool(BaseTool):
    name: str = "adverse_media_search"
    description: str = (
        "Search the web/news. Every result is stored as a snapshot first. Returns snapshot_id, url, title, "
        "published_date and a short excerpt. Use read_source with the snapshot_id to read the full text "
        "before citing it. " + UNTRUSTED_NOTE
    )
    args_schema: Type[BaseModel] = _SearchArgs

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            res = sources.web_search(query, max_results=max_results)
        except sources.SourceError as exc:
            return json.dumps({"error": str(exc)})
        for r in res:
            r["excerpt"] = sources.wrap_untrusted(r["snapshot_id"], r["excerpt"])
        return json.dumps(res, ensure_ascii=False)


class _ListDocsArgs(BaseModel):
    company_name: str = Field(..., description="Vendor name exactly as given in the task")


class RegulatoryDocumentsTool(BaseTool):
    name: str = "list_regulatory_documents"
    description: str = (
        "List curated regulatory documents (e.g. SEBI orders) collected for this vendor. Returns snapshot_id, "
        "url, title, date and size for each. Read them with read_source."
    )
    args_schema: Type[BaseModel] = _ListDocsArgs

    def _run(self, company_name: str) -> str:
        return json.dumps(sources.list_documents(company_name), ensure_ascii=False)


class _ReadArgs(BaseModel):
    snapshot_id: str = Field(..., description="snapshot_id returned by a search/list tool")
    start: int = Field(0, ge=0, description="Character offset; use the 'end' value of the previous call to continue")


class ReadSourceTool(BaseTool):
    name: str = "read_source"
    description: str = (
        "Read the stored text of a snapshot (12,000 characters per call; 'more'=true means call again with "
        "start=end). Copy supporting passages VERBATIM from this text. " + UNTRUSTED_NOTE
    )
    args_schema: Type[BaseModel] = _ReadArgs

    def _run(self, snapshot_id: str, start: int = 0) -> str:
        try:
            return json.dumps(sources.read_snapshot(snapshot_id, start=start), ensure_ascii=False)
        except sources.SourceError as exc:
            return json.dumps({"error": str(exc)})
