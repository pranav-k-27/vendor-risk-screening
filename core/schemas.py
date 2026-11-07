"""Framework-independent data contracts.

Every orchestrator (CrewAI today, LangGraph later) must produce and consume
these models. Nothing in this module imports an agent framework.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class VendorInput(BaseModel):
    name: str = Field(..., min_length=2, description="Legal or trading name")
    country: str = Field("IN", description="ISO-2 country code")
    website: str | None = None
    lei: str | None = Field(None, description="If known, skips fuzzy entity resolution")
    nse_symbol: str | None = None
    aliases: list[str] = Field(default_factory=list)
    confirmed: bool = Field(
        False,
        description="Set True after a human has confirmed identity (e.g. synthetic or no-LEI vendors)",
    )


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------
class EntityStatus(str, Enum):
    RESOLVED = "resolved"                    # high-confidence registry match
    USER_CONFIRMED = "user_confirmed"        # human confirmed identity
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNRESOLVED = "unresolved"                # nothing found in registry


class EntityCandidate(BaseModel):
    lei: str | None = None
    legal_name: str
    other_names: list[str] = Field(default_factory=list)
    country: str | None = None
    city: str | None = None
    registration_status: str | None = None
    match_score: float = 0.0


class EntityResolution(BaseModel):
    status: EntityStatus
    input_name: str
    resolved_name: str
    lei: str | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    method: str = ""
    candidates: list[EntityCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        names = [self.resolved_name, self.input_name, *self.aliases]
        seen, out = set(), []
        for n in names:
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out


# ---------------------------------------------------------------------------
# Sanctions
# ---------------------------------------------------------------------------
class MatchStrength(str, Enum):
    STRONG = "strong"        # treated as a hard stop pending human review
    POTENTIAL = "potential"  # flagged for review, does not add score


class SanctionsHit(BaseModel):
    list_name: str
    entry_id: str
    listed_name: str
    matched_on: str
    query_name: str
    score: float
    strength: MatchStrength
    program: str | None = None


class SanctionsResult(BaseModel):
    screened_names: list[str]
    lists_checked: list[str]
    list_versions: dict[str, str] = Field(default_factory=dict)
    entries_screened: int = 0
    hits: list[SanctionsHit] = Field(default_factory=list)

    @property
    def has_strong_hit(self) -> bool:
        return any(h.strength == MatchStrength.STRONG for h in self.hits)

    @property
    def has_potential_hit(self) -> bool:
        return any(h.strength == MatchStrength.POTENTIAL for h in self.hits)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
class Snapshot(BaseModel):
    snapshot_id: str = Field(..., description="First 16 hex chars of sha256 (file name)")
    sha256: str
    source_url: str
    source_type: Literal["web", "news", "regulatory_document", "sanctions_list", "registry", "fixture"]
    title: str | None = None
    published_date: str | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    content_path: str
    text_path: str
    byte_length: int


class RiskDimension(str, Enum):
    REGULATORY = "regulatory"
    ADVERSE_MEDIA = "adverse_media"


class EvidenceType(str, Enum):
    OFFICIAL_FINDING = "official_finding"    # regulator order imposing penalty/direction
    SETTLEMENT = "settlement"                # settled without admission
    UNDER_APPEAL = "under_appeal"            # order exists but is under appeal
    MEDIA_CONFIRMED = "media_confirmed"      # media reporting an official action
    MEDIA_ALLEGATION = "media_allegation"    # allegation not (yet) substantiated
    UNVERIFIED = "unverified"                # weak/anonymous source


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RawFinding(BaseModel):
    """What a research agent is allowed to output. Nothing more."""

    title: str = Field(..., description="One-line factual description")
    summary: str = Field(..., description="2-3 sentences, factual, no speculation")
    dimension: RiskDimension
    evidence_type: EvidenceType
    severity: Severity
    source_url: str
    snapshot_id: str = Field(..., description="snapshot_id returned by the tool that fetched the source")
    supporting_passage: str = Field(
        ..., description="Verbatim passage copied from the source that supports the finding (20-600 chars)"
    )
    event_date: str | None = Field(None, description="YYYY-MM-DD if stated in the source")
    entity_name_in_source: str = Field(..., description="Exact company name as written in the source")


class FindingsBatch(BaseModel):
    findings: list[RawFinding] = Field(default_factory=list)
    queries_run: list[str] = Field(default_factory=list)
    sources_reviewed: int = 0
    notes: str = ""


class ValidatedFinding(RawFinding):
    finding_id: str
    agent: str
    verified: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    passage_match_score: float = 0.0
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues, e.g. figures not in the source")
    merged_finding_ids: list[str] = Field(default_factory=list, description="Other findings for the same event, counted once")


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
class SourceStatus(str, Enum):
    OK = "ok"
    NO_DATA = "no_data"     # source checked, nothing relevant
    FAILED = "failed"       # source could not be checked
    SKIPPED = "skipped"     # deliberately not run (e.g. sanctions hard stop)


class ScoreContribution(BaseModel):
    finding_id: str
    dimension: RiskDimension
    base_points: float
    evidence_multiplier: float
    recency_multiplier: float
    points: float
    rule: str


class RiskBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class RiskAssessment(BaseModel):
    score: float = Field(..., ge=0, le=100)
    band: RiskBand
    confidence: ConfidenceLevel
    completeness: float = Field(..., ge=0, le=1)
    dimension_scores: dict[str, float]
    contributions: list[ScoreContribution]
    hard_stop: bool = False
    review_flags: list[str] = Field(default_factory=list)
    source_status: dict[str, SourceStatus]
    rules_version: str
    rationale: list[str] = Field(default_factory=list, description="Deterministic, rule-generated reasons")


# ---------------------------------------------------------------------------
# Explanation (LLM) and its deterministic check
# ---------------------------------------------------------------------------
class Explanation(BaseModel):
    executive_summary: str = Field(..., description="3-5 sentences. Must not restate a different score or band.")
    key_drivers: list[str] = Field(default_factory=list, description="Each item must cite finding_ids in [brackets]")
    evidence_gaps: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    cited_finding_ids: list[str] = Field(default_factory=list)


class ExplanationCheck(BaseModel):
    consistent: bool
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Report + review
# ---------------------------------------------------------------------------
class ReportStatus(str, Enum):
    NEEDS_ENTITY_CONFIRMATION = "NEEDS_ENTITY_CONFIRMATION"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


class ReviewDecision(BaseModel):
    decision: Literal["APPROVED", "REJECTED", "ESCALATED"]
    reviewer: str
    comment: str = ""
    decided_at: datetime = Field(default_factory=utcnow)
    reviewed_report_sha256: str


class RunMetrics(BaseModel):
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    successful_requests: int = 0


class AssessmentReport(BaseModel):
    run_id: str
    orchestrator: str
    model: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    status: ReportStatus
    vendor: VendorInput
    entity: EntityResolution | None = None
    sanctions: SanctionsResult | None = None
    findings: list[ValidatedFinding] = Field(default_factory=list)
    rejected_findings: list[ValidatedFinding] = Field(default_factory=list)
    assessment: RiskAssessment | None = None
    explanation: Explanation | None = None
    explanation_check: ExplanationCheck | None = None
    errors: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    review: ReviewDecision | None = None
