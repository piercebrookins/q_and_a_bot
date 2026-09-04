from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, Field


class ClaimLabel(StrEnum):
    FACT = "fact"
    INFERENCE = "inference"


class Evidence(BaseModel):
    evidence_id: str
    source_type: str
    title: str
    created_at: str | None = None
    customer_name: str | None = None
    region: str | None = None
    country: str | None = None
    excerpt: str
    score: float = 0.0

    def prompt_text(self) -> str:
        metadata = ", ".join(
            x for x in (self.source_type, self.customer_name, self.region, self.country, self.created_at) if x
        )
        return f"[{self.evidence_id}] {self.title} ({metadata})\n{self.excerpt}"


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=1500)
    label: ClaimLabel
    evidence_ids: list[str] = Field(min_length=1, max_length=12)


class AnswerBundle(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    uncertainty: list[str] = Field(default_factory=list, max_length=5)
    conflicts: list[str] = Field(default_factory=list, max_length=5)
    abstained: bool = False
    gap: str | None = Field(default=None, max_length=1000)


class EvidenceReport(BaseModel):
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    findings: list[str] = Field(default_factory=list, max_length=12)
    gaps: list[str] = Field(default_factory=list, max_length=5)


class GraphState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    route: str
    search_query: str
    focus_customer: str
    evidence: list[dict]
    answer: dict
    error: str
    tool_calls: int
    model_calls: int
    rewrites: int
