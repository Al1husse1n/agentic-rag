"""Pydantic schemas for Golden Q&A tactical hotfix management."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenQAEntry(BaseModel):
    question: str = Field(..., min_length=2, description="Verified question prompt")
    answer: str = Field(..., min_length=2, description="Authoritative verified answer")
    source_doc_id: str = Field(..., description="Referenced document name or ID")
    instructor_id: str = Field(default="Instructor", description="Name or ID of instructor who approved the override")
    match_score: float | None = Field(default=None, description="Fuzzy match similarity score (0.0 to 1.0)")


class GoldenQACreateRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Question prompt to hotfix")
    answer: str = Field(..., min_length=2, max_length=5000, description="Verified answer to override")
    source_doc_id: str = Field(..., min_length=1, max_length=255, description="Source document ID or filename")
    instructor_id: str | None = Field(default=None, description="Optional instructor identifier (defaults to current user)")


class GoldenQAListResponse(BaseModel):
    total: int = Field(..., description="Total number of cached golden Q&A entries")
    items: list[GoldenQAEntry] = Field(default_factory=list, description="List of golden Q&A overrides")


class GoldenQACheckRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Question to check against golden cache")
    threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Minimum fuzzy match similarity threshold")


class GoldenQACheckResponse(BaseModel):
    matched: bool = Field(..., description="Whether a matching golden Q&A hotfix was found")
    entry: GoldenQAEntry | None = Field(default=None, description="Matched golden Q&A record if found")

