from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000, description="User or student query question")
    category: str | None = Field(default=None, description="Optional document category filter (e.g., Admissions)")
    vector_backend: str = Field(default="chroma", description="Vector database backend to query")
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Audit escalation threshold below which inquiries are logged for instructor review",
    )


class SourceCitation(BaseModel):
    document_id: UUID | None = Field(default=None, description="Database UUID of the referenced document if indexed")
    filename: str = Field(default="unknown", description="Name of the referenced source document")
    category: str | None = Field(default=None, description="Category of the source document")
    similarity_score: float | None = Field(default=None, description="Similarity or reranking match score")
    content_snippet: str | None = Field(default=None, description="Snapshot excerpt of the referenced text chunk")


class ChatQueryResponse(BaseModel):
    answer: str = Field(..., description="Synthesized AI response to the user's question")
    sources: list[SourceCitation] = Field(default_factory=list, description="List of source citations used")
    confidence_score: float = Field(..., description="Calculated confidence or semantic match score (0.0 - 1.0)")
    escalated_to_audit: bool = Field(..., description="Whether the inquiry was flagged for instructor audit review")
    audit_id: UUID | None = Field(default=None, description="UUID of the persisted AuditLog record if escalated")
