"""Document Pydantic v2 Schemas.

Defines request validation and response serialization models for
the document ingestion and knowledge base management API.
"""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    """Schema for document upload metadata validation."""

    category: str = Field(..., description="e.g., Admissions, Fees & Aid, Curriculum")
    audience: str = Field(default="General Public", description="Target audience group")
    vector_backend: str = Field(default="chroma", description="Vector database backend")


class DocumentResponse(BaseModel):
    """Schema for serialized document metadata returned to clients."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    file_size: str
    chunk_count: int
    category: str
    audience: str
    status: str
    vector_backend: str
    uploaded_by_id: UUID
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated response containing total count and a sequence of document items."""

    model_config = ConfigDict(from_attributes=True)

    total: int = Field(..., description="Total number of documents matching the query")
    items: Sequence[DocumentResponse] = Field(default_factory=list, description="List of document records")
