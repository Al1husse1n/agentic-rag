# src/schemas/audit.py
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class AuditLogResponse(BaseModel):
    id: UUID
    user_query: str
    ai_response: str
    status: str
    instructor_notes: str | None
    user_id: UUID | None
    reviewed_by_id: UUID | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    total: int
    items: list[AuditLogResponse]


class AuditResolveRequest(BaseModel):
    instructor_notes: str = Field(..., min_length=1, description="Resolution notes or correct answer provided by instructor")
    status: str = Field(default="resolved", description="New status: 'resolved' or 'escalated'")    