# What it powers: The right-hand column (Unresolved Visitor Inquiries) showing flagged queries, semantic match percentages, context gaps, and action buttons like "Draft & Add to KB" or "Escalate".
from __future__ import annotations
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from src.core.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    # The low-confidence or failed AI response generated
    ai_response: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Resolution Status: "pending", "resolved", "escalated"
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False, index=True)
    
    # Instructor notes or corrected answer provided during review
    instructor_notes: Mapped[str|None] = mapped_column(Text)

    # Who triggered or asked the initial question (Optional if anonymous/public student)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL")
    )

    reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    author: Mapped[User | None] = relationship(foreign_keys=[user_id])  # type: ignore[name-defined]
    reviewer: Mapped[User | None] = relationship(foreign_keys=[reviewed_by_id])  # type: ignore[name-defined]