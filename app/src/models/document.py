# src/models/document.py
from __future__ import annotations
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from src.core.database import Base


class DocumentSource(Base):
    __tablename__ = "document_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)  # Local path or S3/MinIO URI
    file_size: Mapped[str] = mapped_column(String(50), nullable=False)   # e.g., "4.2 MB"
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Dashboard Categorization & Target Audience
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # e.g., "Admissions", "Fees & Aid"
    audience: Mapped[str] = mapped_column(String(100), default="General Public", nullable=False)  # e.g., "Prospective Families, ALl"
    
    # Pipeline State Tracking
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False, index=True)  # "pending", "vectorizing", "indexed", "failed"
    vector_backend: Mapped[str] = mapped_column(String(50), default="chroma", nullable=False)  # "chroma", "pinecone", or "pgvector"

    #foreign keys
    uploaded_by_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False
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

    # Relationships
    uploader: Mapped[User] = relationship(foreign_keys=[uploaded_by_id])  # type: ignore[name-defined]