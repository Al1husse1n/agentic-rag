# What it powers: The bottom-left profile card (Sarah Jenkins | Admissions Admin) and access control
# Fields: id, email, hashed_password, name, role, avatar_url.

from __future__ import annotations
import uuid
from typing import Optional
from decimal import Decimal
from datetime import UTC, datetime, date
from enum import Enum

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, Numeric, Boolean, Enum as SQLAlchemyEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from sqlalchemy import (
    DateTime, Date, ForeignKey, Integer, String,
    UniqueConstraint, Numeric, Boolean, Text, BigInteger,
    Enum as SQLAlchemyEnum, CheckConstraint
)

from src.core.database import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    email_address: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False) 
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

    created_by: Mapped[uuid.UUID | None] = mapped_column(       #remove |None once you created the first admin
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True                   #change to False once you created the first admin
    )
    updated_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )

    # relationships
    creator: Mapped[User] = relationship(
        foreign_keys=[created_by],
        remote_side=[id],            
        back_populates="created_users"
    )
    updater: Mapped[User] = relationship(
        foreign_keys=[updated_by],
        remote_side=[id],
        back_populates="updated_users"
    )
    created_users: Mapped[list[User]] = relationship(
        foreign_keys=[created_by],
        back_populates="creator"
    )
    updated_users: Mapped[list[User]] = relationship(
        foreign_keys=[updated_by],
        back_populates="updater"
    )
    user_roles: Mapped[list[UserRole]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    ) 
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)               # | None → affects both the Python type (type checker knows it can be None) and the DB column (NULL allowed). nullable=True is for only db null

    # relationships
    user_roles: Mapped[list[UserRole]] = relationship(
        back_populates="role",
        cascade="all, delete-orphan"
    )
    role_permissions: Mapped[list[RolePermission]] = relationship(
        back_populates="role",
        cascade="all, delete-orphan"
    )


class Permission(Base):
    __tablename__ = "permissions"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True) 
    description: Mapped[str | None] = mapped_column(Text)

    #relaionships
    role_permissions: Mapped[list[RolePermission]] = relationship(
        back_populates="permission",
        cascade="all, delete-orphan"
    )


class UserRole(Base):
    __tablename__ = "user_roles"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign Keys
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id"),
        nullable=False
    )


    # relationships
    user: Mapped[User] = relationship(
        foreign_keys=[user_id],
        back_populates="user_roles"
    )
    role: Mapped[Role] = relationship(
        foreign_keys=[role_id],
        back_populates="user_roles"
    )


class RolePermission(Base):
    __tablename__ = "role_permissions"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign keys
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id"),
        nullable=False
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("permissions.id"),
        nullable=False
    )

    #relationships
    role: Mapped[Role] = relationship(
        foreign_keys=[role_id],
        back_populates="role_permissions"
    )
    permission: Mapped[Permission] = relationship(
        foreign_keys=[permission_id],
        back_populates="role_permissions"
    )




