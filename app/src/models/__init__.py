"""Database Models Package.

This package provides centralized exports for all SQLAlchemy ORM models,
including core users, roles, permissions, and association tables for the
normalized Role-Based Access Control (RBAC) system.
"""

from src.models.user import (
    Permission,
    Role,
    RolePermission,
    User,
    UserRole,
)

__all__ = [
    "Permission",
    "Role",
    "RolePermission",
    "User",
    "UserRole",
]
