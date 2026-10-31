"""Schemas Package.

Centralized Pydantic v2 data models for input validation, API request contracts,
and serialized output responses across all API endpoints.
"""

from src.schemas.auth import (
    LoginRequest,
    Token,
    TokenData,
    UserResponse,
)
from src.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
)

__all__ = [
    "DocumentCreate",
    "DocumentListResponse",
    "DocumentResponse",
    "LoginRequest",
    "Token",
    "TokenData",
    "UserResponse",
]
