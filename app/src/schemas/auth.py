"""Authentication and Authorization Pydantic Schemas.

This module defines the validation schemas and serialization contracts for:
1. Direct credentials and OAuth2 Bearer token responses returned by authentication endpoints.
2. Internal JWT payload validation structures.
3. Authenticated user profile representation with flattened RBAC roles and permissions.

All schemas follow Pydantic v2 best practices.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    """Schema for JSON-based credential authentication.

    Used when clients submit authentication credentials via a JSON body
    instead of multipart or URL-encoded form data.
    """

    email_address: EmailStr = Field(
        ...,
        description="User's registered email address.",
        examples=["sarah.jenkins@university.edu"],
    )
    password: str = Field(
        ...,
        min_length=1,
        description="Plaintext password for account authentication.",
        examples=["P@ssw0rd123!"],
    )


class Token(BaseModel):
    """Schema representing the standard OAuth2 Bearer token response.

    Returned to the client after a successful authentication attempt.
    Complies with RFC 6749 Section 5.1 (Successful Response).
    """

    access_token: str = Field(
        ...,
        description="The encoded JSON Web Token (JWT) string used to authenticate subsequent requests.",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )
    token_type: str = Field(
        default="bearer",
        description="Type of token issued; always 'bearer' for OAuth2 Bearer authentication.",
        examples=["bearer"],
    )
    expires_in: int = Field(
        ...,
        description="Token lifetime duration in seconds from the moment of issue.",
        examples=[1800],
    )


class TokenData(BaseModel):
    """Internal schema for validating decoded JWT claim payloads.

    Used by token verification routines in `src.core.security` and `src.api.deps`
    to guarantee that extracted tokens contain a valid subject identity and expiration.
    """

    sub: str | None = Field(
        default=None,
        description="Subject identifier stored inside the token payload (typically user UUID in string format).",
    )
    exp: datetime | None = Field(
        default=None,
        description="UTC datetime timestamp indicating when the token expires.",
    )


class UserResponse(BaseModel):
    """Public serialized representation of an authenticated user.

    Provides frontend clients with essential user profile data alongside
    flattened lists of assigned roles and granular permissions for client-side
    feature gating and UI rendering.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(
        ...,
        description="Unique identifier (UUID v4) for the user record.",
    )
    email_address: EmailStr = Field(
        ...,
        description="User's primary email address used for login.",
        examples=["sarah.jenkins@university.edu"],
    )
    full_name: str = Field(
        ...,
        description="Full display name of the user.",
        examples=["Sarah Jenkins"],
    )
    is_active: bool = Field(
        ...,
        description="Whether the user account is active and permitted to authenticate.",
    )
    roles: list[str] = Field(
        default_factory=list,
        description="List of assigned role names (e.g., ['admin', 'instructor']).",
        examples=[["admin", "instructor"]],
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="Flattened list of all granular permissions granted across all assigned roles.",
        examples=[["document:read", "document:write", "rag:query"]],
    )
