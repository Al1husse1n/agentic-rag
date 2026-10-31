from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Token(BaseModel):
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
    sub: str | None = Field(
        default=None,
        description="Subject identifier stored inside the token payload (typically user UUID in string format).",
    )
    exp: datetime | None = Field(
        default=None,
        description="UTC datetime timestamp indicating when the token expires.",
    )


class UserResponse(BaseModel):
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

