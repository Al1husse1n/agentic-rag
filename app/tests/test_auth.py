"""Authentication and Security Layer Test Suite.

Verifies:
1. Password hashing with Argon2id and constant-time verification.
2. OAuth2 password flow login endpoint (`POST /api/v1/auth/login`):
   - Successful authentication with Bearer token issuance.
   - Rejection of invalid passwords and nonexistent emails (401).
   - Rejection of empty/whitespace input (400/422).
   - Rejection of deactivated accounts (403).
3. Authenticated profile endpoint (`GET /api/v1/auth/me`):
   - Retrieval of user details with eager-loaded roles and permissions.
   - Rejection of missing, malformed, or expired tokens (401).
   - Rejection of tokens belonging to deactivated accounts (403).
"""

import uuid
from datetime import timedelta
from typing import Any

import httpx
import pytest
from src.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

# ---------------------------------------------------------------------------
# Unit Tests: Cryptographic Primitives & JWT Lifecycle
# ---------------------------------------------------------------------------

def test_hash_and_verify_password():
    """Verify that hash_password uses Argon2id and verify_password checks correctly."""
    plain_password = "SecurePassword123!"
    hashed = hash_password(plain_password)

    assert hashed != plain_password
    assert hashed.startswith("$argon2id$")
    assert verify_password(plain_password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_jwt_encode_and_decode_cycle():
    """Verify that create_access_token and decode_access_token maintain claim integrity."""
    subject_id = uuid.uuid4()
    token = create_access_token(subject=subject_id)

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == str(subject_id)
    assert "exp" in payload
    assert "iat" in payload


def test_jwt_decode_expired_token():
    """Verify that decode_access_token safely returns None for an expired token."""
    subject_id = uuid.uuid4()
    expired_token = create_access_token(
        subject=subject_id,
        expires_delta=timedelta(minutes=-10),
    )

    payload = decode_access_token(expired_token)
    assert payload is None


def test_jwt_decode_malformed_token():
    """Verify that decode_access_token rejects tampered or malformed token strings."""
    assert decode_access_token("not-a-valid-jwt") is None
    assert decode_access_token("eyJhbGciOiJIUzI1NiJ9.invalid.signature") is None


# ---------------------------------------------------------------------------
# Integration Tests: Login Endpoint (POST /api/v1/auth/login)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_success(client: httpx.AsyncClient, instructor_user: dict[str, Any]):
    """Verify valid credentials receive a 200 OK with an RFC 6749 compliant Bearer token."""
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": instructor_user["email"],
            "password": instructor_user["password"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0

    # Ensure returned token decodes back to the instructor user's UUID
    payload = decode_access_token(data["access_token"])
    assert payload is not None
    assert payload["sub"] == str(instructor_user["id"])


@pytest.mark.asyncio
async def test_login_case_insensitive_email(client: httpx.AsyncClient, instructor_user: dict[str, Any]):
    """Verify login email normalization handles mixed case without failure."""
    mixed_case_email = instructor_user["email"].upper()
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": mixed_case_email,
            "password": instructor_user["password"],
        },
    )

    assert response.status_code == 200
    assert "access_token" in response.json()


@pytest.mark.asyncio
async def test_login_incorrect_password(client: httpx.AsyncClient, instructor_user: dict[str, Any]):
    """Verify bad password returns 401 Unauthorized with WWW-Authenticate header."""
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": instructor_user["email"],
            "password": "IncorrectPassword999!",
        },
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert "Incorrect email address or password" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_nonexistent_email(client: httpx.AsyncClient):
    """Verify nonexistent user returns 401 Unauthorized without disclosing account presence."""
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "ghost@university.edu",
            "password": "AnyPassword123!",
        },
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert "Incorrect email address or password" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_whitespace_email(client: httpx.AsyncClient):
    """Verify empty or all-whitespace email inputs are rejected with 400 Bad Request."""
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "   ",
            "password": "Password123!",
        },
    )

    assert response.status_code == 400
    assert "Email address and password must not be empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_deactivated_account(client: httpx.AsyncClient, inactive_user: dict[str, Any]):
    """Verify deactivated user credentials trigger a 403 Forbidden error."""
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": inactive_user["email"],
            "password": inactive_user["password"],
        },
    )

    assert response.status_code == 403
    assert "deactivated" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Integration Tests: Authenticated Profile Endpoint (GET /api/v1/auth/me)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_me_success(client: httpx.AsyncClient, instructor_user: dict[str, Any]):
    """Verify /me returns complete profile with eager-loaded roles and permissions."""
    response = await client.get("/api/v1/auth/me", headers=instructor_user["headers"])

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(instructor_user["id"])
    assert data["email_address"] == instructor_user["email"]
    assert data["is_active"] is True

    # Validate RBAC hydration
    assert "instructor" in data["roles"]
    assert "admin" in data["roles"]
    assert "documents:read" in data["permissions"]
    assert "documents:write" in data["permissions"]
    assert "audit:read" in data["permissions"]
    assert "audit:write" in data["permissions"]


@pytest.mark.asyncio
async def test_me_unauthenticated(client: httpx.AsyncClient):
    """Verify unauthenticated requests to /me are rejected with 401 Unauthorized."""
    response = await client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_me_malformed_token(client: httpx.AsyncClient):
    """Verify request with malformed JWT is rejected with 401 Unauthorized."""
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid.jwt.signature"},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_me_expired_token(client: httpx.AsyncClient, instructor_user: dict[str, Any]):
    """Verify request with expired JWT is rejected with 401 Unauthorized."""
    expired_token = create_access_token(
        subject=instructor_user["id"],
        expires_delta=timedelta(minutes=-30),
    )

    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )

    assert response.status_code == 401
    assert "invalid, expired, or malformed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_me_deactivated_user_token(client: httpx.AsyncClient, inactive_user: dict[str, Any]):
    """Verify token belonging to a deactivated user yields 403 Forbidden even if JWT is valid."""
    response = await client.get("/api/v1/auth/me", headers=inactive_user["headers"])

    assert response.status_code == 403
    assert "deactivated" in response.json()["detail"].lower()
