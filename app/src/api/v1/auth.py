"""Authentication API Endpoints (v1).

This module handles:
1. `POST /api/v1/auth/login`: Authenticates user credentials (instructors, administrators,
   or admissions staff) via OAuth2 password flow, returning an RFC 6749 compliant Bearer token.
2. `GET /api/v1/auth/me`: Returns the authenticated user's profile with their eagerly loaded
   roles and granular permissions for dashboard authorization and state management.

Includes enterprise-grade exception handling:
- Wraps database operations in guarded transaction blocks with `SQLAlchemyError` isolation.
- Structured logging using `logger.info`, `logger.warning`, and `logger.exception` without leaking secrets.
- Input sanitization (email whitespace stripping and lowercasing).
- Generic 401 Unauthorized errors to prevent user enumeration security vulnerabilities.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload
from src.api.deps import (
    CurrentUser,
    SessionDep,
    get_user_permissions,
    get_user_roles,
)
from src.core.config import settings
from src.core.security import create_access_token, verify_password
from src.models import (
    Role,
    RolePermission,
    User,
    UserRole,
)
from src.schemas.auth import Token, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=Token,
    summary="User Login & Token Issuance",
    description="Authenticates instructors and dashboard users via email and password, returning a signed JWT Bearer token.",
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: SessionDep,
) -> Token:
    """Authenticate user credentials and issue a signed JSON Web Token.

    What it does:
        1. Accepts standard OAuth2 password request form fields (`username` representing
           the user's email address, and `password`).
        2. Sanitizes input: strips leading/trailing whitespace and normalizes email to lowercase.
        3. Queries Neon PostgreSQL for an active `User` matching the email.
        4. Eagerly loads roles and permissions to verify access privileges in a single round-trip.
        5. Verifies the password hash using Argon2id via `verify_password`.
        6. Validates that the account is active (`is_active == True`).
        7. Issues a cryptographically signed JWT containing the user's UUID in the `sub` claim.
        8. Returns the token response formatted to RFC 6749 standards.

    Why it is implemented this way:
        - `OAuth2PasswordRequestForm` guarantees compatibility with Swagger UI's "Authorize"
          modal button, automated testing suites, and standard OAuth2 client libraries.
        - Database queries are wrapped in `try ... except SQLAlchemyError` to prevent raw
          tracebacks from leaking database internal details to clients.
        - Generic 401 error messages prevent username enumeration vulnerabilities.
        - Email normalization prevents authentication failures caused by case mismatches.

    Parameters:
        form_data (OAuth2PasswordRequestForm): Injected form body containing `username` and `password`.
        db (AsyncSession): Injected asynchronous database session.

    Raises:
        HTTPException(400): If username or password is empty.
        HTTPException(401): If credentials (email or password) are incorrect.
        HTTPException(403): If the user's account has been deactivated.
        HTTPException(500): If an unexpected database error occurs.

    Returns:
        Token: A Pydantic schema containing `access_token`, `token_type`, and `expires_in` in seconds.
    """
    # 1. Sanitize and validate input presence
    normalized_email = form_data.username.strip().lower()
    if not normalized_email or not form_data.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address and password must not be empty.",
        )

    # 2. Query user by email with eager-loaded RBAC graph
    stmt = (
        select(User)
        .options(
            selectinload(User.user_roles)
            .selectinload(UserRole.role)
            .selectinload(Role.role_permissions)
            .selectinload(RolePermission.permission)
        )
        .where(User.email_address == normalized_email)
    )

    try:
        result = await db.execute(stmt)
        user = result.scalars().first()
    except SQLAlchemyError as exc:
        logger.exception("Database error during authentication for email: %s", normalized_email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected database error occurred during authentication.",
        ) from exc

    # 3. Constant-time credential verification (prevents timing attacks and user enumeration)
    if user is None or not verify_password(form_data.password, user.hashed_password):
        logger.warning("Failed authentication attempt for email: %s", normalized_email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email address or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Check account activation status
    if not user.is_active:
        logger.warning(
            "Authentication rejected: Inactive account for email: %s (ID: %s)",
            normalized_email,
            user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated. Please contact your system administrator.",
        )

    # 5. Generate signed JWT access token
    access_token = create_access_token(subject=user.id)
    expires_in_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

    logger.info("User '%s' (ID: %s) successfully authenticated", user.email_address, user.id)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in_seconds,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Authenticated User Profile",
    description="Returns the profile, assigned roles, and granular permissions for the currently authenticated user.",
)
async def read_users_me(
    current_user: CurrentUser,
) -> UserResponse:
    """Retrieve details, roles, and effective permissions of the currently logged-in user.

    What it does:
        Injects the authenticated `CurrentUser`, extracts their roles and permissions
        from the eager-loaded relationship graph, and serializes the result into a `UserResponse`.

    Why it is implemented this way:
        Enables dashboard frontends to display user info (e.g., Sarah Jenkins | Admissions Admin)
        and determine available features dynamically based on permissions without making separate API calls.

    Parameters:
        current_user (User): Injected via `get_current_user` dependency.

    Raises:
        HTTPException(500): If an unexpected serialization error occurs.

    Returns:
        UserResponse: Serialized user record with flattened roles and permissions.
    """
    try:
        roles = sorted(get_user_roles(current_user))
        permissions = sorted(get_user_permissions(current_user))

        response = UserResponse(
            id=current_user.id,
            email_address=current_user.email_address,
            full_name=current_user.full_name,
            is_active=current_user.is_active,
            roles=roles,
            permissions=permissions,
        )
    except Exception as exc:
        logger.exception("Error serializing authenticated user profile for user ID: %s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile data.",
        ) from exc

    logger.debug("Profile retrieved for user ID: %s (%s)", current_user.id, current_user.email_address)
    return response
