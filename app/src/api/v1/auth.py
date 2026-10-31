
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
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

    stmt = (
        select(User)
        .options(
            selectinload(User.user_roles)
            .selectinload(UserRole.role)
            .selectinload(Role.role_permissions)
            .selectinload(RolePermission.permission)
        )
        .where(User.email_address == form_data.username)
    )
    result = await db.execute(stmt)
    user = result.scalars().first()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email address or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated. Please contact your system administrator.",
        )

    access_token = create_access_token(subject=user.id)
    expires_in_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

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

    roles = sorted(get_user_roles(current_user))
    permissions = sorted(get_user_permissions(current_user))

    return UserResponse(
        id=current_user.id,
        email_address=current_user.email_address,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        roles=roles,
        permissions=permissions,
    )
