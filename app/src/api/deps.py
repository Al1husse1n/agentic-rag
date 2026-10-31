"""FastAPI Dependency Injection and Security Guards Module.
"""
from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.core.database import get_db
from src.core.security import decode_access_token
from src.models import (
    Role,
    RolePermission,
    User,
    UserRole,
)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    description="JWT Bearer token authentication header using the RFC 6750 standard.",
)

# Type alias for injected async database sessions
SessionDep = Annotated[AsyncSession, Depends(get_db)]

def get_user_roles(user: User) -> set[str]:
    if not user.user_roles:
        return set()
    return {ur.role.name for ur in user.user_roles if ur.role is not None}


def get_user_permissions(user: User) -> set[str]:     
    permissions: set[str] = set()
    if not user.user_roles:
        return permissions

    for ur in user.user_roles:
        if ur.role and ur.role.role_permissions:
            for rp in ur.role.role_permissions:
                if rp.permission and rp.permission.name:
                    permissions.add(rp.permission.name)

    return permissions


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: SessionDep,
) -> User:

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials: Token is invalid, expired, or malformed.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    subject: str | None = payload.get("sub")
    if not subject:
        raise credentials_exception

    try:
        user_uuid = UUID(subject)
    except (TypeError, ValueError):
        raise credentials_exception

    stmt = (
        select(User)
        .options(
            selectinload(User.user_roles)
            .selectinload(UserRole.role)
            .selectinload(Role.role_permissions)
            .selectinload(RolePermission.permission)
        )
        .where(User.id == user_uuid)
    )

    result = await db.execute(stmt)
    user = result.scalars().first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user record was not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )


    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated. Contact an administrator.",
        )

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]



def require_role(required_role: str) -> Callable[[User], User]:
    def role_checker(current_user: CurrentUser) -> User:
        roles = get_user_roles(current_user)
        if required_role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: This action requires the '{required_role}' role.",
            )
        return current_user

    return role_checker


def require_permission(required_permission: str) -> Callable[[User], User]:

    def permission_checker(current_user: CurrentUser) -> User:
        permissions = get_user_permissions(current_user)
        if required_permission not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Missing required permission '{required_permission}'.",
            )
        return current_user

    return permission_checker
