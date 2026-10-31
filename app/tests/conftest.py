"""Central Pytest Fixtures and Async Database Setup.

This module provides:
1. An isolated, in-memory SQLite asynchronous engine (`sqlite+aiosqlite:///:memory:`)
   utilizing `StaticPool` to preserve state across async connections per test.
2. FastAPI `app.dependency_overrides` for `get_db` injection.
3. An asynchronous HTTP test client (`httpx.AsyncClient` with `ASGITransport`).
4. Pre-seeded test users:
   - `instructor_user`: An active administrator/instructor with full permissions (`documents:read`,
     `documents:write`, `audit:read`, `audit:write`).
   - `student_user`: An active user with restricted read-only permissions (`documents:read`).
   - `inactive_user`: A deactivated user account (`is_active=False`) for testing security locks.
5. Automatic isolation of the file upload directory using `tmp_path`.
"""

import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import src.api.v1.document as document_module
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool
from src.core.database import Base, get_db
from src.core.main import app
from src.core.security import create_access_token, hash_password
from src.models import (
    Permission,
    Role,
    RolePermission,
    User,
    UserRole,
)

# ---------------------------------------------------------------------------
# Database & Client Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="function")
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an isolated, in-memory SQLite async engine for each test function."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a direct SQLAlchemy AsyncSession for test assertions and setup."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def client(test_engine: AsyncEngine) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an asynchronous HTTP client configured with test database dependency overrides."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# File System Isolation Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_upload_directory(tmp_path: Path) -> Any:
    """Redirect document storage to a temporary directory for each test."""
    original_upload_dir = document_module.UPLOAD_DIR
    test_upload_dir = tmp_path / "test_pdfs"
    test_upload_dir.mkdir(parents=True, exist_ok=True)

    document_module.UPLOAD_DIR = test_upload_dir
    yield test_upload_dir
    document_module.UPLOAD_DIR = original_upload_dir


# ---------------------------------------------------------------------------
# Pre-seeded RBAC User Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="function")
async def instructor_user(test_session: AsyncSession) -> dict[str, Any]:
    """Seed and return an active Instructor/Admin with full RBAC permissions."""
    raw_password = "Password123!"

    # 1. Create permissions
    perms = [
        Permission(id=uuid.uuid4(), name="documents:read", description="Read knowledge documents"),
        Permission(id=uuid.uuid4(), name="documents:write", description="Upload knowledge documents"),
        Permission(id=uuid.uuid4(), name="audit:read", description="View HITL audit inquiries"),
        Permission(id=uuid.uuid4(), name="audit:write", description="Resolve HITL audit inquiries"),
    ]
    test_session.add_all(perms)

    # 2. Create roles and role permissions
    role_instructor = Role(id=uuid.uuid4(), name="instructor", description="Course instructor")
    role_admin = Role(id=uuid.uuid4(), name="admin", description="System administrator")
    test_session.add_all([role_instructor, role_admin])

    for perm in perms:
        test_session.add(RolePermission(id=uuid.uuid4(), role=role_instructor, permission=perm))
        test_session.add(RolePermission(id=uuid.uuid4(), role=role_admin, permission=perm))

    # 3. Create user
    user = User(
        id=uuid.uuid4(),
        email_address="instructor@university.edu",
        hashed_password=hash_password(raw_password),
        full_name="Dr. Sarah Jenkins",
        is_active=True,
    )
    test_session.add(user)

    test_session.add(UserRole(id=uuid.uuid4(), user=user, role=role_instructor))
    test_session.add(UserRole(id=uuid.uuid4(), user=user, role=role_admin))

    await test_session.commit()
    await test_session.refresh(user)

    token = create_access_token(subject=user.id)
    return {
        "user": user,
        "id": user.id,
        "email": user.email_address,
        "password": raw_password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest_asyncio.fixture(scope="function")
async def student_user(test_session: AsyncSession) -> dict[str, Any]:
    """Seed and return an active Student user with restricted read-only permissions."""
    raw_password = "Password123!"

    perm_read = Permission(id=uuid.uuid4(), name="documents:read", description="Read knowledge documents")
    test_session.add(perm_read)

    role_student = Role(id=uuid.uuid4(), name="student", description="Regular student")
    test_session.add(role_student)

    test_session.add(RolePermission(id=uuid.uuid4(), role=role_student, permission=perm_read))

    user = User(
        id=uuid.uuid4(),
        email_address="student@university.edu",
        hashed_password=hash_password(raw_password),
        full_name="Alex Student",
        is_active=True,
    )
    test_session.add(user)
    test_session.add(UserRole(id=uuid.uuid4(), user=user, role=role_student))

    await test_session.commit()
    await test_session.refresh(user)

    token = create_access_token(subject=user.id)
    return {
        "user": user,
        "id": user.id,
        "email": user.email_address,
        "password": raw_password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest_asyncio.fixture(scope="function")
async def inactive_user(test_session: AsyncSession) -> dict[str, Any]:
    """Seed and return a deactivated user account (`is_active=False`)."""
    raw_password = "Password123!"

    user = User(
        id=uuid.uuid4(),
        email_address="deactivated@university.edu",
        hashed_password=hash_password(raw_password),
        full_name="Locked Account",
        is_active=False,
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    token = create_access_token(subject=user.id)
    return {
        "user": user,
        "id": user.id,
        "email": user.email_address,
        "password": raw_password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }
