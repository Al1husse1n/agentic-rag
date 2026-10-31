"""HITL Audit and Inquiries Inbox Test Suite.

Verifies:
1. Listing unresolved/audit logs (`GET /api/v1/audit/`):
   - Retrieval of audit inquiry records with pagination.
   - Filtering inquiries by status (e.g., `status=pending`, `status=resolved`).
   - RBAC permission enforcement (requiring `audit:read`).
2. Resolving audit inquiries (`PATCH /api/v1/audit/{audit_id}/resolve`):
   - Updating status to 'resolved' or 'escalated' with instructor notes.
   - Automatically recording the reviewing instructor's user ID.
   - 404 handling on non-existent inquiry records.
   - Request body validation (rejecting empty instructor notes).
   - RBAC permission enforcement (requiring `audit:write`).
"""

import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.audit import AuditLog

# ---------------------------------------------------------------------------
# Listing Endpoint Tests (GET /api/v1/audit/)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_audit_logs_success(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify that an instructor with `audit:read` can view all audit log records."""
    # Seed 3 audit entries
    for i in range(3):
        log = AuditLog(
            id=uuid.uuid4(),
            user_query=f"Student inquiry #{i}",
            ai_response=f"AI tentative response #{i}",
            status="pending" if i < 2 else "resolved",
            instructor_notes=None if i < 2 else "Verified manually",
        )
        test_session.add(log)
    await test_session.commit()

    response = await client.get("/api/v1/audit/", headers=instructor_user["headers"])

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["items"]) == 3
    assert data["items"][0]["user_query"].startswith("Student inquiry")


@pytest.mark.asyncio
async def test_list_audit_logs_status_filter(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify that filtering by status works correctly."""
    # Seed 2 pending, 1 resolved
    log1 = AuditLog(id=uuid.uuid4(), user_query="Pending Q1", ai_response="Ans1", status="pending")
    log2 = AuditLog(id=uuid.uuid4(), user_query="Pending Q2", ai_response="Ans2", status="pending")
    log3 = AuditLog(id=uuid.uuid4(), user_query="Resolved Q3", ai_response="Ans3", status="resolved")
    test_session.add_all([log1, log2, log3])
    await test_session.commit()

    # Query status=pending
    res_pending = await client.get("/api/v1/audit/?status=pending", headers=instructor_user["headers"])
    assert res_pending.status_code == 200
    data_pending = res_pending.json()
    assert data_pending["total"] == 2
    for item in data_pending["items"]:
        assert item["status"] == "pending"

    # Query status=resolved
    res_resolved = await client.get("/api/v1/audit/?status=resolved", headers=instructor_user["headers"])
    assert res_resolved.status_code == 200
    data_resolved = res_resolved.json()
    assert data_resolved["total"] == 1
    assert data_resolved["items"][0]["status"] == "resolved"


@pytest.mark.asyncio
async def test_list_audit_logs_pagination(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify pagination slice parameters (skip, limit)."""
    for i in range(4):
        test_session.add(
            AuditLog(
                id=uuid.uuid4(),
                user_query=f"Query {i}",
                ai_response=f"Resp {i}",
                status="pending",
            )
        )
    await test_session.commit()

    res = await client.get("/api/v1/audit/?skip=0&limit=2", headers=instructor_user["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 4
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_list_audit_logs_permission_denied_for_student(
    client: httpx.AsyncClient,
    student_user: dict[str, Any],
):
    """Verify that a user lacking `audit:read` (like student) receives 403 Forbidden."""
    response = await client.get("/api/v1/audit/", headers=student_user["headers"])

    assert response.status_code == 403
    assert "missing required permission" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_audit_logs_unauthenticated(client: httpx.AsyncClient):
    """Verify unauthenticated requests cannot view audit logs (401)."""
    response = await client.get("/api/v1/audit/")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Resolve Endpoint Tests (PATCH /api/v1/audit/{audit_id}/resolve)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_audit_log_success(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify that an instructor with `audit:write` can resolve an inquiry with notes."""
    audit_id = uuid.uuid4()
    log = AuditLog(
        id=audit_id,
        user_query="What are the admission requirements?",
        ai_response="I could not find the exact admission GPA requirement.",
        status="pending",
        instructor_notes=None,
    )
    test_session.add(log)
    await test_session.commit()

    resolution_notes = "Added minimum GPA criteria (3.0) to Admissions KB."
    response = await client.patch(
        f"/api/v1/audit/{audit_id}/resolve",
        headers=instructor_user["headers"],
        json={
            "instructor_notes": resolution_notes,
            "status": "resolved",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(audit_id)
    assert data["status"] == "resolved"
    assert data["instructor_notes"] == resolution_notes
    assert data["reviewed_by_id"] == str(instructor_user["id"])

    # Verify direct database persistence (expire session cache to reload from DB)
    test_session.expire_all()
    stmt = select(AuditLog).where(AuditLog.id == audit_id)
    result = await test_session.execute(stmt)
    db_item = result.scalars().first()
    assert db_item is not None
    assert db_item.status == "resolved"
    assert db_item.instructor_notes == resolution_notes
    assert db_item.reviewed_by_id == instructor_user["id"]


@pytest.mark.asyncio
async def test_resolve_audit_log_escalate(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify updating status to 'escalated'."""
    audit_id = uuid.uuid4()
    log = AuditLog(
        id=audit_id,
        user_query="Complex financial aid appeal inquiry",
        ai_response="Unknown policy",
        status="pending",
    )
    test_session.add(log)
    await test_session.commit()

    response = await client.patch(
        f"/api/v1/audit/{audit_id}/resolve",
        headers=instructor_user["headers"],
        json={
            "instructor_notes": "Forwarded to dean for policy clarification.",
            "status": "escalated",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "escalated"


@pytest.mark.asyncio
async def test_resolve_audit_log_not_found(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify resolving a non-existent audit UUID returns 404 Not Found."""
    random_id = uuid.uuid4()
    response = await client.patch(
        f"/api/v1/audit/{random_id}/resolve",
        headers=instructor_user["headers"],
        json={
            "instructor_notes": "Some notes",
            "status": "resolved",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_resolve_audit_log_permission_denied_for_student(
    client: httpx.AsyncClient,
    student_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify that a user lacking `audit:write` permission is denied (403)."""
    audit_id = uuid.uuid4()
    test_session.add(
        AuditLog(
            id=audit_id,
            user_query="Sample question",
            ai_response="Sample answer",
            status="pending",
        )
    )
    await test_session.commit()

    response = await client.patch(
        f"/api/v1/audit/{audit_id}/resolve",
        headers=student_user["headers"],
        json={"instructor_notes": "Trying to resolve without permission", "status": "resolved"},
    )

    assert response.status_code == 403
    assert "missing required permission" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_resolve_audit_log_empty_notes_rejected(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify empty instructor notes fail validation with 422 Unprocessable Entity."""
    audit_id = uuid.uuid4()
    response = await client.patch(
        f"/api/v1/audit/{audit_id}/resolve",
        headers=instructor_user["headers"],
        json={"instructor_notes": "", "status": "resolved"},
    )

    assert response.status_code == 422
