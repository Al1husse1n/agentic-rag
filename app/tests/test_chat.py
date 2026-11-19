"""Tests for Chat and SSE Streaming Endpoint with HITL Quality-Control Audit Logging.

Verifies:
1. `log_audit_background`:
   - Accepts `confidence_score` and `retrieved_sources`.
   - Persists `AuditLog` into database with non-null audit tracking values.
2. `POST /api/v1/chat/stream` (SSE streaming):
   - Returns `text/event-stream` media type.
   - Streams start, status, token, and done SSE events.
   - Triggers `log_audit_background` and emits `audit_escalation` event when confidence is below threshold.
   - Does NOT trigger audit escalation when confidence meets/exceeds threshold.
3. `POST /api/v1/chat/query` (Synchronous):
   - Returns structured `ChatQueryResponse`.
   - Enforces threshold escalation and returns `audit_id`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.chat import log_audit_background
from src.models.audit import AuditLog


# ---------------------------------------------------------------------------
# Background Audit Logging Service Unit Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_audit_background_persists_confidence_and_sources(
    test_session: AsyncSession,
):
    """Verify that log_audit_background persists confidence_score and retrieved_sources."""
    audit_id = uuid.uuid4()
    query = "What are the admission requirements for graduate engineering?"
    response = "Applicants require a minimum GPA of 3.2 and GRE scores."
    confidence = 0.52
    sources = [
        {
            "filename": "grad_admissions_guide.pdf",
            "content": "Engineering graduate requirements: minimum GPA 3.2.",
            "sub_question": "What is the minimum GPA?",
            "metadata": {"page": 4},
        }
    ]

    # Invoke log_audit_background directly
    await log_audit_background(
        user_query=query,
        ai_response=response,
        confidence_score=confidence,
        retrieved_sources=sources,
        user_id=None,
        status="pending",
        audit_id=audit_id,
    )

    # Query Neon / DB session to verify record
    stmt = select(AuditLog).where(AuditLog.id == audit_id)
    result = await test_session.execute(stmt)
    record = result.scalars().first()

    assert record is not None
    assert record.user_query == query
    assert record.ai_response == response
    assert record.confidence_score == pytest.approx(0.52, abs=1e-3)
    assert record.retrieved_sources == sources
    assert record.status == "pending"
    assert record.user_id is None


@pytest.mark.asyncio
async def test_log_audit_background_with_authenticated_user(
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify log_audit_background properly attaches user_id when user is authenticated."""
    audit_id = uuid.uuid4()
    user_id = instructor_user["id"]

    await log_audit_background(
        user_query="Can I take an incomplete grade?",
        ai_response="Incompletes must be resolved within one semester.",
        confidence_score=0.45,
        retrieved_sources=[{"content": "Incomplete policy handbook chunk"}],
        user_id=user_id,
        status="pending",
        audit_id=audit_id,
    )

    stmt = select(AuditLog).where(AuditLog.id == audit_id)
    result = await test_session.execute(stmt)
    record = result.scalars().first()

    assert record is not None
    assert record.user_id == user_id
    assert record.confidence_score == 0.45
    assert len(record.retrieved_sources) == 1


# ---------------------------------------------------------------------------
# SSE Streaming Endpoint Tests (POST /api/v1/chat/stream)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_stream_low_confidence_triggers_audit_and_sse_events(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify SSE streaming emits audit escalation and dispatches log_audit_background on low confidence."""
    mock_sources = [
        {"filename": "academic_calendar.pdf", "content": "Classes begin Sept 1."}
    ]
    mock_answer = "Classes start around September according to the calendar."
    mock_confidence = 0.48  # Below default 0.70 threshold

    with patch(
        "src.api.v1.chat.run_agent_workflow",
        new=AsyncMock(return_value=(mock_answer, mock_confidence, mock_sources)),
    ):
        response = await client.post(
            "/api/v1/chat/stream",
            json={
                "query": "When do fall semester classes start?",
                "confidence_threshold": 0.70,
            },
            headers=instructor_user["headers"],
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse SSE payload
    raw_lines = response.text.split("\n")
    events: list[dict[str, Any]] = []
    for line in raw_lines:
        line_clean = line.strip()
        if line_clean.startswith("data: "):
            events.append(json.loads(line_clean[6:]))

    # Check event types received
    event_types = [e["type"] for e in events]
    assert "start" in event_types
    assert "status" in event_types
    assert "sources" in event_types
    assert "token" in event_types
    assert "audit_escalation" in event_types
    assert "done" in event_types

    # Verify audit escalation payload
    escalation_event = next(e for e in events if e["type"] == "audit_escalation")
    assert escalation_event["audit_id"] is not None
    assert escalation_event["escalated"] is True
    assert "not completely certain" in escalation_event["message"]

    # Verify done event
    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["escalated_to_audit"] is True
    assert done_event["disclaimer"] is not None
    assert "not completely certain" in done_event["disclaimer"]
    assert done_event["audit_id"] == escalation_event["audit_id"]

    # Verify the AuditLog entry was persisted in the database
    audit_uuid = uuid.UUID(escalation_event["audit_id"])
    stmt = select(AuditLog).where(AuditLog.id == audit_uuid)
    res = await test_session.execute(stmt)
    persisted = res.scalars().first()

    assert persisted is not None
    assert persisted.confidence_score == pytest.approx(0.48, abs=1e-3)
    assert persisted.retrieved_sources == mock_sources
    assert persisted.user_id == instructor_user["id"]


@pytest.mark.asyncio
async def test_chat_stream_high_confidence_no_audit(
    client: httpx.AsyncClient,
    test_session: AsyncSession,
):
    """Verify SSE streaming does NOT emit audit escalation when confidence meets threshold."""
    mock_answer = "Tuition is due on August 15."
    mock_confidence = 0.95  # Exceeds 0.70 threshold

    with patch(
        "src.api.v1.chat.run_agent_workflow",
        new=AsyncMock(return_value=(mock_answer, mock_confidence, [])),
    ):
        response = await client.post(
            "/api/v1/chat/stream",
            json={
                "query": "When is tuition due?",
                "confidence_threshold": 0.70,
            },
        )

    assert response.status_code == 200

    raw_lines = response.text.split("\n")
    events = [
        json.loads(line.strip()[6:])
        for line in raw_lines
        if line.strip().startswith("data: ")
    ]

    event_types = [e["type"] for e in events]
    assert "audit_escalation" not in event_types

    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["escalated_to_audit"] is False
    assert done_event["audit_id"] is None
    assert done_event.get("disclaimer") is None


# ---------------------------------------------------------------------------
# Synchronous Chat Query Endpoint Tests (POST /api/v1/chat/query)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_query_synchronous_audit_escalation(
    client: httpx.AsyncClient,
    test_session: AsyncSession,
):
    """Verify POST /api/v1/chat/query escalates and logs when confidence is below threshold."""
    mock_sources = [{"filename": "fee_waiver.pdf", "content": "Waivers subject to approval."}]
    mock_answer = "Fee waiver criteria require committee approval."
    mock_confidence = 0.55

    with patch(
        "src.api.v1.chat.run_agent_workflow",
        new=AsyncMock(return_value=(mock_answer, mock_confidence, mock_sources)),
    ):
        response = await client.post(
            "/api/v1/chat/query",
            json={
                "query": "How do fee waivers work?",
                "confidence_threshold": 0.75,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["escalated_to_audit"] is True
    assert data["confidence_score"] == 0.55
    assert data["audit_id"] is not None
    assert data["disclaimer"] is not None
    assert "not completely certain" in data["disclaimer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "fee_waiver.pdf"

    # Confirm record in database
    audit_uuid = uuid.UUID(data["audit_id"])
    stmt = select(AuditLog).where(AuditLog.id == audit_uuid)
    res = await test_session.execute(stmt)
    persisted = res.scalars().first()
    assert persisted is not None
    assert persisted.confidence_score == pytest.approx(0.55, abs=1e-3)
    assert persisted.retrieved_sources == mock_sources

