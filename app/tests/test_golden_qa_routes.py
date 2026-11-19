"""Tests for Golden Q&A Route Functions (API Endpoints).

Verifies:
1. `GET /api/v1/golden-qa/`:
   - Retrieves structured list of all cached Q&A hotfixes.
   - Enforces RBAC permissions (`audit:read`).
2. `POST /api/v1/golden-qa/`:
   - Appends a new hotfix entry to the golden_qa markdown file.
   - Enforces RBAC permissions (`audit:write`).
   - Rejects unauthorized or forbidden users.
3. `POST /api/v1/golden-qa/check`:
   - Returns matched=True and match_score for questions matching cached entries above threshold.
   - Returns matched=False for non-matching queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from src.agent.utils import golden_qa as golden_qa_module


@pytest.fixture(autouse=True)
def isolated_golden_qa_file(tmp_path: Path):
    """Ensure tests run against an isolated temporary golden_qa.md file."""
    test_file = tmp_path / "test_golden_qa.md"
    original_filepath = golden_qa_module.DEFAULT_FILEPATH

    # Point golden_qa module to test file
    golden_qa_module.DEFAULT_FILEPATH = test_file
    yield test_file

    golden_qa_module.DEFAULT_FILEPATH = original_filepath
    if test_file.exists():
        test_file.unlink()


@pytest.mark.asyncio
async def test_create_and_list_golden_qa_entries(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify an instructor can create and subsequently list golden Q&A hotfix entries."""
    # 1. Create a golden Q&A entry
    payload = {
        "question": "What is the graduation fee?",
        "answer": "The graduation fee is $150 payable in your final semester.",
        "source_doc_id": "fees_schedule_2026.pdf",
    }
    create_resp = await client.post(
        "/api/v1/golden-qa/",
        json=payload,
        headers=instructor_user["headers"],
    )
    assert create_resp.status_code == 201
    created_data = create_resp.json()
    assert created_data["question"] == payload["question"]
    assert created_data["answer"] == payload["answer"]
    assert created_data["source_doc_id"] == payload["source_doc_id"]
    assert created_data["instructor_id"] == instructor_user["user"].full_name

    # 2. List the entries
    list_resp = await client.get(
        "/api/v1/golden-qa/",
        headers=instructor_user["headers"],
    )
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    found = any(item["question"] == payload["question"] for item in list_data["items"])
    assert found


@pytest.mark.asyncio
async def test_create_golden_qa_permission_denied_for_student(
    client: httpx.AsyncClient,
    student_user: dict[str, Any],
):
    """Verify that a student without `audit:write` is rejected with 403 Forbidden."""
    payload = {
        "question": "Can I bypass tuition?",
        "answer": "No.",
        "source_doc_id": "rules.pdf",
    }
    response = await client.post(
        "/api/v1/golden-qa/",
        json=payload,
        headers=student_user["headers"],
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_check_golden_qa_match_and_miss(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify checking a query against the golden cache returns matched or not matched."""
    # Seed an entry
    await client.post(
        "/api/v1/golden-qa/",
        json={
            "question": "Where is the financial aid office located?",
            "answer": "Student Services Center, Room 204.",
            "source_doc_id": "campus_map.pdf",
        },
        headers=instructor_user["headers"],
    )

    # 1. Test Match (slightly rephrased query)
    match_resp = await client.post(
        "/api/v1/golden-qa/check",
        json={
            "question": "Where is the financial aid office?",
            "threshold": 0.70,
        },
    )
    assert match_resp.status_code == 200
    match_data = match_resp.json()
    assert match_data["matched"] is True
    assert match_data["entry"] is not None
    assert match_data["entry"]["answer"] == "Student Services Center, Room 204."
    assert match_data["entry"]["match_score"] is not None
    assert match_data["entry"]["match_score"] >= 0.70

    # 2. Test Miss (completely unrelated question)
    miss_resp = await client.post(
        "/api/v1/golden-qa/check",
        json={
            "question": "How deep is Lake Tana?",
            "threshold": 0.70,
        },
    )
    assert miss_resp.status_code == 200
    miss_data = miss_resp.json()
    assert miss_data["matched"] is False
    assert miss_data["entry"] is None
