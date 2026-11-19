"""FastAPI Route Functions for Golden Q&A Tactical Hotfixes.

Exposes endpoints to:
1. List all golden Q&A entries (`GET /api/v1/golden-qa/`) via `parse_golden_qa_file`.
2. Append a new verified Q&A hotfix (`POST /api/v1/golden-qa/`) via `append_golden_qa_entry`.
3. Check a student question against the golden cache (`POST /api/v1/golden-qa/check`) via `check_golden_cache`.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from src.agent.utils.golden_qa import (
    acheck_golden_cache,
    append_golden_qa_entry,
    check_golden_cache,
    parse_golden_qa_file,
)
from src.api.deps import CurrentUser, require_permission
from src.schemas.golden_qa import (
    GoldenQACheckRequest,
    GoldenQACheckResponse,
    GoldenQACreateRequest,
    GoldenQAEntry,
    GoldenQAListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/golden-qa", tags=["Golden Q&A Hotfixes"])


@router.get(
    "/",
    response_model=GoldenQAListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Golden Q&A Hotfixes",
    description="Retrieve all verified Q&A overrides from the golden_qa cache file.",
)
async def list_golden_qa_entries(
    current_user: CurrentUser,
    _: None = Depends(require_permission("audit:read")),
) -> GoldenQAListResponse:
    """Read and return all structured golden Q&A overrides."""
    entries_raw = parse_golden_qa_file()
    items = [
        GoldenQAEntry(
            question=entry["question"],
            answer=entry["answer"],
            source_doc_id=entry.get("source_doc_id", "unknown"),
            instructor_id=entry.get("instructor_id", "TA"),
        )
        for entry in entries_raw
    ]
    return GoldenQAListResponse(total=len(items), items=items)


@router.post(
    "/",
    response_model=GoldenQAEntry,
    status_code=status.HTTP_201_CREATED,
    summary="Create Golden Q&A Hotfix",
    description="Append a new authoritative verified Q&A override to the golden_qa cache file.",
)
async def create_golden_qa_entry(
    body: GoldenQACreateRequest,
    current_user: CurrentUser,
    _: None = Depends(require_permission("audit:write")),
) -> GoldenQAEntry:
    """Append a verified hotfix entry to the golden_qa markdown file."""
    instructor_name = body.instructor_id or current_user.full_name or "Instructor"

    append_golden_qa_entry(
        question=body.question,
        answer=body.answer,
        source_doc_id=body.source_doc_id,
        instructor_id=instructor_name,
    )
    logger.info(
        "Appended Golden Q&A hotfix by '%s' for question: '%s'",
        instructor_name,
        body.question,
    )

    return GoldenQAEntry(
        question=body.question.strip(),
        answer=body.answer.strip(),
        source_doc_id=body.source_doc_id.strip(),
        instructor_id=instructor_name.strip(),
    )


@router.post(
    "/check",
    response_model=GoldenQACheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Golden Q&A Cache Match",
    description="Check student question against cached golden Q&A entries using fast LLM semantic evaluation.",
)
async def check_golden_qa_match(
    body: GoldenQACheckRequest,
) -> GoldenQACheckResponse:
    """Check if an incoming query matches any verified golden Q&A overrides."""
    match = await acheck_golden_cache(
        student_question=body.question,
        threshold=body.threshold,
    )

    if match:
        return GoldenQACheckResponse(
            matched=True,
            entry=GoldenQAEntry(
                question=match["question"],
                answer=match["answer"],
                source_doc_id=match.get("source_doc_id", "unknown"),
                instructor_id=match.get("instructor_id", "TA"),
                match_score=match.get("match_score"),
            ),
        )

    return GoldenQACheckResponse(matched=False, entry=None)
