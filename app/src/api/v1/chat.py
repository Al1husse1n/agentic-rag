"""Chat and SSE Streaming Endpoint with HITL Quality-Control Audit Logging.

What this module does:
1. Provides the `log_audit_background` task function which persists low-confidence
   or failed responses to Neon PostgreSQL, including `confidence_score` and
   `retrieved_sources` for context gap analysis.
2. Implements the SSE (Server-Sent Events) streaming endpoint (`POST /api/v1/chat/stream`)
   that streams live reasoning/token updates and triggers `log_audit_background`
   whenever the confidence threshold is breached.
3. Implements the synchronous fallback endpoint (`POST /api/v1/chat/query`) returning
   a structured `ChatQueryResponse`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

import anyio
from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.api.deps import OptionalCurrentUser
from src.core.database import AsyncSessionLocal
from src.models.audit import AuditLog
from src.schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    SourceCitation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Agentic Chat & Streaming"])

# Default quality-control threshold: queries scoring below this trigger HITL audit
DEFAULT_AUDIT_CONFIDENCE_THRESHOLD = 0.70

# User-facing disclaimer message when AI confidence is below threshold
LOW_CONFIDENCE_USER_MESSAGE = (
    "I'm not completely certain about this answer based on our current course materials. "
    "I've automatically flagged your question for an instructor to review and verify for you."
)


async def log_audit_background(
    user_query: str,
    ai_response: str,
    confidence_score: float | None = None,
    retrieved_sources: list[dict[str, Any]] | dict[str, Any] | None = None,
    user_id: UUID | None = None,
    status: str = "pending",
    audit_id: UUID | None = None,
) -> None:

    assigned_id = audit_id or uuid.uuid4()
    logger.info(
        "Persisting low-confidence audit record in background: audit_id=%s, confidence=%s, user_id=%s",
        assigned_id,
        confidence_score,
        user_id,
    )

    try:
        async with AsyncSessionLocal() as session:
            audit_entry = AuditLog( 
                id=assigned_id,
                user_query=user_query,
                ai_response=ai_response,
                confidence_score=confidence_score,
                retrieved_sources=retrieved_sources,
                status=status,
                user_id=user_id,
            )
            session.add(audit_entry)
            await session.commit()
            logger.info("Successfully persisted AuditLog record: %s", assigned_id)

    except Exception as exc:
        logger.exception("Failed to persist AuditLog in background task: %s", exc)


# Context Extraction & Agent Invocation Helper
def extract_sources_snapshot(
    reranked_results: list[dict[str, Any]] | None,
    retrieved_results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Create a structured JSON-serializable list of document chunks for context gap analysis."""
    snapshots: list[dict[str, Any]] = []
    source_list = reranked_results if reranked_results else (retrieved_results or [])

    for item in source_list:
        sub_q = item.get("question", "") if isinstance(item, dict) else getattr(item, "question", "")
        documents = item.get("documents", []) if isinstance(item, dict) else getattr(item, "documents", [])

        for doc in documents:
            content = getattr(doc, "page_content", str(doc))
            metadata = getattr(doc, "metadata", {})
            filename = metadata.get("filename") or metadata.get("source") or "knowledge_document"

            snapshots.append({
                "sub_question": sub_q,
                "filename": filename,
                "content": content,
                "metadata": metadata,
            })

    return snapshots


async def run_agent_workflow(
    query: str,
    category: str | None = None,
    vector_backend: str = "chroma",
) -> tuple[str, float, list[dict[str, Any]]]:
    """Execute the multi-agent LangGraph RAG pipeline or safely produce fallback evidence.

    Returns:
        tuple[str, float, list[dict[str, Any]]]: (ai_answer, confidence_score, retrieved_sources)
    """
    try:
        # Lazy import of compiled agent graph to avoid heavy startup 
        from src.agent.main import main as agent_graph

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "max_try": 3,
            "try_no": 0,
        }

        # Run synchronous LangGraph invoke in an async worker thread
        result = await anyio.to_thread.run_sync(agent_graph.invoke, initial_state)

        # Extract synthesized answer
        answer = result.get("final_answer")
        if not answer:
            messages = result.get("messages", [])
            answer = messages[-1].content if messages else "No synthesized answer returned."

        # Extract validator confidence score
        val_result = result.get("validation_result") or {}
        if isinstance(val_result, dict):
            raw_score = val_result.get("confidence_score")
            confidence = float(raw_score) if raw_score is not None else 0.85
        else:
            confidence = getattr(val_result, "confidence_score", 0.85)

        # Extract retrieved & reranked context documents
        reranked = result.get("reranked_results") or []
        retrieved = result.get("retrieved_results") or []
        sources = extract_sources_snapshot(reranked, retrieved)

        return str(answer), float(confidence), sources

    except Exception as exc:
        logger.warning(
            "Agent pipeline execution fallback for query '%s': %s",
            query,
            exc,
        )
        fallback_answer = (
            "I could not locate sufficient verified course materials to answer your question with confidence. "
            "Your inquiry has been escalated to an instructor for review."
        )
        return fallback_answer, 0.0, []



def format_sse(data: dict[str, Any], event: str | None = None) -> str:
    """Format dictionary into RFC 8895 compliant Server-Sent Events text."""
    payload = json.dumps(data)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


async def chat_sse_event_stream(
    query: str,
    confidence_threshold: float,
    user_id: UUID | None,
    background_tasks: BackgroundTasks,
    category: str | None = None,
    vector_backend: str = "chroma",
) -> AsyncGenerator[str, None]:
    """Asynchronous generator that produces SSE events during chat processing."""
    # Connection start event
    yield format_sse({"type": "start", "query": query, "status": "processing"})
    await asyncio.sleep(0.01)

    
    yield format_sse({
        "type": "status",
        "stage": "retrieval",
        "message": f"Querying knowledge base ({vector_backend}) and synthesizing answer...",
    })
    await asyncio.sleep(0.01)

    # Execute live agentic RAG workflow
    ai_response, confidence_score, retrieved_sources = await run_agent_workflow(
        query=query,
        category=category,
        vector_backend=vector_backend,
    )

    # Stream retrieved source snapshots if available
    if retrieved_sources:
        yield format_sse({
            "type": "sources",
            "count": len(retrieved_sources),
            "sources": retrieved_sources,
        })
        await asyncio.sleep(0.01)

    # Stream response tokens smoothly
    words = ai_response.split(" ")
    for idx, word in enumerate(words):
        chunk = word + (" " if idx < len(words) - 1 else "")
        yield format_sse({"type": "token", "content": chunk})
        await asyncio.sleep(0.01)

    # Check quality-control threshold and trigger background audit logging if breached
    is_low_confidence = confidence_score < confidence_threshold
    audit_id: UUID | None = None

    if is_low_confidence:
        audit_id = uuid.uuid4()
        logger.warning(
            "Confidence score %.2f is below threshold %.2f for query '%s'. Escalating to instructor audit (audit_id=%s).",
            confidence_score,
            confidence_threshold,
            query,
            audit_id,
        )

        # Dispatch background task with live values
        background_tasks.add_task(
            log_audit_background,
            user_query=query,
            ai_response=ai_response,
            confidence_score=confidence_score,
            retrieved_sources=retrieved_sources,
            user_id=user_id,
            status="pending",
            audit_id=audit_id,
        )

        # Yield SSE audit escalation notification with user-friendly disclaimer
        yield format_sse({
            "type": "audit_escalation",
            "escalated": True,
            "audit_id": str(audit_id),
            "message": LOW_CONFIDENCE_USER_MESSAGE,
        })
        await asyncio.sleep(0.01)

    # Final completion event
    yield format_sse({
        "type": "done",
        "answer": ai_response,
        "escalated_to_audit": is_low_confidence,
        "disclaimer": LOW_CONFIDENCE_USER_MESSAGE if is_low_confidence else None,
        "audit_id": str(audit_id) if audit_id else None,
    })




@router.post(
    "/stream",
    summary="Stream Chat Response via SSE",
    description=(
        "Streams token-by-token AI answers with live status updates over Server-Sent Events (SSE). "
        "If confidence_score is below the threshold, live values are passed to log_audit_background "
        "and securely persisted in Neon PostgreSQL."
    ),
)
async def chat_stream_endpoint(
    request: ChatQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: OptionalCurrentUser = None,
) -> StreamingResponse:
    """Handle incoming student query with SSE streaming and background quality-control auditing."""
    user_id = current_user.id if current_user else None
    threshold = request.confidence_threshold or DEFAULT_AUDIT_CONFIDENCE_THRESHOLD

    generator = chat_sse_event_stream(
        query=request.query,
        confidence_threshold=threshold,
        user_id=user_id,
        background_tasks=background_tasks,
        category=request.category,
        vector_backend=request.vector_backend,
    )

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=headers,
        background=background_tasks,
    )


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Synchronous Chat Query",
    description="Standard request-response chat query endpoint for headless or non-streaming clients.",
)
async def chat_query_endpoint(
    request: ChatQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: OptionalCurrentUser = None,
) -> ChatQueryResponse:
    """Execute synchronous chat query with background audit logging for low-confidence results."""
    user_id = current_user.id if current_user else None
    threshold = request.confidence_threshold or DEFAULT_AUDIT_CONFIDENCE_THRESHOLD

    ai_response, confidence_score, retrieved_sources = await run_agent_workflow(
        query=request.query,
        category=request.category,
        vector_backend=request.vector_backend,
    )

    is_low_confidence = confidence_score < threshold
    audit_id: UUID | None = None

    if is_low_confidence:
        audit_id = uuid.uuid4()
        background_tasks.add_task(
            log_audit_background,
            user_query=request.query,
            ai_response=ai_response,
            confidence_score=confidence_score,
            retrieved_sources=retrieved_sources,
            user_id=user_id,
            status="pending",
            audit_id=audit_id,
        )

    # Format source citations
    citations: list[SourceCitation] = [
        SourceCitation(
            filename=src.get("filename", "unknown"),
            category=request.category,
            similarity_score=confidence_score,
            content_snippet=src.get("content", "")[:200] if src.get("content") else None,
        )
        for src in retrieved_sources
    ]

    return ChatQueryResponse(
        answer=ai_response,
        sources=citations,
        confidence_score=confidence_score,
        escalated_to_audit=is_low_confidence,
        audit_id=audit_id,
        disclaimer=LOW_CONFIDENCE_USER_MESSAGE if is_low_confidence else None,
    )
