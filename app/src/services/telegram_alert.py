import logging
from typing import Any
from uuid import UUID

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)


async def send_low_confidence_telegram_alert(
    audit_id: UUID,
    user_query: str,
    ai_response: str,
    confidence_score: float | None,
    retrieved_sources: list[dict[str, Any]] | dict[str, Any] | None = None,
    user_id: UUID | None = None,
) -> None:
    """Send a low-confidence audit alert to the admin Telegram chat without blocking."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_ADMIN_CHAT_ID:
        logger.debug("Telegram alert skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_ADMIN_CHAT_ID not set.")
        return

    score_display = f"{confidence_score:.2f}" if confidence_score is not None else "N/A"
    source_count = len(retrieved_sources) if isinstance(retrieved_sources, list) else 0
    user_display = str(user_id) if user_id is not None else "Anonymous / Unauthenticated"

    # Escape or keep plain text to avoid Markdown parse errors on dynamic LLM output
    message = (
        "⚠️ Low-Confidence RAG Query Escalated\n\n"
        f"Audit ID: {audit_id}\n"
        f"User ID: {user_display}\n"
        f"Confidence: {score_display}\n"
        f"Sources Found: {source_count}\n\n"
        f"Query:\n{user_query[:500]}\n\n"
        f"Draft Answer:\n{ai_response[:500]}"
    )

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_ADMIN_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        logger.info("Sent Telegram alert for audit_id=%s", audit_id)
    except Exception as exc:
        logger.warning("Failed to send Telegram alert for audit_id=%s: %s", audit_id, exc)