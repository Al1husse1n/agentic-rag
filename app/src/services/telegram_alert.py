import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)


def find_source_file(filename: str, metadata: dict[str, Any] | None = None) -> Path | None:
    """Attempt to locate the physical source document file on disk."""
    meta = metadata or {}
    candidates = [
        meta.get("file_path"),
        meta.get("source"),
        Path("./data/pdfs") / filename,
        Path("./data") / filename,
        Path(filename),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
    return None


def format_sources_summary(
    retrieved_sources: list[dict[str, Any]] | dict[str, Any] | None,
) -> str:
    """Format retrieved source document chunks with filename, page, and excerpt."""
    if not retrieved_sources:
        return "None (Potential Context Gap detected)"

    sources_list: list[dict[str, Any]] = (
        [{"filename": str(k), "content": str(v), "metadata": {}} for k, v in retrieved_sources.items()]
        if isinstance(retrieved_sources, dict)
        else retrieved_sources
    )

    formatted_docs: list[str] = []
    # Display up to top 3 referenced document chunks
    for idx, src in enumerate(sources_list[:3], start=1):
        filename = str(src.get("filename") or "unknown_document")
        raw_meta = src.get("metadata")
        metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        page = metadata.get("page")
        page_str = f" (p. {page})" if page is not None else ""
        content = str(src.get("content") or "").strip()
        snippet = f"{content[:180]}..." if len(content) > 180 else content

        entry = f"• [{idx}] {filename}{page_str}"
        if snippet:
            entry += f"\n  Excerpt: {snippet}"
        formatted_docs.append(entry)

    return "\n\n".join(formatted_docs)


async def send_telegram_document_file(
    bot_token: str,
    chat_id: str,
    file_path: Path,
    caption: str | None = None,
) -> bool:
    """Upload and send a physical source document file to Telegram admin chat."""
    if not file_path.is_file():
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(file_path, "rb") as f:
                files = {"document": (file_path.name, f, "application/octet-stream")}
                data = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                resp = await client.post(url, data=data, files=files)
                resp.raise_for_status()
        logger.info("Sent source document file '%s' to Telegram", file_path.name)
        return True
    except Exception as exc:
        logger.warning("Failed to send source document file '%s' to Telegram: %s", file_path.name, exc)
        return False


async def send_low_confidence_telegram_alert(
    audit_id: UUID,
    user_query: str,
    ai_response: str,
    confidence_score: float | None,
    retrieved_sources: list[dict[str, Any]] | dict[str, Any] | None = None,
    user_id: UUID | None = None,
) -> None:
    """Send a low-confidence audit alert and source documents to the admin Telegram chat."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_ADMIN_CHAT_ID:
        logger.debug("Telegram alert skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_ADMIN_CHAT_ID not set.")
        return

    score_display = f"{confidence_score:.2f}" if confidence_score is not None else "N/A"
    source_count = len(retrieved_sources) if isinstance(retrieved_sources, list) else 0
    user_display = str(user_id) if user_id is not None else "Anonymous / Unauthenticated"
    sources_summary = format_sources_summary(retrieved_sources)

    # 1. Send the primary alert message with query, answer, and source document citations
    message = (
        "⚠️ Low-Confidence RAG Query Escalated\n\n"
        f"Audit ID: {audit_id}\n"
        f"User ID: {user_display}\n"
        f"Confidence: {score_display}\n"
        f"Sources Found: {source_count}\n\n"
        f"Query:\n{user_query[:400]}\n\n"
        f"Draft Answer:\n{ai_response[:400]}\n\n"
        f"Source Documents:\n{sources_summary}"
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

    # 2. If physical source document files exist on disk, send them as attachments
    if isinstance(retrieved_sources, list):
        sent_files: set[str] = set()
        for src in retrieved_sources[:2]:  # Limit to top 2 files to avoid spamming
            if not isinstance(src, dict):
                continue
            filename = str(src.get("filename") or "")
            raw_meta = src.get("metadata")
            metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            if filename and filename not in sent_files:
                file_path = find_source_file(filename, metadata)
                if file_path:
                    sent_files.add(filename)
                    caption = f"📄 Source Document: {filename}\nAudit ID: {audit_id}"
                    await send_telegram_document_file(
                        bot_token=settings.TELEGRAM_BOT_TOKEN,
                        chat_id=settings.TELEGRAM_ADMIN_CHAT_ID,
                        file_path=file_path,
                        caption=caption,
                    )