"""Unit tests for Telegram admin low-confidence alerting service."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.config import settings
from src.services.telegram_alert import send_low_confidence_telegram_alert


@pytest.mark.asyncio
async def test_telegram_alert_skipped_when_unconfigured():
    """Verify that alert is gracefully skipped if bot token or chat ID is not set."""
    with patch.object(settings, "TELEGRAM_BOT_TOKEN", None), patch.object(
        settings, "TELEGRAM_ADMIN_CHAT_ID", None
    ):
        # Should execute cleanly without raising
        await send_low_confidence_telegram_alert(
            audit_id=uuid.uuid4(),
            user_query="How do I register?",
            ai_response="Registration is online.",
            confidence_score=0.45,
            retrieved_sources=[],
            user_id=None,
        )


@pytest.mark.asyncio
async def test_telegram_alert_payload_includes_user_id():
    """Verify that the Telegram alert payload includes user_id (or anonymous indicator)."""
    audit_id = uuid.uuid4()
    user_id = uuid.uuid4()

    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch.object(settings, "TELEGRAM_BOT_TOKEN", "123456:FAKE_TOKEN"), patch.object(
        settings, "TELEGRAM_ADMIN_CHAT_ID", "-100123456789"
    ), patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:

        await send_low_confidence_telegram_alert(
            audit_id=audit_id,
            user_query="Can I apply for housing late?",
            ai_response="Late applications are placed on a waitlist.",
            confidence_score=0.35,
            retrieved_sources=[{"content": "Housing rule snippet"}],
            user_id=user_id,
        )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert payload["chat_id"] == "-100123456789"
        assert str(user_id) in payload["text"]
        assert str(audit_id) in payload["text"]
        assert "0.35" in payload["text"]
        assert "User ID:" in payload["text"]


@pytest.mark.asyncio
async def test_telegram_alert_anonymous_user():
    """Verify that unauthenticated / anonymous users are labeled accordingly."""
    audit_id = uuid.uuid4()

    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch.object(settings, "TELEGRAM_BOT_TOKEN", "123456:FAKE_TOKEN"), patch.object(
        settings, "TELEGRAM_ADMIN_CHAT_ID", "-100123456789"
    ), patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:

        await send_low_confidence_telegram_alert(
            audit_id=audit_id,
            user_query="Who is the dean?",
            ai_response="The dean is Dr. Smith.",
            confidence_score=0.40,
            retrieved_sources=[],
            user_id=None,
        )

        assert mock_post.called
        payload = mock_post.call_args.kwargs["json"]
        assert "Anonymous / Unauthenticated" in payload["text"]
