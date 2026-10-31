from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import jwt
from pwdlib import PasswordHash
from src.core.config import settings


password_hash: PasswordHash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


def create_access_token(
    subject: str | UUID,
    extra_claims: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> str:

    now = datetime.now(UTC)

    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    # Incorporate custom claims if provided (without overwriting reserved claims)
    if extra_claims:
        for key, value in extra_claims.items():
            if key not in ("sub", "iat", "exp"):
                payload[key] = value

    encoded_jwt: str = jwt.encode(
        payload=payload,
        key=settings.SECRET_KEY.get_secret_value(),
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        payload: dict[str, Any] = jwt.decode(
            jwt=token,
            key=settings.SECRET_KEY.get_secret_value(),
            algorithms=[settings.ALGORITHM],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "require": ["exp", "sub"],
            },
        )
        return payload
    except jwt.PyJWTError:
        return None
