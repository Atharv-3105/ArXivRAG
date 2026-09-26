"""POST /auth/register and POST /auth/login -> bearer token."""

import re

import structlog
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, field_validator

from src.api.deps import services
from src.api.services import Services
from src.auth.security import (
    MAX_PASSWORD_BYTES,
    create_access_token,
    hash_password,
    verify_password,
)
from src.core.config import settings
from src.core.exceptions import AuthError, ForbiddenError

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., min_length=8)

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL.match(v):
            raise ValueError("not a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def bcrypt_limit(cls, v: str) -> str:
        if len(v.encode()) > MAX_PASSWORD_BYTES:  # bcrypt ignores everything after 72 bytes
            raise ValueError(f"at most {MAX_PASSWORD_BYTES} bytes")
        return v


class Registration(Credentials):
    display_name: str | None = Field(default=None, max_length=100)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int


def _token(user_id: int) -> Token:
    token, expires_in = create_access_token(user_id)
    return Token(access_token=token, expires_in=expires_in, user_id=user_id)


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(body: Registration, svc: Services = Depends(services)) -> Token:
    if not settings.ALLOW_REGISTRATION:
        raise ForbiddenError("Registration is closed", code="REGISTRATION_CLOSED")
    is_admin = body.email in {e.lower() for e in settings.ADMIN_EMAILS}
    user = await svc.users.create(
        body.email, hash_password(body.password), body.display_name, is_admin
    )
    logger.info("User registered", user_id=user.id, admin=is_admin)
    return _token(user.id)


@router.post("/login", response_model=Token)
async def login(body: Credentials, svc: Services = Depends(services)) -> Token:
    user = await svc.users.by_email(body.email)
    # Same error and (via the dummy hash) same timing whether the email or the password is wrong.
    if not verify_password(body.password, user.password_hash if user else None) or user is None:
        raise AuthError("Invalid email or password", code="INVALID_CREDENTIALS")
    return _token(user.id)
