"""POST /auth/register and POST /auth/login -> bearer token."""

import re

import structlog
from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field, field_validator

from src.api.deps import services
from src.api.limits import client_ip
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
async def register(
    request: Request, body: Registration, svc: Services = Depends(services)
) -> Token:
    if not settings.ALLOW_REGISTRATION:
        raise ForbiddenError("Registration is closed", code="REGISTRATION_CLOSED")
    svc.auth_throttle.check_register(client_ip(request))
    is_admin = body.email in {e.lower() for e in settings.ADMIN_EMAILS}
    user = await svc.users.create(
        body.email, hash_password(body.password), body.display_name, is_admin
    )
    logger.info("User registered", user_id=user.id, admin=is_admin)
    return _token(user.id)


@router.post("/login", response_model=Token)
async def login(request: Request, body: Credentials, svc: Services = Depends(services)) -> Token:
    email, ip = body.email.strip().lower(), client_ip(request)
    svc.auth_throttle.check_login(email, ip)  # before the password check: bcrypt is the cost
    user = await svc.users.by_email(body.email)
    # Same error and (via the dummy hash) same timing whether the email or the password is wrong.
    if not verify_password(body.password, user.password_hash if user else None) or user is None:
        svc.auth_throttle.login_failed(email, ip)
        raise AuthError("Invalid email or password", code="INVALID_CREDENTIALS")
    svc.auth_throttle.login_succeeded(email, ip)
    return _token(user.id)
