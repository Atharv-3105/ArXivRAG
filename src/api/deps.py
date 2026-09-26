"""FastAPI dependencies shared by routes."""

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.services import Services
from src.auth.security import decode_access_token
from src.auth.users import UserRecord
from src.core.exceptions import AuthError

_bearer = HTTPBearer(auto_error=False)  # we raise our own error shape instead of FastAPI's


def services(request: Request) -> Services:
    return request.app.state.services


async def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserRecord:
    """The user behind `Authorization: Bearer <jwt>`. The row is loaded on every request, so a
    deleted account's still-valid token stops working at once."""
    if credentials is None:
        raise AuthError("Sign in required", code="AUTH_REQUIRED")
    user_id = decode_access_token(credentials.credentials)
    user = await services(request).users.by_id(user_id)
    if user is None:
        raise AuthError("Account no longer exists", code="INVALID_TOKEN")
    return user
