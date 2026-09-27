"""Passwords (bcrypt) and access tokens (JWT, HS256).

bcrypt: a deliberately slow, salted hash; `BCRYPT_ROUNDS` sets the cost (2^rounds iterations).
It only reads the first 72 bytes of input, so longer passwords are rejected at validation rather
than silently truncated.

JWT: stateless, so the API needs no session store on a free host. The token carries only the user
id and an expiry; the user row is still loaded per request, so deleting an account kills its
tokens immediately. There is no refresh token: expiry is short enough (ACCESS_TOKEN_TTL_MINUTES)
that logging in again is the refresh.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from src.core.config import settings
from src.core.exceptions import AuthError

MAX_PASSWORD_BYTES = 72
# Verifying against this when the email is unknown keeps "no such user" and "wrong password"
# equally slow, so response time does not reveal which emails are registered.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt(rounds=4)).decode()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)).decode()


def verify_password(password: str, password_hash: str | None) -> bool:
    candidate = password.encode()[:MAX_PASSWORD_BYTES]
    if password_hash is None:
        bcrypt.checkpw(candidate, _DUMMY_HASH.encode())
        return False
    return bcrypt.checkpw(candidate, password_hash.encode())


def create_access_token(user_id: int, now: datetime | None = None) -> tuple[str, int]:
    """(token, expires_in_seconds)."""
    now = now or datetime.now(UTC)
    ttl = timedelta(minutes=settings.ACCESS_TOKEN_TTL_MINUTES)
    claims = {"sub": str(user_id), "iat": now, "exp": now + ttl, "iss": settings.APP_NAME}
    token = jwt.encode(claims, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, int(ttl.total_seconds())


def decode_access_token(token: str) -> int:
    """User id from a valid token; AuthError for anything else (expired, forged, malformed)."""
    # Rotation: new tokens are signed with SECRET_KEY; tokens signed with SECRET_KEY_PREVIOUS keep
    # working until they expire, so rotating the key does not log everyone out at once.
    keys = [k for k in (settings.SECRET_KEY, settings.SECRET_KEY_PREVIOUS) if k]
    for i, key in enumerate(keys):
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[settings.JWT_ALGORITHM],  # a list: never trust the token's own "alg"
                issuer=settings.APP_NAME,
                options={"require": ["sub", "exp", "iat"]},
            )
            return int(claims["sub"])
        except jwt.ExpiredSignatureError as e:
            raise AuthError("Token expired", code="TOKEN_EXPIRED") from e
        except jwt.InvalidSignatureError:
            if i + 1 < len(keys):
                continue  # maybe signed with the previous key
            raise AuthError("Invalid token", code="INVALID_TOKEN") from None
        except (jwt.InvalidTokenError, ValueError) as e:
            raise AuthError("Invalid token", code="INVALID_TOKEN") from e
    raise AuthError("Invalid token", code="INVALID_TOKEN")
