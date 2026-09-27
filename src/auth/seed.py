"""Seed the owner's admin account at startup, from environment variables only.

    SEED_ADMIN_EMAIL=you@example.com
    SEED_ADMIN_PASSWORD=<a long random password>      # in .env, never in git

Runs on every start and is idempotent: it creates the account if it is missing and promotes it to
admin if it exists without the flag. It never changes an existing password (a restart must not
silently reset credentials) - to rotate it, change it in the app or delete the account.
Admins are exempt from the per-minute limit and the daily quota.
"""

import structlog

from src.auth.security import MAX_PASSWORD_BYTES, hash_password
from src.auth.users import UserStore
from src.core.config import settings

logger = structlog.get_logger(__name__)


async def ensure_admin(users: UserStore) -> str:
    """'created' | 'promoted' | 'exists' | 'skipped'."""
    email = (settings.SEED_ADMIN_EMAIL or "").strip().lower()
    if not email:
        return "skipped"
    password = (
        settings.SEED_ADMIN_PASSWORD.get_secret_value() if settings.SEED_ADMIN_PASSWORD else ""
    )
    existing = await users.by_email(email)
    if existing:
        if existing.is_admin:
            return "exists"
        await users.set_admin(existing.id, True)
        logger.info("Seeded account promoted to admin", user_id=existing.id)
        return "promoted"
    if not (8 <= len(password.encode()) <= MAX_PASSWORD_BYTES):
        logger.warning("SEED_ADMIN_EMAIL set but SEED_ADMIN_PASSWORD missing or not 8-72 bytes")
        return "skipped"
    user = await users.create(email, hash_password(password), "Admin", True)
    logger.info("Seeded admin account", user_id=user.id)
    return "created"
