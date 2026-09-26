from datetime import UTC, datetime

from fastapi import APIRouter

from src.core.config import settings

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": "0.1.0",
        "timestamp": datetime.now(UTC).isoformat(),
    }
