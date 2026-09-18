from fastapi import APIRouter 
from datetime import datetime, timezone
from src.core.config import settings

router = APIRouter()

@router.get("/health", tags = ["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    }