import structlog

logger = structlog.get_logger(__name__)


class ArxivMindError(Exception):
    """Base error. `code` is machine-readable; `status_code` is what the API returns for it."""

    status_code: int = 500

    def __init__(self, message: str, code: str = "UNKNOWN", metadata: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.metadata = metadata or {}

    def log_error(self, **kwargs):
        logger.error(self.message, error_code=self.code, metadata=self.metadata, **kwargs)


class FetchError(ArxivMindError):
    status_code = 502


class ParseError(ArxivMindError):
    status_code = 422


class IndexError(ArxivMindError):  # noqa: A001 - domain name kept for backwards compatibility
    status_code = 503


class ValidationError(ArxivMindError):
    status_code = 400


class LLMError(ArxivMindError):
    """The language model failed (timeout, provider error, refusal)."""

    status_code = 503


class RetrievalError(ArxivMindError):
    status_code = 503


class AuthError(ArxivMindError):
    status_code = 401


class ForbiddenError(ArxivMindError):
    status_code = 403


class NotFoundError(ArxivMindError):
    status_code = 404


class ConflictError(ArxivMindError):
    status_code = 409


class RateLimitError(ArxivMindError):
    status_code = 429

    def __init__(self, message: str, retry_after: int = 60, code: str = "RATE_LIMITED"):
        super().__init__(message, code=code, metadata={"retry_after": retry_after})
        self.retry_after = retry_after
