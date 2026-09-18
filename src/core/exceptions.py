import structlog 

logger = structlog.get_logger(__name__)

class ArxivMindError(Exception):
    def __init__(self, message: str, code: str = "UNKNOWN", metadata: dict | None = None):
        super().__init__(message)
        self.message = message 
        self.code = code 
        self.metadata = metadata or {}
        
    def log_error(self, **kwargs):
        logger.error(self.message, error_code = self.code, metadata = self.metadata, **kwargs)
        

class FetchError(ArxivMindError):
    pass 

class ParseError(ArxivMindError):
    pass 

class IndexError(ArxivMindError):
    pass 

class ValidationError(ArxivMindError):
    pass 