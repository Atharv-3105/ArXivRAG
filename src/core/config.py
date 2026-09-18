from pydantic_settings import BaseSettings, SettingsConfigDict 
from pydantic import Field, HttpUrl, validator
import os 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False, 
        extra = "ignore",
    )
    
    
    APP_NAME: str = Field(default = "ArXivMind", description = "Service name")
    DEBUG: bool = Field(default = False, description = "Enable debug mode")
    LOG_LEVEL: str = Field(default = "INFO", description = "Logging Level")
    SECRET_KEY: str = Field(..., description = "Required for sessions/auth")
    
    #API
    HOST: str = Field(default = "0.0.0.0")
    PORT: str = Field(default = 8000)
    CORS_ORIGIN: list[str] = Field(default = ["https://localhost:3000", "http://localhost:8080"])
    
    DATABASE_URL: str | None = Field(default = None, description = "PostgreSQL connection string")
    QDRANT_URL: str | None = Field(default = None, description = "Qdrant API endpoint")
    
    @validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return v.upper()
    

settings = Settings()