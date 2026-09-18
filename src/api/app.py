import logging 
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings 
from src.core.logging import setup_logging, TraceIDMiddleware
from src.api.routes import health

def create_app() -> FastAPI:
    
    setup_logging()
    logger = logging.getLogger("arxiv-project")
    logger.info("Initializing application", debug = settings.DEBUG)
    
    app = FastAPI(
        title = settings.APP_NAME,
        debug = settings.DEBUG,
        docs_url = "/docs" if settings.DEBUG else None,
        redoc_url = "/redoc" if settings.DEBUG else None,
    )
    
    #MiddleWare
    app.add_middleware(TraceIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins = settings.CORS_ORIGIN,
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"],
    )
    
    
    #Routers
    app.include_router(health.router, prefix = "/api/v1")
    app.include_router(health.router)  #Root health for load balancers
    
    logger.info("Application Ready")
    
    return app 

app = create_app()

