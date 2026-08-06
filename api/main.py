"""Main module booting the FastAPI application."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from api.routes import router
from api.dependencies import get_model_manager
from src.logger import get_logger

logger = get_logger("api_main")

app = FastAPI(
    title="Hotel Dynamic Pricing RL API",
    description="Production API optimizing room pricing using Tabular Q-Learning and Deep Q-Networks (DQN).",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    """Pre-loads demand models and agent weights at startup."""
    logger.info("Starting up FastAPI application...")
    try:
        # Trigger model loading
        manager = get_model_manager()
        logger.info(
            f"API Startup successful. Active model preference set to: '{manager.active_model_name}'"
        )
    except Exception as e:
        logger.error(f"API Startup model load encountered error: {e}")


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches all unhandled server exceptions."""
    logger.error(f"Unhandled error occurred on request {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected server error occurred: {str(exc)}"},
    )


# Mount routes
app.include_router(router)
