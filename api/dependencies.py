"""Dependencies module providing singleton service instances via FastAPI Depends."""

from typing import Generator
from api.services import ModelManagerService

# Singleton instance initialized at startup
_model_manager_service = None


def get_model_manager() -> ModelManagerService:
    """Returns the cache singleton instance of ModelManagerService.

    Initializes the service if it doesn't already exist.

    Returns:
        ModelManagerService: The active service instance.
    """
    global _model_manager_service
    if _model_manager_service is None:
        _model_manager_service = ModelManagerService()
    return _model_manager_service
