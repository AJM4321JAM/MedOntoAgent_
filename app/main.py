from fastapi import FastAPI

from app.api.routes_llm_test import router as llm_test_router
from app.config.settings import get_settings


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for the MedOntoAgent ontology-constrained "
        "clinical intelligence system."
    ),
    version=settings.app_version,
)


app.include_router(llm_test_router)


@app.get("/")
def root() -> dict[str, str]:
    """
    Basic root endpoint used to confirm that the API is running.
    """
    return {
        "service": settings.app_name,
        "environment": settings.app_env,
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Health endpoint for local testing and future cloud health checks.
    """
    return {
        "status": "healthy",
        "environment": settings.app_env,
    }