from fastapi import FastAPI

from app.api.routes_baseline import (
    router as baseline_router,
)
from app.api.routes_llm_test import (
    router as llm_test_router,
)
from app.api.routes_query import (
    router as query_router,
)
from app.config.settings import (
    get_settings,
)


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for the MedOntoAgent "
        "ontology-constrained clinical intelligence system."
    ),
    version=settings.app_version,
)


app.include_router(
    llm_test_router
)

app.include_router(
    baseline_router
)

app.include_router(
    query_router
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "environment": settings.app_env,
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "environment": settings.app_env,
    }