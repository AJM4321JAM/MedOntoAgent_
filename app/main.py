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
from app.observability.logging_config import (
    configure_logging,
)
from app.observability.tracing import (
    configure_telemetry,
)
from app.security.request_limits import (
    RequestSizeLimitMiddleware,
)


settings = get_settings()

configure_logging(
    level=settings.log_level,
)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for the MedOntoAgent "
        "ontology-constrained clinical intelligence system."
    ),
    version=settings.app_version,
)


app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_request_bytes,
)


configure_telemetry(
    app=app,
    settings=settings,
)


if settings.enable_dev_routes:
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


# Protected demonstration audit endpoint.
from app.api.routes_demo import router as demo_router
app.include_router(demo_router)

from app.api.routes_workspace import router as workspace_router
app.include_router(workspace_router)
