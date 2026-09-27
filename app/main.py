from fastapi import FastAPI


app = FastAPI(
    title="MedOntoAgent API",
    description=(
        "Backend API for the MedOntoAgent ontology-constrained "
        "clinical intelligence system."
    ),
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    """
    Basic root endpoint used to confirm that the API is running.
    """
    return {
        "service": "MedOntoAgent",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Health endpoint for local testing and future cloud health checks.
    """
    return {
        "status": "healthy",
    }