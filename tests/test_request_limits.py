from fastapi import FastAPI
from fastapi.testclient import (
    TestClient,
)

from app.security.request_limits import (
    RequestSizeLimitMiddleware,
)


def create_test_client(
    max_bytes: int,
) -> TestClient:
    app = FastAPI()

    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=max_bytes,
    )

    @app.post("/echo")
    async def echo() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return TestClient(
        app
    )


def test_small_request_is_allowed() -> None:
    client = create_test_client(
        max_bytes=1024,
    )

    response = client.post(
        "/echo",
        json={
            "message": "hello",
        },
    )

    assert response.status_code == 200


def test_large_request_is_rejected() -> None:
    client = create_test_client(
        max_bytes=128,
    )

    response = client.post(
        "/echo",
        json={
            "message": (
                "x" * 1000
            ),
        },
    )

    assert (
        response.status_code
        == 413
    )

    assert response.json() == {
        "detail": "Request body too large.",
    }
