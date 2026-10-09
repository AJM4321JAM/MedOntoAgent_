from app.config.settings import (
    get_settings,
)
from app.main import app


def _public_paths() -> set[str]:
    schema = app.openapi()

    return set(
        schema["paths"].keys()
    )


def test_dev_route_matches_configuration() -> None:
    settings = get_settings()

    paths = _public_paths()

    if settings.enable_dev_routes:
        assert "/llm-test" in paths
    else:
        assert "/llm-test" not in paths


def test_public_routes_remain_available() -> None:
    paths = _public_paths()

    assert "/" in paths
    assert "/health" in paths
    assert "/query" in paths
    assert "/query-baseline" in paths
