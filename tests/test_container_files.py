from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "infrastructure" / "docker" / "Dockerfile"
COMPOSE_FILE = ROOT / "docker-compose.yml"
DOCKERIGNORE = ROOT / ".dockerignore"
COLLECTOR_CONFIG = ROOT / "observability" / "otel-collector-config.yaml"


def test_container_files_exist():
    assert DOCKERFILE.exists()
    assert COMPOSE_FILE.exists()
    assert DOCKERIGNORE.exists()
    assert COLLECTOR_CONFIG.exists()


def test_dockerfile_uses_non_root_user_and_healthcheck():
    content = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in content
    assert "USER medonto" in content
    assert "HEALTHCHECK" in content
    assert "uvicorn" in content
    assert "app.main:app" in content


def test_dockerignore_excludes_sensitive_and_local_files():
    content = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    entries = {line.strip() for line in content if line.strip() and not line.startswith("#")}

    assert ".env" in entries
    assert ".venv" in entries
    assert ".ragas-venv" in entries
    assert "data/source_documents" in entries
    assert "tools/otelcol" in entries


def test_compose_contains_expected_services():
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))

    services = compose["services"]

    assert "medontoagent" in services
    assert "otel-collector" in services


def test_compose_uses_internal_otel_endpoint():
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))

    app = compose["services"]["medontoagent"]
    environment = app["environment"]

    assert environment["OTEL_ENABLED"] == "true"
    assert (
        environment["OTEL_EXPORTER_OTLP_ENDPOINT"]
        == "http://otel-collector:4318"
    )


def test_compose_uses_existing_collector_config():
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))

    collector = compose["services"]["otel-collector"]
    volumes = collector["volumes"]

    assert any(
        "./observability/otel-collector-config.yaml:" in volume
        for volume in volumes
    )


def test_compose_uses_non_privilege_escalation():
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))

    services = compose["services"]

    assert "no-new-privileges:true" in services["medontoagent"]["security_opt"]
    assert "no-new-privileges:true" in services["otel-collector"]["security_opt"]


def test_compose_does_not_hardcode_secret_keys():
    content = COMPOSE_FILE.read_text(encoding="utf-8")

    assert "OPENROUTER_API_KEY:" not in content
    assert "QDRANT_API_KEY:" not in content
    assert "ADMIN_API_KEY:" not in content
