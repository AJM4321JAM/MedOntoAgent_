from __future__ import annotations

import json
import subprocess
from pathlib import Path


REGION = "ap-southeast-2"

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "infrastructure" / "aws"

OTEL_CONFIG = (
    AWS_DIR / "otel-collector-aws.yaml"
).read_text(encoding="utf-8")


def aws_text(*args: str) -> str:
    result = subprocess.run(
        [
            "aws",
            *args,
            "--region",
            REGION,
            "--output",
            "text",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


account_id = aws_text(
    "sts",
    "get-caller-identity",
    "--query",
    "Account",
)

execution_role_arn = aws_text(
    "iam",
    "get-role",
    "--role-name",
    "MedOntoEcsExecutionRole",
    "--query",
    "Role.Arn",
)

task_role_arn = aws_text(
    "iam",
    "get-role",
    "--role-name",
    "MedOntoEcsTaskRole",
    "--query",
    "Role.Arn",
)

openrouter_secret = aws_text(
    "secretsmanager",
    "describe-secret",
    "--secret-id",
    "medontoagent/openrouter-api-key",
    "--query",
    "ARN",
)

qdrant_url_secret = aws_text(
    "secretsmanager",
    "describe-secret",
    "--secret-id",
    "medontoagent/qdrant-url",
    "--query",
    "ARN",
)

qdrant_key_secret = aws_text(
    "secretsmanager",
    "describe-secret",
    "--secret-id",
    "medontoagent/qdrant-api-key",
    "--query",
    "ARN",
)

admin_secret = aws_text(
    "secretsmanager",
    "describe-secret",
    "--secret-id",
    "medontoagent/admin-api-key",
    "--query",
    "ARN",
)

image_uri = (
    f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/"
    "medontoagent:v1"
)

task_definition = {
    "family": "medontoagent",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "3072",
    "executionRoleArn": execution_role_arn,
    "taskRoleArn": task_role_arn,
    "containerDefinitions": [
        {
            "name": "medontoagent",
            "image": image_uri,
            "essential": True,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp",
                }
            ],
            "environment": [
                {"name": "APP_ENV", "value": "production"},
                {"name": "AWS_REGION", "value": REGION},
                {"name": "ENABLE_DEV_ROUTES", "value": "false"},
                {"name": "OTEL_ENABLED", "value": "true"},
                {
                    "name": "OTEL_SERVICE_NAME",
                    "value": "medontoagent",
                },
                {
                    "name": "OTEL_SERVICE_VERSION",
                    "value": "0.1.0",
                },
                {
                    "name": "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "value": "http://127.0.0.1:4318",
                },
                {
                    "name": "OPENROUTER_MODEL",
                    "value": "openrouter/free",
                },
                {
                    "name": "EMBEDDING_MODEL",
                    "value": (
                        "sentence-transformers/"
                        "all-MiniLM-L6-v2"
                    ),
                },
                {
                    "name": "EMBEDDING_DIMENSIONS",
                    "value": "384",
                },
                {
                    "name": "EMBEDDING_BATCH_SIZE",
                    "value": "32",
                },
                {
                    "name": "EMBEDDING_DEVICE",
                    "value": "cpu",
                },
                {
                    "name": "QDRANT_COLLECTION",
                    "value": (
                        "medontoagent_clinical_documents"
                    ),
                },
                {
                    "name": "RETRIEVAL_TOP_K",
                    "value": "5",
                },
                {
                    "name": "MAX_AGENT_RETRIES",
                    "value": "1",
                },
            ],
            "secrets": [
                {
                    "name": "OPENROUTER_API_KEY",
                    "valueFrom": openrouter_secret,
                },
                {
                    "name": "QDRANT_URL",
                    "valueFrom": qdrant_url_secret,
                },
                {
                    "name": "QDRANT_API_KEY",
                    "valueFrom": qdrant_key_secret,
                },
                {
                    "name": "ADMIN_API_KEY",
                    "valueFrom": admin_secret,
                },
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/medontoagent",
                    "awslogs-region": REGION,
                    "awslogs-stream-prefix": "app",
                },
            },
            "healthCheck": {
                "command": [
                    "CMD-SHELL",
                    (
                        "python -c \"import urllib.request; "
                        "urllib.request.urlopen("
                        "'http://127.0.0.1:8000/health', "
                        "timeout=3).read()\" || exit 1"
                    ),
                ],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 20,
            },
        },
        {
            "name": "otel-collector",
            "image": (
                "otel/opentelemetry-collector-contrib:"
                "0.162.0"
            ),
            "essential": False,
            "command": [
                "--config=env:OTELCOL_CONFIG",
            ],
            "environment": [
                {
                    "name": "AWS_REGION",
                    "value": REGION,
                },
                {
                    "name": "OTELCOL_CONFIG",
                    "value": OTEL_CONFIG,
                },
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": (
                        "/ecs/medontoagent-otel"
                    ),
                    "awslogs-region": REGION,
                    "awslogs-stream-prefix": "collector",
                },
            },
        },
    ],
}

output = AWS_DIR / "task-definition.rendered.json"

output.write_text(
    json.dumps(task_definition, indent=2),
    encoding="utf-8",
)

print(f"Created {output}")
