FROM python:3.12-slim

# ---------------------------------------------------------
# Python/runtime behavior
# ---------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


# ---------------------------------------------------------
# Create non-root application user
# ---------------------------------------------------------

RUN groupadd --system medonto \
    && useradd \
        --system \
        --gid medonto \
        --create-home \
        --home-dir /home/medonto \
        --shell /usr/sbin/nologin \
        medonto


# ---------------------------------------------------------
# Install Python dependencies first for layer caching
# ---------------------------------------------------------

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --upgrade pip \
    && python -m pip install \
        -r /app/requirements.txt


# ---------------------------------------------------------
# Copy only runtime project files
# ---------------------------------------------------------

COPY app /app/app
COPY ontology /app/ontology
COPY evals /app/evals
COPY scripts /app/scripts
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md


# ---------------------------------------------------------
# Runtime-writable directories
# ---------------------------------------------------------

RUN mkdir -p \
        /app/data/processed \
        /app/evals/results \
    && chown -R \
        medonto:medonto \
        /app


# ---------------------------------------------------------
# Security: do not run application as root
# ---------------------------------------------------------

USER medonto


# ---------------------------------------------------------
# FastAPI port
# ---------------------------------------------------------

EXPOSE 8000


# ---------------------------------------------------------
# Container health check
# ---------------------------------------------------------

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" \
    || exit 1


# ---------------------------------------------------------
# Production entrypoint
# ---------------------------------------------------------

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
