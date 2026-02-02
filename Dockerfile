# Unleash V35 Production Dockerfile
# Phase 15: Production Deployment
#
# Build: docker build -t unleash:v35 .
# Run: docker run -p 8000:8000 --env-file .env unleash:v35

FROM python:3.14-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Builder stage - install dependencies
# ============================================================
FROM base AS builder

# Copy dependency files
COPY pyproject.toml .
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -e . || pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Install core SDKs (minimal for production)
RUN pip install --no-cache-dir \
    anthropic \
    openai \
    pydantic>=2.0 \
    click \
    rich \
    structlog \
    httpx \
    aiohttp

# ============================================================
# Production stage
# ============================================================
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY core/ core/
COPY config/ config/
COPY scripts/ scripts/
COPY tests/ tests/

# Create non-root user
RUN groupadd -r unleash && useradd -r -g unleash unleash
RUN mkdir -p /home/unleash/.unleash && chown -R unleash:unleash /home/unleash /app

USER unleash

# Set application environment
ENV UNLEASH_ENV=production \
    UNLEASH_LOG_LEVEL=INFO \
    HOME=/home/unleash

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python core/health.py -q || exit 1

# Expose port (if running as service)
EXPOSE 8000

# Default command - show status
CMD ["python", "-m", "core.cli.unified_cli", "status"]

# ============================================================
# Alternative entrypoints
# ============================================================
# Run CLI interactively:
#   docker run -it --env-file .env unleash:v35 python -m core.cli.unified_cli --help
#
# Run health check:
#   docker run --env-file .env unleash:v35 python core/health.py
#
# Run validation:
#   docker run --env-file .env unleash:v35 python scripts/final_validation.py
