FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml first for layer caching
COPY pyproject.toml .

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir ".[server]" 2>/dev/null || pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "openai>=1.30.0" \
    "requests>=2.31.0"

# Copy all project files
COPY . .

# Install project itself (registers the server entry point)
RUN pip install --no-cache-dir -e .

ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["python", "-m", "server.app"]
