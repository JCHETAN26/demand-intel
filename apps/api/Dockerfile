# Forecast / inventory / events API. The serving bundle is mounted (or baked) at
# $DI_SERVING_DIR; the image holds code and runtime dependencies only.
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra api
COPY src ./src
RUN uv sync --frozen --no-dev --extra api

FROM python:3.11-slim
RUN useradd --create-home app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/src /app/src
ENV PATH=/opt/venv/bin:$PATH DI_SERVING_DIR=/srv/serving PYTHONUNBUFFERED=1
USER app
WORKDIR /app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')"
CMD ["uvicorn", "demand_intel.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
