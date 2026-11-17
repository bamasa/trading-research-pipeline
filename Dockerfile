# The production image: the package, its model extras, and `trp prod` as the entry point.
# Build: docker build -t trp .     Check: docker run --rm trp check deploy/deployment.example.yaml
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra models --no-install-project
COPY src ./src
COPY deploy ./deploy
RUN uv sync --frozen --no-dev --extra models

# No key is read and no order is sent: the broker is the paper ledger.
RUN useradd --create-home trp && mkdir -p /app/runs /app/data && chown -R trp /app/runs /app/data
USER trp
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 9108
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9108/health', timeout=2)"
ENTRYPOINT ["trp", "prod"]
CMD ["check", "deploy/deployment.example.yaml"]
