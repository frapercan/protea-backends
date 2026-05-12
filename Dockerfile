# Slim runtime image for the protea-backends PLM embedding plugins.
#
# The plugin __init__ modules are deliberately import-cheap: torch /
# transformers / esm imports live inside per-backend methods. This image
# therefore ships only the base library (numpy + protea-contracts via
# git) and leaves the heavy ML extras off the default install.
#
# Downstream images that want a specific backend should derive from this
# one and `pip install protea-backends[esm|t5|ankh|esm3c|all]` or layer
# torch / transformers / esm explicitly. That separation belongs to
# T-OPS.12 (protea-method-runtime) and follow-ups.

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.3.2

COPY pyproject.toml README.md ./
COPY poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY src/ ./src/
RUN poetry install --only main --no-interaction --no-ansi

FROM python:3.12-slim

# libgomp1 is needed by numpy / future torch layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Library image: importable as protea_backends with the four backend
# plugin entry-points registered (esm / t5 / ankh / esm3c). Loading any
# backend at runtime requires the matching extras to be installed.
CMD ["python", "-c", "import protea_backends; print('protea-backends', protea_backends.__name__, 'ready')"]
