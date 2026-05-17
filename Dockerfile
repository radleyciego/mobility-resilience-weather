FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY tests/ tests/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["make"]
