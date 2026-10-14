# Stage 1: Build & Dependencies
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Pre-download the default ONNX MiniLM model to bake it into the image
RUN python -c "import chromadb; client = chromadb.Client(); coll = client.create_collection('init'); coll.add(ids=['1'], documents=['init text'])"

# Stage 2: Minimal Non-Root Runtime
FROM python:3.11-slim as runner

WORKDIR /app

# Create unprivileged non-root user
RUN groupadd -r freight && useradd -r -g freight freight

# Copy installed Python packages and Chroma cache from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /home/freight/.cache

# Copy application source, configuration, and data
COPY src /app/src
COPY data /app/data
COPY pyproject.toml /app/pyproject.toml

# Set permissions
RUN mkdir -p /app/outputs /app/.cache && chown -R freight:freight /app /home/freight

USER freight

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/freight

# Default entrypoint runs the full pipeline
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--no-cache"]