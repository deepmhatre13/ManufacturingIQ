# ── ManufacturingIQ Production Dockerfile ─────────────────────────────────────
# Multi-stage build: builder → runtime
# Base: Python 3.12 slim (Debian Bookworm)

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Build dependencies for packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        libffi-dev \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        shared-mime-info \
        fonts-liberation \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first to avoid pulling the full CUDA toolkit (~2GB)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# System libraries required at runtime:
#   • WeasyPrint 65: pango, cairo, gdk-pixbuf, fontconfig, libffi
#   • sentence-transformers / torch: libgomp (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fonts-liberation \
        libgomp1 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# ── Download SentenceTransformer model at build time ──────────────────────────
# Baking the model into the image avoids slow/failing HuggingFace downloads
# at container startup in Render's network-restricted environment.
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/embeddings
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models/embeddings'); \
print('SentenceTransformer model cached successfully')"

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Runtime environment ────────────────────────────────────────────────────────
ENV EMBEDDING_CACHE_DIR=/app/models/embeddings
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Render injects $PORT; default to 8000 for local docker run
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
