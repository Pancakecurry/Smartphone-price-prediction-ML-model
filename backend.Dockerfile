# ─────────────────────────────────────────────────────────────────────────────
# Backend Dockerfile — FastAPI Inference + RAG Server
# Serves:  POST /predict   (Unified RF Pipeline)
#          POST /chat       (Groq ReAct Agent + ChromaDB)
#          GET  /api/v1/market-data
#          POST /api/v1/trigger-pipeline
# ─────────────────────────────────────────────────────────────────────────────

# python:3.11-slim required — scikit-learn==1.8.0 and pandas==2.3.x mandate Python>=3.11
FROM python:3.11-slim

# Suppress pip warnings and ensure stdout is unbuffered for clean logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── 0. System build dependencies ───────────────────────────────────────────────
# Required by: hnswlib (ChromaDB), scikit-learn (Cython), tokenizers (Rust)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 1. Upgrade build tools, then install Python dependencies ───────────────────
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ── 2. Copy full project context (data/, mlruns/, src/ included per .dockerignore) ──
COPY . .

# ── 3. Rewrite host-machine absolute paths in MLflow metadata → /app ───────────
# MLflow bakes the artifact root path at training time. All .yaml / .json files
# in mlruns/ still contain /Users/arnavuppal/... and must be patched before boot.
RUN python patch_mlflow.py

# ── 4. Runtime environment ─────────────────────────────────────────────────────
# GROQ_API_KEY must be injected at runtime via:
#   docker run -e GROQ_API_KEY=... or Docker Compose env_file: .env
EXPOSE 7860

# ── 4. Start server ────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "7860"]
