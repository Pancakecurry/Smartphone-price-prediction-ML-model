# ─────────────────────────────────────────────────────────────────────────────
# Frontend Dockerfile — Streamlit Dashboard
# All data is fetched from the backend API at runtime.
# API_BASE_URL should point to the backend container (see docker-compose.yml).
# ─────────────────────────────────────────────────────────────────────────────

# python:3.11-slim required — matches backend and satisfies pandas/scikit-learn>=3.11
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── 0. System build dependencies ───────────────────────────────────────────────
# Required by: hnswlib (ChromaDB), sentence-transformers (tokenizers/Rust)
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

# ── 2. Copy project source ─────────────────────────────────────────────────────
# Only app.py and src/ are strictly required — data/ is served via the API.
COPY . .

# ── 3. Runtime environment ─────────────────────────────────────────────────────
# Set the backend API URL so the frontend knows where to send requests.
# Override at runtime: docker run -e API_BASE_URL=http://backend:8000
ENV API_BASE_URL=http://localhost:8000

EXPOSE 8501

# ── 4. Start Streamlit ─────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
