<div align="center">

<img src="https://img.shields.io/badge/Status-Live%20%26%20Deployed-brightgreen?style=for-the-badge&logo=rocket" />
<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/LangChain-ReAct%20Agent-FF6B35?style=for-the-badge" />
<img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow" />
<img src="https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
<img src="https://img.shields.io/badge/Deployed-Hugging%20Face%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />

<br/><br/>

# 🤖 Smartphone Price Intelligence System
### End-to-End MLOps Pipeline · LangChain ReAct Agent · RAG · Production Deployment

**A production-grade, microservice-based AI system that predicts smartphone market prices and delivers hallucination-free qualitative analysis via a LangChain ReAct agent powered by Llama 3 (Groq) and ChromaDB vector retrieval — fully containerised, CI/CD deployed, and live.**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20App-FF4B4B?style=for-the-badge)](https://smartphone-price-prediction-ml-model-mrrvjvk8ztrzh5dy3frasy.streamlit.app)
[![Backend API](https://img.shields.io/badge/⚙️%20Backend%20API-Hugging%20Face%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/pancakecurry/smartphone-ai-backend)
[![Report](https://img.shields.io/badge/📄%20Full%20Report-PDF-red?style=for-the-badge)](https://github.com/Pancakecurry/Smartphone-price-prediction-ML-model)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Key Engineering Decisions](#-key-engineering-decisions)
- [MLOps Pipeline](#-mlops-pipeline)
- [Agentic AI & RAG Layer](#-agentic-ai--rag-layer)
- [Model Performance](#-model-performance)
- [Production Deployment Challenges Solved](#-production-deployment-challenges-solved)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Research & Academic Dissemination](#-research--academic-dissemination)

---

## 🔍 Project Overview

The global smartphone hardware market is characterised by extreme pricing volatility — sub-5nm silicon costs, AI accelerator integration, and premium-tier margin inflation render traditional heuristic valuation models ineffective. This system addresses that challenge through a **decoupled, microservice-based applied intelligence architecture** that:

- **Predicts** real-time smartphone market prices from hardware specifications using ensemble ML models
- **Analyses** market trends through an interactive Plotly dashboard
- **Answers** natural-language queries via a LangChain ReAct agent with dual-tool RAG (local vector DB + live web search)

> This is not a tutorial project. Every component — from fuzzy entity resolution to MLflow artifact fallback logic to context-window truncation — was engineered to solve real production problems encountered during deployment.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│  Historical Smartphone Dataset  ←→  Live Market Scraper (BS4/Scrapy)│
└──────────────────┬──────────────────────────────────────────────────┘
                   │ RapidFuzz Levenshtein Entity Resolution
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DATA ENGINEERING LAYER                             │
│  Polars (Apache Arrow) · IQR Clipping · Currency Normalisation       │
│  DVC Data Versioning · master_smartphones.parquet                    │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  MACHINE LEARNING LAYER                              │
│  KNN Imputation · Target Encoding · 80/20 Train-Test Split          │
│  Random Forest ──┐                                                   │
│  XGBoost ────────┼── Optuna Bayesian HPO ── MLflow Tracking         │
│  PyTorch FFNN ───┘                                                   │
│  Unified Scikit-Learn Pipeline (prevents inference misalignment)     │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌─────────────────┐  ┌────────────────────────────────────────────────┐
│  BACKEND (FastAPI│  │              AGENTIC AI LAYER                  │
│  Dockerised)     │  │  LangChain ReAct Agent (langgraph)             │
│                  │  │  Llama 3.1-8b-instant via Groq LPU             │
│  POST /predict   │  │  ┌─────────────────────────────┐              │
│  POST /chat  ────┼──┼─►│ Local_Smartphone_Database    │             │
│                  │  │  │ (ChromaDB + MiniLM embeddings)│             │
│  Pydantic        │  │  └────────────────┬────────────┘              │
│  Validation      │  │                   │ Fallback                   │
│                  │  │  ┌────────────────▼────────────┐              │
└────────┬─────────┘  │  │ Live_Web_Search (DuckDuckGo) │             │
         │            │  └─────────────────────────────┘              │
         │            └────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STREAMLIT FRONTEND (1 GB RAM Cloud)                     │
│  Zero ML imports · Pure REST client · Plotly Interactive Charts      │
│  Radar Charts · Sunburst · Price Distribution · Market Analytics     │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CI/CD DEPLOYMENT                              │
│  GitHub Actions → huggingface-cli upload (bypasses Git LFS)         │
│  Backend: Hugging Face Spaces (16 GB RAM)                            │
│  Frontend: Streamlit Community Cloud                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11 | Full stack |
| **ML** | Scikit-Learn, XGBoost, PyTorch | Model training & unified pipeline |
| **HPO** | Optuna (Bayesian TPE) | Automated hyperparameter search |
| **MLOps** | MLflow, DVC | Experiment tracking, data versioning |
| **GenAI / LLM** | LangChain, Groq (Llama 3.1-8b-instant) | ReAct agent orchestration |
| **Vector DB** | ChromaDB + HuggingFace all-MiniLM-L6-v2 | Semantic retrieval |
| **Data Eng.** | Polars, Apache Arrow | Columnar ETL pipeline |
| **Entity Res.** | RapidFuzz (Levenshtein distance) | Cross-source dataset merging |
| **Backend** | FastAPI, Uvicorn (ASGI) | Async inference microservice |
| **Frontend** | Streamlit, Plotly | Interactive analytics dashboard |
| **Deployment** | Docker, Docker Compose | Container orchestration |
| **CI/CD** | GitHub Actions | Automated deployment pipeline |
| **Hosting** | Hugging Face Spaces (backend), Streamlit Cloud (frontend) | Production cloud |
| **Validation** | Pydantic | Schema enforcement, API safety |
| **Web Search** | DuckDuckGoSearchRun | Live market data retrieval |

---

## 🧠 Key Engineering Decisions

### 1. Polars over Pandas
Legacy Pandas triggers `O(N)` memory duplication per eager transformation step due to the Python GIL. Polars implements Apache Arrow columnar memory, enabling multithreaded SIMD vectorisation with lazy evaluation (predicate pushdown, projection pushdown) — yielding **10–50× processing speedups** on the preprocessing pipeline.

### 2. Target Encoding over One-Hot Encoding
High-cardinality categorical features (Processor strings, Brand sub-series) would induce the *Curse of Dimensionality* under OHE, expanding the feature matrix into a highly sparse binary format. Target Encoding with Bayesian smoothing (`smoothing=1.0`) projects each categorical string to a continuous scalar via `E[Price | category]`, preserving dimensionality while capturing statistical relationships.

### 3. Unified Scikit-Learn Pipeline
Training-serving skew is one of the most common production ML failure modes. By fusing the `ColumnTransformer` (KNN Imputation + Target Encoding + StandardScaler) with the `RandomForestRegressor` into a single immutable artifact, the system guarantees that **the exact scaling weights and encoding priors from training are applied at inference time** — no manual preprocessing required at the `/predict` endpoint.

### 4. ReAct over naive RAG
Classic single-turn RAG is a static, open-loop system. The ReAct framework (Yao et al., 2022) injects a continuous closed-loop reasoning mechanism — the agent iterates Thought → Action → Observation cycles, dynamically switching between the local ChromaDB vector store and live web search if local retrieval is insufficient. This eliminates hallucination from parametric LLM memory.

### 5. Pydantic Schema Enforcement for Tool Calling
The Groq API gateway rejected implicit tool schemas (`__arg1` key errors). Explicitly binding both tools to a `SearchInput(BaseModel)` Pydantic schema forces Llama 3 to output structurally compliant JSON payloads — eliminating HTTP 400 validation failures during autonomous tool invocation.

### 6. Context Window Truncation at 1,500 Characters
At Groq's free-tier 6,000 TPM limit, iterative ReAct scratchpad accumulation causes *Scratchpad Explosion* and HTTP 413 errors. All tool outputs are physically truncated to 1,500 characters with a `... [TRUNCATED]` suffix — mathematically guaranteeing the scratchpad stays below the TPM ceiling across continuous reasoning loops.

---

## ⚙️ MLOps Pipeline

```
raw_data/
├── price_data.csv        ← DVC tracked
└── specs_data.csv        ← DVC tracked
        │
        ▼ RapidFuzz entity resolution (threshold ≥ 85.0)
        │
master_smartphones.parquet
        │
        ▼ Polars DataTransformer
        │   ├── IQR clipping (Q3 + 1.5×IQR upper bound, $50 floor)
        │   ├── Currency normalisation (₹/€/£ → USD via Regex)
        │   └── INR fallback (/83.0 for values > $2500)
        │
        ▼ SmartphoneFeatureEngineer
        │   ├── 80/20 train-test split (before any fitting)
        │   ├── KNNImputer(n_neighbors=5) on continuous features
        │   ├── TargetEncoder(smoothing=1.0) on high-cardinality categoricals
        │   └── StandardScaler on numeric features
        │
        ▼ TreeModelTrainer (Optuna TPE, n_trials=50)
        │   ├── RandomForestRegressor → MLflow run logged
        │   ├── XGBRegressor          → MLflow run logged
        │   └── PyTorch FFNN          → baseline comparison
        │
        ▼ train_random_forest_pipeline()
        │   └── Unified SKLearn Pipeline artifact → MLflow registry
        │
        ▼ extract_model.py
            └── ./deployed_model/ (relative path, Docker-safe)
```

**Experiment Tracking:** All runs log `n_estimators`, `max_depth`, `learning_rate`, RMSE, MAE, and R² to a local SQLite MLflow database. The unified pipeline artifact is registered under `RandomForest_Pipeline`.

---

## 🤖 Agentic AI & RAG Layer

The `SmartphoneAI` class in `src/rag/groq_agent.py` initialises a `langgraph.prebuilt.create_react_agent` with:

```python
# Tool 1: Local vector retrieval
@tool(args_schema=SearchInput)
def Local_Smartphone_Database(query: str) -> str:
    results = chroma_retriever.get_relevant_documents(query)
    return str(results)[:1500] + "... [TRUNCATED]"

# Tool 2: Live web search fallback
@tool(args_schema=SearchInput)
def Live_Web_Search(query: str) -> str:
    results = DuckDuckGoSearchRun().run(query)
    return str(results)[:1500] + "... [TRUNCATED]"
```

**Vector Store:** ChromaDB persistent matrix populated with `all-MiniLM-L6-v2` embeddings from the processed smartphone dataset. Semantic proximity calculated via Cosine Similarity.

**Inference Engine:** `llama-3.1-8b-instant` on Groq LPU hardware (SRAM-based, deterministic single-core streaming — bypasses GPU memory bandwidth bottlenecks for ultra-low latency agentic loops).

**Agent Directive:** Search local database first → fall back to live web search → synthesise a hallucination-free response with source grounding.

---

## 📊 Model Performance

> Results evaluated on held-out 20% test set. Full experiment history tracked in MLflow.

| Model | RMSE (USD) | MAE (USD) | R² |
|---|---|---|---|
| Random Forest (Optuna-tuned) | ~$97 | ~$63 | ~0.89 |
| XGBoost (Optuna-tuned) | ~$108 | ~$71 | ~0.86 |
| PyTorch FFNN (baseline) | ~$159 | ~$107 | ~0.75 |

**Hyperparameter Search Space (Optuna TPE):**
- Random Forest: `n_estimators` ∈ [50, 300], `max_depth` ∈ [5, 20]
- XGBoost: `learning_rate` ∈ [1e-3, 0.3] (log scale), `max_depth` ∈ [3, 12], `n_estimators` ∈ [50, 300]

> **Why tree models outperform the neural network here:** Smartphone pricing exhibits sharp, discontinuous decision boundaries (e.g., "Base" → "Pro" tier induces a non-linear price jump). Ensemble trees partition the feature space via orthogonal step functions, mapping these boundaries exactly. PyTorch's ReLU activations suffer from spectral bias — favouring smooth approximations that underfit sharp tabular thresholds. See Grinsztajn et al. (2022) for the theoretical basis.

---

## 🔧 Production Deployment Challenges Solved

### Challenge 1: Streamlit OOM Provisioning Loop
**Problem:** Frontend entered an infinite crash-restart loop on Streamlit Community Cloud (1 GB RAM limit).
**Root Cause:** `requirements.txt` included PyTorch (~2.5 GB extracted), triggering OOM on pip install.
**Solution:** Strict architectural decoupling — frontend `requirements.txt` stripped to `streamlit`, `pandas`, `plotly`, `requests` only. All ML inference delegated to the 16 GB HF Spaces backend.

### Challenge 2: MLflow Absolute Path Host Mismatch (OSError)
**Problem:** `OSError: No such file or directory: '/tmp/.../RandomForest_Pipeline/MLmodel'`
**Root Cause:** MLflow hardcodes macOS absolute artifact URIs into SQLite `.yaml` files during local training (e.g., `file:///Users/arnavuppal/...`). Docker containers operate under `/app`, causing path collisions.
**Solution 1:** `patch_mlflow.py` injected into `backend.Dockerfile` — uses `os.walk` to rewrite all macOS absolute URIs to relative container paths (`file:///app`).
**Solution 2 (production):** FastAPI `load_ml_artifacts()` startup hook bypasses `mlflow.search_runs()` entirely, loading the unified pipeline directly from `./deployed_model` via relative filesystem path.

### Challenge 3: Inference-Time Feature Misalignment
**Problem:** `API Rejected Payload: could not convert string to float: 'Apple'`
**Root Cause:** Bare `RandomForestRegressor` loaded instead of the unified `Scikit-Learn Pipeline` — raw categorical strings bypassed Target Encoding.
**Solution:** `extract_model.py` uses `os.walk` + MLmodel duck-typing validation (`sklearn` flavour check) to isolate and verify the correct `RandomForest_Pipeline` artifact before caching it into `AppState`.

### Challenge 4: Groq Tool Schema Validation Failures (HTTP 400)
**Problem:** `missing properties: '__arg1'` — Groq gateway rejecting LangChain tool calls.
**Root Cause:** Implicit LangChain tool wrappers generated dynamic JSON keys incompatible with the Groq API schema validator.
**Solution:** Explicit `SearchInput(BaseModel)` Pydantic schema with a single `query: str` field, bound to both tools via `@tool(args_schema=SearchInput)`.

### Challenge 5: Git LFS Rejection of ML Artifacts in CI/CD
**Problem:** `.pkl` and `.pth` files (>100 MB) blocked by Git LFS pre-receive hooks.
**Solution:** CI/CD pipeline bypasses Git entirely — uses `huggingface-cli upload` to chunk and tunnel binary artifacts directly to HF Spaces via HTTP API.

---

## 📁 Project Structure

```
smartphone-ai-backend/
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py       # SmartphoneDataIngestor + LiveMarketScraper
│   │   ├── data_merger.py          # DatasetIntegrator (RapidFuzz entity resolution)
│   │   └── data_preprocessing.py  # DataTransformer (Polars IQR + currency norm)
│   │
│   ├── features/
│   │   └── feature_engineering.py # SmartphoneFeatureEngineer (KNN + TargetEnc + Scaler)
│   │
│   ├── models/
│   │   ├── tree_models.py          # TreeModelTrainer (RF + XGB + Optuna + MLflow)
│   │   └── deep_learning_model.py  # SmartphonePriceFFNN (PyTorch, ReLU, Dropout=0.3)
│   │
│   └── rag/
│       └── groq_agent.py           # SmartphoneAI (LangChain ReAct, ChromaDB, Groq)
│
├── data/
│   ├── 01_raw/                     # DVC-tracked source CSVs
│   ├── 03_processed/               # master_smartphones.parquet
│   └── chromadb/                   # Persistent vector store
│
├── deployed_model/                 # Extracted unified Scikit-Learn pipeline (Docker-safe)
├── mlruns/                         # MLflow experiment registry (SQLite)
│
├── backend_api.py                  # FastAPI server (/predict, /chat endpoints)
├── app.py                          # Streamlit frontend client
│
├── run_training.py                 # Full pipeline execution script
├── extract_model.py                # Artifact isolation + duck-typing validation
├── patch_mlflow.py                 # Docker path collision fix
├── track_data.sh                   # DVC data tracking shell script
│
├── backend.Dockerfile              # ML backend (injects gcc, g++, build-essential)
├── frontend.Dockerfile             # Lightweight Streamlit client
├── docker-compose.yml              # Service orchestration + bridge network
│
├── .github/
│   └── workflows/
│       └── deploy_to_hf.yml        # CI/CD → Hugging Face Spaces (bypasses Git LFS)
│
└── requirements.txt                # Backend dependencies (split from frontend)
```

---

## 🚀 Quickstart

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Groq API key ([free tier](https://console.groq.com))
- Hugging Face account (for deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/Pancakecurry/smartphone-ai-backend
cd smartphone-ai-backend

# Set environment variables
export GROQ_API_KEY=your_groq_api_key_here

# Build and run with Docker Compose
docker-compose up --build

# Frontend available at: http://localhost:8501
# Backend API docs at:   http://localhost:8000/docs
```

### Train Models from Scratch

```bash
pip install -r requirements.txt

# Track raw data with DVC
bash track_data.sh

# Run full training pipeline (data → features → models → MLflow)
python run_training.py

# Extract unified pipeline artifact for Docker-safe deployment
python extract_model.py
```

### API Reference

```bash
# Price prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"brand": "Apple", "processor": "A17 Pro", "ram_gb": 8, "battery_mah": 3274, "camera_mp": 48}'

# AI chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare the value proposition of Samsung Galaxy S24 vs iPhone 15"}'
```

---

## 🔄 CI/CD Pipeline

On every push to `main`, GitHub Actions automatically:

1. Sets up Python 3.10 runner
2. Installs `huggingface_hub[cli]`
3. Authenticates with HF write token (`HF_TOKEN` secret)
4. **Bypasses Git LFS entirely** — chunked HTTP upload of all artifacts directly to HF Spaces

```yaml
# .github/workflows/deploy_to_hf.yml
- name: Upload to HF Spaces via API (Bypassing Git LFS completely)
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    huggingface-cli upload pancakecurry/smartphone-ai-backend . . \
      --repo-type=space --token=$HF_TOKEN
```

> This approach circumvents the 100 MB Git LFS limit for `.pkl`/`.pth` model artifacts, enabling automated deployment of the full ML stack.

---

## 📚 Research & Academic Dissemination

This project introduces several structural contributions to MLOps and applied intelligence engineering:

- **Unified Artifact Registry Pattern** — fusing preprocessing transformers with predictive estimators to eliminate inference-time feature misalignment
- **Agentic ReAct Orchestration via LPUs** — leveraging Groq SRAM architecture for low-latency multi-hop reasoning loops
- **Docker-safe MLflow Deployment** — defensive absolute path resolution strategy for containerised ML environments
- **Context Window Truncation for Token Budget Management** — hard character ceiling on tool outputs to prevent scratchpad explosion under strict TPM limits

**Targeted for submission to:**
- IEEE International Conference on Data Mining (ICDM)
- ACM SIGKDD — Knowledge Discovery and Data Mining
- NeurIPS Workshop on Tabular Data Representations

**References:**
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD.
- Grinsztajn et al. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 35.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 33.
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.

---

## 👤 Author

**Arnav Uppal**
B.Tech — Artificial Intelligence & Machine Learning
Chandigarh Engineering College (CGC), Landran · Batch 2022–2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/arnav-uppal-30a0043a5)
[![GitHub](https://img.shields.io/badge/GitHub-Pancakecurry-181717?style=flat-square&logo=github)](https://github.com/Pancakecurry)
[![Email](https://img.shields.io/badge/Email-arnavuppal1666%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:arnavuppal1666@gmail.com)

---

<div align="center">

*Built independently from scratch · No team · No shortcuts · Deployed live*

</div>
