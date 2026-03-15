"""
Decoupled FastAPI Backend for Smartphone Price Prediction.

Exposes REST API endpoints allowing the UI frontend to securely request
ML Price Predictions (Phase 3) and RAG Analytics context (Phase 4).
"""
import os
import math
import mlflow
import polars as pl
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.features.feature_engineering import SmartphoneFeatureEngineer
from src.rag.groq_agent import SmartphoneAI
from src.data.data_ingestion import SmartphoneDataIngestor, LiveMarketScraper
from src.data.data_merger import DatasetIntegrator
from src.rag.vector_store_builder import SmartphoneKnowledgeBase
from src.logger import get_logger
from src.config import PROCESSED_DATA_DIR

logger = get_logger(__name__)

# Initialize FastAPI App
app = FastAPI(title="Smartphone Market API", version="1.0")

# Configure CORS Middleware for Desktop UI integration mapping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class PhoneSpecs(BaseModel):
    """Payload definition explicitly matching the ML Feature Engineer boundaries."""
    Brand: str
    ram_gb: float
    battery_mah: float
    camera_mp: float
    Processor: str = "Unknown"

class ChatQuery(BaseModel):
    """Payload definition locking RAG prompt injections safely."""
    query: str

# ---------------------------------------------------------
# Global App State (Loaded on Startup)
# ---------------------------------------------------------
class AppState:
    rf_pipeline  = None   # Unified sklearn Pipeline(preprocessor → RF) — primary
    preprocessor = None   # Legacy fallback: separate ColumnTransformer
    rf_model     = None   # Legacy fallback: separate RF model
    ai_agent     = None

@app.on_event("startup")
def load_ml_artifacts():
    """
    Initializes and caches all ML artifacts on server boot.
    Prefers the unified Pipeline artifact ('RandomForest_Pipeline') which bundles
    preprocessing + model into a single object, eliminating scaling mismatch.
    Falls back to loading the separate preprocessor + RF model for backward compatibility.
    """
    logger.info("Initializing Backend AI services globally...")

    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        # ── 1. Try to load unified Pipeline first ──────────────────────
        pipeline_runs = mlflow.search_runs(
            experiment_names=["Smartphone_Price_Prediction"],
            filter_string="tags.mlflow.runName = 'RandomForest_Pipeline'",
            order_by=["metrics.RMSE ASC"]
        )

        if not pipeline_runs.empty:
            best_run_id = pipeline_runs.iloc[0]["run_id"]
            pipeline_uri = f"runs:/{best_run_id}/RandomForest_Pipeline"
            AppState.rf_pipeline = mlflow.sklearn.load_model(pipeline_uri)
            logger.info(
                f"✅ Loaded UNIFIED Pipeline (preprocessor + RF) from RunID: {best_run_id}. "
                f"No manual scaling needed at inference."
            )
        else:
            # ── 2. Legacy fallback: separate preprocessor + RF model ───
            logger.warning(
                "No 'RandomForest_Pipeline' MLflow run found. "
                "Falling back to separate preprocessor + RF model. "
                "Run train_random_forest_pipeline() to fix scaling issues."
            )
            logger.info("Refitting Scikit-Learn ColumnTransformer (legacy mode)...")
            engineer = SmartphoneFeatureEngineer()
            engineer.fit_transform_pipeline()
            AppState.preprocessor = engineer.preprocessor

            legacy_runs = mlflow.search_runs(
                experiment_names=["Smartphone_Price_Prediction"],
                filter_string="tags.mlflow.runName = 'RandomForest_Optuna_Optimized'",
                order_by=["metrics.RMSE ASC"]
            )
            if legacy_runs.empty:
                raise LookupError(
                    "No RF model found in MLflow. Run the training pipeline first."
                )
            best_run_id = legacy_runs.iloc[0]["run_id"]
            model_uri = f"runs:/{best_run_id}/model"
            AppState.rf_model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded legacy RF model from RunID: {best_run_id}")

        # ── 3. Mount Smartphone RAG Agent ──────────────────────────────
        logger.info("Instantiating Llama 3 LCEL Engine Context...")
        AppState.ai_agent = SmartphoneAI()

    except Exception as e:
        logger.critical(f"FATAL: Application failed backend initialization: {e}")
        raise RuntimeError(f"Global Boot Failure: {e}") from e

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.post("/predict")
def predict_price(specs: PhoneSpecs):
    """
    Accepts raw phone specs and returns a predicted price.

    If a unified Pipeline is loaded (AppState.rf_pipeline), passes the raw input
    directly — the pipeline handles ColumnTransformer scaling internally.
    Falls back to the legacy separate-preprocessor path if no pipeline is available.
    """
    import pandas as pd

    if AppState.rf_pipeline is None and (AppState.rf_model is None or AppState.preprocessor is None):
        raise HTTPException(status_code=503, detail="ML models not initialized. Check server logs.")

    try:
        # Build a raw Pandas DataFrame matching the training schema exactly
        raw_input = pd.DataFrame([{
            "Brand":       specs.Brand,
            "ram_gb":      specs.ram_gb,
            "battery_mah": specs.battery_mah,
            "camera_mp":   specs.camera_mp,
            "Processor":   specs.Processor,
        }])

        if AppState.rf_pipeline is not None:
            # ✅ Unified pipeline path — no manual scaling
            prediction = AppState.rf_pipeline.predict(raw_input)
        else:
            # Legacy path — manual transform then predict
            X_scaled = AppState.preprocessor.transform(raw_input)
            prediction = AppState.rf_model.predict(X_scaled)

        return {"predicted_price": round(float(prediction[0]), 2)}

    except Exception as e:
        logger.error(f"Prediction Pipeline Crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_agent(chat_query: ChatQuery):
    """
    Connects into Groq native web-services safely returning formatted Context strings 
    mapped locally via ChromaDB.
    """
    if AppState.ai_agent is None:
        raise HTTPException(status_code=503, detail="AI LCEL Retrieval Agent offline physically.")
        
    try:
        response = AppState.ai_agent.ask_question(chat_query.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat Inference Crash tracking RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Market Data Endpoint
# ---------------------------------------------------------

@app.get("/api/v1/market-data")
def get_market_data():
    """
    Reads the current master_smartphones.parquet and returns all rows as JSON.
    Safe for the Streamlit frontend to consume directly — no ML imports needed client-side.
    """
    parquet_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"
    if not parquet_path.exists():
        raise HTTPException(
            status_code=404,
            detail="master_smartphones.parquet not found. Run the ingestion pipeline first."
        )
    try:
        df = pl.read_parquet(parquet_path)
        # Replace NaN/Inf with None so JSON serialisation never crashes
        records = [
            {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in row.items()}
            for row in df.to_dicts()
        ]
        return {"rows": len(records), "data": records}
    except Exception as e:
        logger.error(f"market-data read error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Pipeline Trigger Endpoint (BackgroundTask)
# ---------------------------------------------------------

class PipelineStatus:
    """Simple shared-state flag so clients can poll pipeline progress."""
    running: bool = False
    last_result: str = "Pipeline has not been triggered yet."


def _run_pipeline_background() -> None:
    """
    Full pipeline executed in the background:
      1. Live market upsert (LiveMarketScraper → DatasetIntegrator.upsert_live_data)
      2. Feature engineering
      3. ML model re-training  (skipped to keep runtime short; re-enable as needed)
      4. Vector DB rebuild
    """
    if PipelineStatus.running:
        logger.warning("Pipeline trigger ignored — a run is already in progress.")
        return

    PipelineStatus.running = True
    PipelineStatus.last_result = "Pipeline running..."
    logger.info("=== Background Pipeline Triggered ===")

    try:
        parquet_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"

        # Phase 1a: Historical ingestion (only if parquet missing)
        if not parquet_path.exists():
            logger.info("[BG] Running historical data ingestion...")
            SmartphoneDataIngestor().run()

        # Phase 1b: Live market upsert
        logger.info("[BG] Fetching live market prices...")
        historical_df = pl.read_parquet(parquet_path)
        live_df = LiveMarketScraper().fetch_live_prices()
        integrator = DatasetIntegrator(historical_df, historical_df)
        unified_df = integrator.upsert_live_data(historical_df, live_df, output_path=parquet_path)
        logger.info(f"[BG] Upsert complete. Dataset: {len(unified_df)} rows.")

        # Phase 2: Feature engineering
        logger.info("[BG] Re-fitting feature engineering pipeline...")
        engineer = SmartphoneFeatureEngineer(data_path=str(parquet_path))
        engineer.fit_transform_pipeline()
        AppState.preprocessor = engineer.preprocessor
        logger.info("[BG] Feature engineering complete. Preprocessor refreshed.")

        # Phase 3: Retrain unified pipeline and hot-swap in AppState
        logger.info("[BG] Retraining unified RF Pipeline...")
        from src.models.tree_models import TreeModelTrainer
        trainer = TreeModelTrainer()
        trainer.train_random_forest_pipeline(n_trials=5)
        # Reload the freshly trained pipeline from MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        pipeline_runs = mlflow.search_runs(
            experiment_names=["Smartphone_Price_Prediction"],
            filter_string="tags.mlflow.runName = 'RandomForest_Pipeline'",
            order_by=["metrics.RMSE ASC"]
        )
        if not pipeline_runs.empty:
            best_run_id = pipeline_runs.iloc[0]["run_id"]
            AppState.rf_pipeline = mlflow.sklearn.load_model(
                f"runs:/{best_run_id}/RandomForest_Pipeline"
            )
            AppState.rf_model = None   # clear legacy artifact
            logger.info("[BG] Unified Pipeline hot-swapped into AppState.")

        # Phase 4: Rebuild vector DB so /chat reflects new data
        logger.info("[BG] Rebuilding ChromaDB vector store...")
        SmartphoneKnowledgeBase().build_database()
        logger.info("[BG] Vector DB rebuilt.")

        PipelineStatus.last_result = (
            f"Pipeline completed successfully. "
            f"Dataset now has {len(unified_df)} rows."
        )
        logger.info("=== Background Pipeline Completed ===")

    except Exception as e:
        error_msg = f"Background pipeline failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        PipelineStatus.last_result = error_msg
    finally:
        PipelineStatus.running = False


@app.post("/api/v1/trigger-pipeline")
def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    Fires the full data refresh + vector DB rebuild as a non-blocking background task.
    Returns immediately so the UI stays responsive.
    """
    if PipelineStatus.running:
        return {
            "status": "already_running",
            "message": "A pipeline run is already in progress. Please wait."
        }

    background_tasks.add_task(_run_pipeline_background)
    return {
        "status": "triggered",
        "message": "Pipeline started in the background. Poll GET /api/v1/pipeline-status for updates."
    }


@app.get("/api/v1/pipeline-status")
def get_pipeline_status():
    """Lightweight polling endpoint so the frontend can track background pipeline progress."""
    return {
        "running": PipelineStatus.running,
        "last_result": PipelineStatus.last_result,
    }
