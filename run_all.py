"""
Production Master Orchestration Pipeline.

Sequentially mounts, structures, and computes the entire Smartphone Price Prediction
AI application. Progresses precisely through Web scraping (Phase 1), Matrix Engineering (Phase 2),
and Deep Learning MLflow evaluations natively (Phase 3). 
Halts instantaneously if any data assumptions are systematically fractured downstream.
"""
import sys
import polars as pl
from src.logger import get_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Import Object Oriented Engine boundaries
from src.data.data_ingestion import SmartphoneDataIngestor
from src.features.feature_engineering import SmartphoneFeatureEngineer
from src.rag.vector_store_builder import SmartphoneKnowledgeBase
from run_training import execute_training

logger = get_logger(__name__)

def execute_master_pipeline() -> None:
    """
    Constructs the end-to-end Machine learning execution runtime evaluating Phase 1 through 3.
    """
    logger.info("*" * 70)
    logger.info("       🚀 MASTER ORCHESTRATOR: SMARTPHONE AI PREDICTION 🚀")
    logger.info("*" * 70)

    # ==========================================
    # PHASE 1: Data Ingestion & Integration
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("--- STARTING PHASE 1: DATA INGESTION & INTEGRATION ---")
    logger.info("=" * 50)
    try:
        # Launch Production String Parser
        ingestor = SmartphoneDataIngestor()
        ingestor.run()
        
        logger.info("[✓] PHASE 1 COMPLETED: Production CSV Scaled and Serialized to Parquet.")
        
    except Exception as e:
        logger.critical(f"FATAL EXCEPTION: Phase 1 Data Ingestion failed - {e}")
        raise RuntimeError(f"Phase 1 Ingestion crash: {e}") from e

    # ==========================================
    # PHASE 2: Feature Engineering & Preprocessing
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("--- STARTING PHASE 2: FEATURE ENGINEERING ---")
    logger.info("=" * 50)
    try:
        master_output_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"
        engineer = SmartphoneFeatureEngineer(data_path=str(master_output_path))
        engineer.fit_transform_pipeline()
        logger.info("[✓] PHASE 2 COMPLETED: Matrix Numpy Arrays & SCALERS Generated.")
        
    except Exception as e:
        logger.critical(f"FATAL EXCEPTION: Phase 2 Feature Engineering failed - {e}")
        raise RuntimeError(f"Phase 2 Engineering crash: {e}") from e

    # ==========================================
    # PHASE 3: ML Modeling & Optuna Search Executions
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("--- STARTING PHASE 3: ML MODELING & MLFLOW TRACKING ---")
    logger.info("=" * 50)
    try:
        execute_training()
        logger.info("[✓] PHASE 3 COMPLETED: Model Binaries Optuna-Tuned & MLflow Tracked.")
        
    except Exception as e:
        logger.critical(f"FATAL EXCEPTION: Phase 3 Model Tracking Sequence failed - {e}")
        raise RuntimeError(f"Phase 3 ML execution crash: {e}") from e


    # ==========================================
    # PHASE 4: VECTOR DATABASE CONSTRUCTION
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("--- STARTING PHASE 4: VECTOR DATABASE CONSTRUCTION ---")
    logger.info("=" * 50)
    try:
        knowledge_base = SmartphoneKnowledgeBase()
        knowledge_base.build_database()
        logger.info("[✓] PHASE 4 COMPLETED: Embeddings saved to data/chromadb/.")
        
    except Exception as e:
        logger.critical(f"FATAL EXCEPTION: Phase 4 Vector Database Construction failed - {e}")
        raise RuntimeError(f"Phase 4 Vector Database crash: {e}") from e

    # ==========================================
    # MASTER RUN COMPLETION
    # ==========================================
    logger.info("\n" + "*" * 70)
    logger.info("ALL PHASES COMPLETED SUCCESSFULLY. SYSTEM READY FOR PHASE 5 (UI)")
    logger.info("*" * 70)
    
if __name__ == "__main__":
    try:
        execute_master_pipeline()
    except RuntimeError as ex:
        logger.critical(f"PIPELINE HALTED: {ex}")
        sys.exit(1)
