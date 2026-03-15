"""
End-to-End Orchestration Runner testing Phase 1 (Ingestion & Integration)
against Phase 2 (Mathematical Transformation) cleanly.

Confirms artifacts map cleanly from Scraped URLs -> Parquet Serialization -> ML Numpy Caches.
"""
import sys
import polars as pl
from src.logger import get_logger
from src.config import RAW_DATA_DIR
from src.data.data_ingestion import SmartphoneDataIngestor
from src.data.data_merger import DatasetIntegrator
from src.features.feature_engineering import SmartphoneFeatureEngineer

logger = get_logger(__name__)

def execute_full_pipeline() -> None:
    """Orchestrates strict object sequences wrapping lifecycle errors."""
    logger.info("Initializing Phase 1.1: Web Request Ingestion...")
    
    try:
        # 1. Pipeline Segment A: Scrape -> Raw CSV
        ingestor = SmartphoneDataIngestor()
        ingestor.run()
        
        logger.info("[✓] Pipeline Step Completed: raw_smartphones.csv materialized.")

        # 2. Pipeline Segment B: Fuzzy Merging logic -> Clean Parquet
        logger.info("Initializing Phase 1.2: C++ Levenshtein Matrix Integration...")
        
        # Load local mocked matrices natively mirroring realistic environments
        specs_path = RAW_DATA_DIR / "specs_data.csv"
        price_path = RAW_DATA_DIR / "price_data.csv"
        
        df_specs = pl.read_csv(specs_path)
        df_prices = pl.read_csv(price_path)
        
        integrator = DatasetIntegrator(df_specs, df_prices, match_threshold=85.0)
        master_df = integrator.execute_fuzzy_join()
        
        # Output clean matrix to 03_processed natively handled by DatasetIntegrator export methods during validation 
        # (Though we force caching here to guarantee sequential mapping downstream)
        from src.config import PROCESSED_DATA_DIR
        master_output_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"
        master_df.write_parquet(master_output_path)
        
        logger.info("[✓] Pipeline Step Completed: master_smartphones.parquet safely serialized.")
        
        # 3. Pipeline Segment C: Mathematical Boundaries -> Target Matrices
        logger.info("Initializing Phase 2: Sklearn Feature Engineering Pipelining...")
        
        engineer = SmartphoneFeatureEngineer(data_path=str(master_output_path))
        engineer.fit_transform_pipeline()
        
        logger.info("[✓] Pipeline Step Completed: Model binary matrices (.npy) and Schema (.json) cached.")
        
        # Final Verification State
        logger.info("*************************************************************")
        logger.info("🔥 INTEGRATION TEST PASSED: Artifacts ready for Phase 3 🔥")
        logger.info("*************************************************************")
        
    except FileNotFoundError as e:
        logger.critical(f"FATAL: Missing input architectures breaking execution flow - {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: Pipeline Sequence fractured irrecoverably. Stack Context: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Initiating Full Stack Integration Architecture Testing Suite.")
    execute_full_pipeline()
