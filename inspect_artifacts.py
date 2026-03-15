"""
Utility script to visually inspect the generated Machine Learning matrices from Phase 2.
Provides a strictly formatted report detailing network definitions, schema mapping parameters,
and an explicit preview of the scaled feature boundaries.
"""
import json
import numpy as np
from src.config import PROCESSED_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)

def inspect_artifacts() -> None:
    """Load and format the mathematical artifacts for visual review."""
    
    x_train_path = PROCESSED_DATA_DIR / "X_train_scaled.npy"
    x_test_path = PROCESSED_DATA_DIR / "X_test_scaled.npy"
    schema_path = PROCESSED_DATA_DIR / "engineered_features.json"
    
    try:
        # 1. Direct Memory Loading
        X_train = np.load(x_train_path)
        X_test = np.load(x_test_path)
        
        # 2. Extract strictly tracked schemas
        with open(schema_path, "r") as f:
            schema_data = json.load(f)
            features = schema_data.get("features", [])
            
        # 3. Formatted Delivery Reporting
        print("\n" + "="*60)
        print("          PHASE 2: ML ARTIFACT INSPECTION REPORT")
        print("="*60)
        
        print("\n📊 1. MATRIX STRUCTURAL ALIGNMENT")
        print("-" * 40)
        print(f"X_train Shape: {X_train.shape}")
        print(f"X_test Shape:  {X_test.shape}")
        
        print("\n🏷️  2. ENGINEERED SCHEMA (TOP 5 FEATURES EXTRACTED)")
        print("-" * 40)
        top_features = features[:5] if features else ["No features identified in JSON."]
        for idx, feature in enumerate(top_features, 1):
            print(f"   [{idx}] {feature}")
            
        print("\n🔍 3. SCALED FEATURE MATRIX PREVIEW (X_train)")
        print("-" * 40)
        
        # Enforce highly-readable standardized Numpy numerical formatting
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        
        # Dynamically append headers if shape dimensions safely align
        if features and len(features) == X_train.shape[1]:
            header = " | ".join([f"{str(f)[:12]:>12}" for f in features])
            print(f"[{header}]")
            
        preview_rows = min(5, X_train.shape[0])
        for i in range(preview_rows):
            print(X_train[i])
            
        print("\n" + "="*60 + "\n")
        logger.info("Artifact inspection successfully dumped to terminal output.")
        
    except FileNotFoundError as e:
        logger.critical(f"FATAL: Required verification payload could not be located on disk - {e}")
    except Exception as e:
        logger.critical(f"FATAL: Numpy/JSON serialization error encountered: {e}")

if __name__ == "__main__":
    inspect_artifacts()
