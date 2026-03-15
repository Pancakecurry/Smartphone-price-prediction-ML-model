"""
Feature Engineering module for Phase 2 Modeling.

Responsible for safely extracting the processed target variable and enacting
a strict Training/Testing boundary split to prevent test-set data leakage
prior to any numerical scaling or category encoding algorithms.

This phase leverages Scikit-Learn `ColumnTransformer` to enforce KNN 
Iterative Imputation, Target Encoding, and Standard Scalers correctly.
"""
import os
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import json
from src.logger import get_logger
from src.config import PROCESSED_DATA_DIR

logger = get_logger(__name__)


class SmartphoneFeatureEngineer:
    """
    Handles robust separation and transformation of Machine Learning features utilizing
    Scikit-Learn Pipelines to enforce statistical integrity and prevent leakage.
    """
    
    def __init__(self, data_path: str = str(PROCESSED_DATA_DIR / "master_smartphones.parquet")):
        """
        Load the validated dataset and instantly execute an 80/20 target split.
        
        Args:
            data_path (str): The absolute path pointing to the processed payload.
        """
        try:
            self.df = pl.read_parquet(data_path)
            logger.info(f"Feature Engineer loaded dataset with dimensions: {self.df.shape}")
        except FileNotFoundError:
            logger.critical(f"FATAL: Missing Master Parquet payload at {data_path}")
            raise

        # Strictly Define Target Variable
        self.target_col = "Price"
        if self.target_col not in self.df.columns:
            raise ValueError(f"Mandatory ML target '{self.target_col}' not found in Schema.")
            
        self._initialize_split()
        self._build_pipeline()

    def _initialize_split(self) -> None:
        """
        Privately separates the Polars DataFrame into Training and Validation boundaries.
        Forces test_size=0.20 and random_state for reproducability constraints.
        """
        X = self.df.drop(self.target_col)
        y = self.df.select(self.target_col).to_series()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.to_pandas(), y.to_pandas(), test_size=0.20, random_state=42
        )
        
        logger.info(f"Target Isolated. X_train Matrix: {self.X_train.shape} | X_test Matrix: {self.X_test.shape}")

    def _build_pipeline(self) -> None:
        """
        Constructs the Scikit-Learn ColumnTransformer pipeline architectures cleanly.
        """
        # Feature Lists
        numeric_features = ["ram_gb", "battery_mah", "camera_mp"]
        # Ensure only columns actually present are scaled (mock df might lack some properties initially)
        numeric_features = [col for col in numeric_features if col in self.X_train.columns]
        
        categorical_features = ["Brand"]
        categorical_features = [col for col in categorical_features if col in self.X_train.columns]
        
        # Numeric Chain: KNN Imputation (n=5) -> StandardScaler (z = x - μ / σ)
        numeric_transformer = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])
        
        # Categorical Chain: Target Encoding mapped closely to Price.
        categorical_transformer = TargetEncoder(smoothing=1.0)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ],
            remainder="drop" # Discard generic string data like 'Smartphone_Name'
        )
        logger.info("Scikit-Learn ColumnTransformer initialized securely.")

    def fit_transform_pipeline(self) -> None:
        """
        Primary execution environment locking math states against data leakage.
        
        1. Fit/Transform EXCLUSIVELY upon X_train and y_train contexts.
        2. Transform blindly upon X_test.
        3. Serialize all final outputs into memory-mapped NumPy binaries `.npy`.
        """
        logger.info("Executing Preprocessor FIT boundary natively upon Training set.")
        X_train_scaled = self.preprocessor.fit_transform(self.X_train, self.y_train)
        
        logger.info("Executing Preprocessor TRANSFORM boundary against Test set securely.")
        X_test_scaled = self.preprocessor.transform(self.X_test)
        
        # Serialize mathematically obfuscated column names into human readable schema
        self.get_feature_names_out()
        
        # Convert outputs safely to explicit Numpy formats to respect memory persistence.
        y_train_np = self.y_train.to_numpy()
        y_test_np = self.y_test.to_numpy()
        
        self._export_binaries(X_train_scaled, X_test_scaled, y_train_np, y_test_np)

    def _export_binaries(self, X_tr: np.ndarray, X_te: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray) -> None:
        """Helper to cache engineered multidimensional arrays into optimized storage."""
        # Ensure extraction path maintains integrity
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        np.save(PROCESSED_DATA_DIR / "X_train_scaled.npy", X_tr)
        np.save(PROCESSED_DATA_DIR / "X_test_scaled.npy", X_te)
        np.save(PROCESSED_DATA_DIR / "y_train.npy", y_tr)
        np.save(PROCESSED_DATA_DIR / "y_test.npy", y_te)
        
        logger.info("🎉 Export fully completed.")
        logger.info(f"Scaled X_Train Dim: {X_tr.shape} | X_Test Dim: {X_te.shape}")

    def get_feature_names_out(self) -> None:
        """
        Extract the mapped schema nomenclature natively from Scikit-Learn transformers.
        Required because .npy matrices wipe structural column context completely.
        
        Outputs the schema as JSON for dashboard tracking (e.g., Streamlit Random Forest).
        """
        try:
            # Reconstruct numeric array sequence
            num_features = self.preprocessor.transformers_[0][2]
            # Reconstruct categoric array sequence
            cat_features = self.preprocessor.transformers_[1][2]
            
            # Combine identically to the ColumnTransformer injection sequence
            generated_schema = num_features + cat_features
            
            export_path = PROCESSED_DATA_DIR / "engineered_features.json"
            
            with open(export_path, "w") as f:
                json.dump({"features": generated_schema}, f, indent=4)
                
            logger.info(f"Schema names {generated_schema} exported exclusively to: {export_path}")
        except Exception as e:
            logger.error(f"Failed extracting transformed column identifiers tracking target schema: {e}")
            raise


if __name__ == "__main__":
    logger.info("Starting Phase 2: Feature Engineering pipeline mapping...")
    
    try:
        engineer = SmartphoneFeatureEngineer()
        engineer.fit_transform_pipeline()
        logger.info("Pipeline serialization cached sequentially to PROCESSED_DATA_DIR layer.")
        
    except Exception as e:
        logger.critical(f"FATAL Exception configuring Feature Engineering logic: {e}")
