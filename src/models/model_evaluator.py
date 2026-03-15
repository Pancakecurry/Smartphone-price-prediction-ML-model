"""
MLflow Evaluator Module for Phase 3 Modeling.

Responsible for tracking model experiments, calculating KPIs (RMSE, R2),
and saving training artifacts securely to the local mlruns directory.
"""
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any
from src.config import PROCESSED_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)

class BaseModelTrainer:
    """
    Base Model Training Handler wrapping Scikit-Learn algorithms with MLflow
    tracking mechanics. Ensures parameter and metric persistence.
    """
    
    def __init__(self, experiment_name: str = "Smartphone_Price_Prediction"):
        """
        Initializes MLflow tracking context and loads preprocessed Numpy matrices.
        """
        self.experiment_name = experiment_name
        
        # Configure MLflow tracking
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow Evaluator Online. Tracking to local sqlite:///mlflow.db under: '{self.experiment_name}'")
        
        # Load preprocessed schema into memory
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Privately maps the scaled Phase 2 Numpy output arrays natively."""
        try:
            self.X_train = np.load(PROCESSED_DATA_DIR / "X_train_scaled.npy")
            self.X_test = np.load(PROCESSED_DATA_DIR / "X_test_scaled.npy")
            self.y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
            self.y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
            
            logger.info(f"Modeling inputs loaded. Train shape: {self.X_train.shape} | Test shape: {self.X_test.shape}")
        except FileNotFoundError as e:
            logger.critical(f"FATAL: Missing mandatory Phase 2 processed artifacts: {e}")
            raise

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Mathematically scores the model utilizing Root Mean Squared Error and R-Squared logic.
        
        Args:
            y_true: The actual target arrays.
            y_pred: The algorithmic predictions mapped against the target.
            
        Returns:
            Dict[str, float]: A structured log of the prediction KPIs.
        """
        # RMSE utilizing the secure numpy sqrt algorithm wrapper
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        
        # MLflow SQLite backend fractures on literal NaN entries resulting from 1-sample Test environments
        if np.isnan(r2) or len(y_true) < 2:
            r2 = 0.0
            
        metrics = {
            "RMSE": rmse,
            "R2_Score": r2
        }
        
        logger.info(f"Model Evaluated | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        return metrics



if __name__ == "__main__":
    logger.info("BaseModelTrainer Evaluator module verified online.")
