"""
Phase 3 Centralized Run Execution Pipeline.

Initializes the MLflow Engine, instantiates rigorous Bayesian Search Optimization algorithms,
and computationally verifies a Multi-Layer PyTorch FFNN in sequential tracking boundaries.
"""
import sys
from src.logger import get_logger
from src.models.tree_models import TreeModelTrainer
from src.models.deep_learning_model import DeepLearningTrainer

logger = get_logger(__name__)

def execute_training() -> None:
    """Orchestrates comprehensive execution of Optuna searches crossing all structural dimensions."""
    logger.info("=" * 60)
    logger.info("       🚀 INITIATING PHASE 3: MODEL TRAINING PIPELINE 🚀")
    logger.info("=" * 60)
    
    try:
        # Load matrices spanning tracking contexts implicitly during init
        tree_trainer = TreeModelTrainer()
        dl_trainer = DeepLearningTrainer()
        
        # 1. Random Forest → Optuna 2D Hyperparameters Grid
        logger.info("\n---> Initiating Random Forest (Optuna Cross-Validated)")
        rf_metrics = tree_trainer.train_random_forest(n_trials=5)
        logger.info(f"✅ Random Forest Locked | RMSE: {rf_metrics['RMSE']:.4f} | R²: {rf_metrics['R2_Score']:.4f}")

        # 2. XGBoost → Optuna 3D Parameters Surface
        logger.info("\n---> Initiating XGBoost (Optuna Cross-Validated)")
        xgb_metrics = tree_trainer.train_xgboost(n_trials=5)
        logger.info(f"✅ XGBoost Locked | RMSE: {xgb_metrics['RMSE']:.4f} | R²: {xgb_metrics['R2_Score']:.4f}")

        # 3. ✅ Unified Pipeline: ColumnTransformer → RandomForest (PRODUCTION ARTIFACT)
        logger.info("\n---> Initiating Unified RF Pipeline (eliminates inference scaling mismatch)")
        pipe_metrics = tree_trainer.train_random_forest_pipeline(n_trials=5)
        logger.info(f"✅ Unified Pipeline Locked | RMSE: {pipe_metrics['RMSE']:.4f} | R²: {pipe_metrics['R2_Score']:.4f}")

        
        # 3. PyTorch FFNN -> 100 Epoch Gradient Decent Bound explicitly
        logger.info("\n---> Initiating PyTorch Feed Forward Neural Network (100 Epochs)")
        ffnn_metrics = dl_trainer.train_network()
        logger.info(f"✅ FFNN Sequence Locked | RMSE: {ffnn_metrics['RMSE']:.4f} | R\u00b2: {ffnn_metrics['R2_Score']:.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL MODELS MATHEMATICALLY OPTIMIZED & CACHED IN MLFLOW 🎉")
        print("=" * 60)
        print("\nTo view your isolated validation trials, Hyperparameters, and artifact binaries:")
        print("👉 Execute:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print("👉 Visit:    http://127.0.0.1:5000\n")
        
    except Exception as e:
        logger.critical(f"FATAL: Phase 3 Modeling Sequence collapsed irrecoverably - {e}")
        sys.exit(1)

if __name__ == "__main__":
    execute_training()
