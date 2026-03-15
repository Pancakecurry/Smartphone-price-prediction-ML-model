"""
Model Validation Testing Suite.

Loads serialized artifacts from MLflow and tests the computational logic
bounding physical realities globally. Ensures models do not predict negative smartphone
prices and guarantees input/output tensors strictly match.
"""
import pytest
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from src.config import PROCESSED_DATA_DIR

# Establish consistent database query link locally
mlflow.set_tracking_uri('sqlite:///mlflow.db')

@pytest.fixture(scope="module")
def sample_test_data() -> np.ndarray:
    """
    Loads Phase 2 Feature Engineered matrices into memory.
    Caps payload at first 10 arrays representing active inference state.
    """
    try:
        X_test_scaled = np.load(PROCESSED_DATA_DIR / "X_test_scaled.npy")
        # Ensure we always retrieve exactly up to 10 rows (if dataset < 10, takes all)
        return X_test_scaled[:10]
    except FileNotFoundError:
        pytest.fail("Cannot locate X_test_scaled.npy. Phase 2 must be run first.")

@pytest.fixture(scope="module")
def trained_rf_model():
    """
    Dynamically searches the local MLflow SQLite registry for the most recently optimized
    Random Forest Scikit-Learn binary instance.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Smartphone_Price_Prediction")
    
    if not experiment:
        pytest.fail("MLflow Experiment not found. Please execute run_training.py.")
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'RandomForest_Optuna_Optimized'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        pytest.fail("No RandomForest run artifacts located inside MLflow UI database.")
        
    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/model"
    
    model = mlflow.sklearn.load_model(model_uri)
    return model

def test_prediction_output_shape(trained_rf_model, sample_test_data):
    """
    Test 1 (Output Shape): 
    Validates array inference mapping. Passes `N` rows into the Scikit-Learn `.predict()`
    method and sequentially asserts we strictly receive exactly `N` independent predictions.
    This protects against algorithms dropping rows silently.
    """
    predictions = trained_rf_model.predict(sample_test_data)
    
    # Assert Exact Dimensionality Maps 
    assert len(predictions) == len(sample_test_data), \
        f"Input map failure. Provided {len(sample_test_data)} rows but received {len(predictions)} predictions."

def test_physical_reality_constraints(trained_rf_model, sample_test_data):
    """
    Test 2 (Physical Reality Check): 
    Smartphones categorically cannot have a negative price globally.
    Evaluates every single Float array output ensuring its bounds are strictly >= 0.
    """
    predictions = trained_rf_model.predict(sample_test_data)
    
    # Mathematics bounds checks
    for i, price in enumerate(predictions):
        assert price >= 0, \
            f"Physical Impossibility Exception: Model predicted negative price (${price:.2f}) at index {i}."

def test_prediction_strict_typing(trained_rf_model, sample_test_data):
    """
    Test 3 (Type Check): 
    Inference pipelines often fail downstream when serializers silently stringify outputs.
    Guarantees every output node inherently maps to a standard NumPy Float/Int Numeric Type recursively.
    """
    predictions = trained_rf_model.predict(sample_test_data)
    
    # Check underlying array structure properties natively mapped against np.number primitive
    assert np.issubdtype(predictions.dtype, np.number), \
        f"Strict Type parsing Error. Output mapped incorrectly to non-computational representation: {predictions.dtype}."
