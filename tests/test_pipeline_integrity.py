"""
Comprehensive mathematical validation suite asserting Data Leakage boundaries and ML schema formatting.
Utilizes PyTest to load generated NumPy binaries strictly ensuring artifacts meet standard ML constraints.
"""
import numpy as np
import pytest
from src.config import PROCESSED_DATA_DIR


@pytest.fixture(scope="module")
def load_artifacts():
    """
    Load the Scikit-Learn transformed matrices natively into memory.
    Executes once per module to optimize performance overhead.
    """
    X_tr = np.load(PROCESSED_DATA_DIR / "X_train_scaled.npy")
    X_te = np.load(PROCESSED_DATA_DIR / "X_test_scaled.npy")
    y_tr = np.load(PROCESSED_DATA_DIR / "y_train.npy")
    y_te = np.load(PROCESSED_DATA_DIR / "y_test.npy")
    
    return X_tr, X_te, y_tr, y_te


def test_no_null_values(load_artifacts):
    """
    Proves KNNImputer neutralized all instances of missing network logic natively.
    Checks `np.isnan(target)` across matrices directly.
    """
    X_tr, X_te, _, _ = load_artifacts
    
    assert np.isnan(X_tr).sum() == 0, "Train matrix contains unhandled NaN occurrences."
    assert np.isnan(X_te).sum() == 0, "Test matrix contains unhandled NaN occurrences."


def test_standardization_proof(load_artifacts):
    """
    Validates Scikit-Learn StandardScaler transformations (z = x - μ / σ).
    Asserts training mean μ≈0 and standard deviation σ≈1 exclusively on the numerical columns.
    
    In the ColumnTransformer, numerical indices [0:3] correlate to `ram_gb`, `battery_mah`, and `camera_mp`.
    """
    X_tr, _, _, _ = load_artifacts
    
    # Isolate Numerics mapped natively sequence
    num_features_tr = X_tr[:, :3]
    
    # Standard Scikit-Learn validation limits
    tolerance = 1e-7
    mean_state = np.mean(num_features_tr, axis=0)
    std_state = np.std(num_features_tr, axis=0)
    
    assert np.allclose(mean_state, 0, atol=tolerance), f"Training Mean Failed Scaling: μ={mean_state}"
    assert np.allclose(std_state, 1, atol=tolerance), f"Training Std. Deviation Failed Scaling: σ={std_state}"


def test_prevented_data_leakage(load_artifacts):
    """
    Validates strict Test-Set mathematical isolation enforcing realistic ML assumptions.
    
    The scaler explicitly derived μ and σ from the Training vector. 
    Therefore, the testing vectors MUST map wildly varying limits reflecting 
    its structural isolation rather than exactly mapping to 0 natively.
    """
    _, X_te, _, _ = load_artifacts
    num_features_te = X_te[:, :3]
    
    test_means = np.mean(num_features_te, axis=0)
    
    # Assert structurally impossible for test variables to equate magically to zero training means
    assert not np.allclose(test_means, 0, atol=1e-7), "CRITICAL FATAL: Scale logic leaked directly into Target validation."


def test_matrix_symmetry(load_artifacts):
    """
    Verifies Feature logic translates 1:1 against its mapped supervised variable target list.
    
    X_train (Features) MUST align dimensions to y_train (Price), avoiding dropped parameters organically.
    """
    X_tr, X_te, y_tr, y_te = load_artifacts
    
    assert X_tr.shape[0] == y_tr.shape[0], "Training Matrix dropped sequences mapping to supervised targets."
    assert X_te.shape[0] == y_te.shape[0], "Testing Matrix dropped sequences mapping to supervised targets."
