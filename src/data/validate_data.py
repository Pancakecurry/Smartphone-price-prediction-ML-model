"""
Data Validation script ensuring ML-Ready dataset integrity.

This module utilizes Polars to assert strict boundary, typing,
and null-space conditions against the cleaned smartphone datasets prior
to execution in our regression models.
"""
import polars as pl
from src.config import PROCESSED_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)

class DataValidationError(ValueError):
    """Custom exceptions triggered during structured dataset invalidation."""
    pass


def validate_parquet(file_path: str = PROCESSED_DATA_DIR / "clean_smartphones.parquet") -> None:
    """
    Execute rigorous assertions against a cleaned Parquet file.
    Raises DataValidationError upon constraint breaches.
    
    Time Complexity: O(N) where bounded dataframe segments are scanned.
    """
    logger.info(f"Initiating Data Validation on payload: {file_path}")
    
    try:
        df = pl.read_parquet(file_path)
    except FileNotFoundError:
        raise DataValidationError(f"Processing halted. Could not find Parquet file at {file_path}")
        
    # Assertion 1: Total Row Count Constraints
    if len(df) == 0:
        raise DataValidationError("Row Constraint Failed: Dataset length resolved to zero bounds.")
        
    logger.debug(f"Row Length Assertion passed cleanly. Initialized {len(df)} rows.")

    # Assertion 2: Strict Null Bounding on Key Target/Feature pairings
    critical_columns = ["Price", "ram_gb"]
    for col in critical_columns:
        if col not in df.columns:
            raise DataValidationError(f"Schema Constraint Failed: Expected column '{col}' missing.")
            
        nulls = df[col].null_count()
        if nulls > 0:
            raise DataValidationError(f"Null Constraint Failed: Found {nulls} missing entries in '{col}'.")
            
    logger.debug("Null State Assertions natively passed cleanly. No leaks found.")

    # Assertion 3: Mathematical Bounds Verification on Price Target
    # Extracts all rows failing the generic bounds check (<= 0)
    failed_prices = df.filter(pl.col("Price") <= 0)
    if len(failed_prices) > 0:
        raise DataValidationError(f"Numeric Bounds Failed: Found {len(failed_prices)} objects mapped with Price <= 0.")
        
    logger.debug("Pricing Boundaries mapped optimally strictly > 0.")
    
    # Assertion 4: Numeric Encoding / Polars Typing Validation on Engineered Columns
    numeric_targets = ["Price", "ram_gb", "battery_mah"]
    for col in numeric_targets:
        if col in df.columns:
            data_type = df[col].dtype
            # Must strictly adhere to numeric encodings without falling back to string classes
            if not isinstance(data_type, (pl.Float64, pl.Float32, pl.Int64, pl.Int32)):
                raise DataValidationError(
                    f"Typing Constraints Failed: Target '{col}' extracted incorrectly typed as {data_type}."
                )

    logger.debug("Numeric typing strictly asserted natively as Machine Learning capable integers/floats.")
    logger.info("🎉 SUCCESS: Data Validation passed cleanly. Dataset respects all inference integrity constraints.")


if __name__ == "__main__":
    try:
        validate_parquet()
    except DataValidationError as e:
        logger.critical(f"PIPELINE BLOCKED: {e}")
    except Exception as e:
        logger.critical(f"FATAL SYSTEM ERROR during Data Validation: {e}")
