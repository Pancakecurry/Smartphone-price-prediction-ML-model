"""
Unit tests for the Data Transformation module utilizing Polars.

Validates the DataCleaner and FeatureEngineer classes to ensure type casting,
missing value imputation, and regex-powered feature extraction behave correctly.
"""
import pytest
import polars as pl

from src.transform.cleaner import DataCleaner
from src.transform.engineer import FeatureEngineer

@pytest.fixture
def raw_dataframe():
    """Fixture providing a mock Polars dataframe with noisy data."""
    return pl.DataFrame({
        "model": ["Phone A", "Phone B", None, "Phone D"],
        "price": ["$999.99", "899.50", "1200", None],
        "specs": [
            "8 GB RAM, 256 GB ROM, Snapdragon",
            "12GB RAM, 512GB Storage, Exynos",
            "4 GB RAM, 64 GB Storage",
            "No specs listed"
        ]
    })


def test_cleaner_cast_types(raw_dataframe):
    """
    Test DataCleaner strips currency symbols and casts Strings to numeric Floats.
    
    Time Complexity: O(N) where N is the length of the mock array.
    """
    cleaner = DataCleaner(raw_dataframe).cast_types()
    df = cleaner.get_data()
    
    assert df["price"].dtype == pl.Float64
    assert df["price"][0] == 999.99
    assert df["price"][1] == 899.50
    assert df["price"][2] == 1200.0


def test_cleaner_handle_missing_values(raw_dataframe):
    """
    Test DataCleaner drops rows missing 'model' or 'price'.
    """
    cleaner = DataCleaner(raw_dataframe).handle_missing_values()
    df = cleaner.get_data()
    
    # Original len was 4. Row index 2 missing model (None), Row index 3 missing price (None).
    # Resulting table should only have the first 2 fully populated rows.
    assert len(df) == 2
    assert df["model"][0] == "Phone A"
    assert df["model"][1] == "Phone B"


def test_feature_engineer_extract_hardware():
    """
    Test FeatureEngineer successfully extracts discrete RAM and Storage metrics via Regex.
    """
    df = pl.DataFrame({
        "specs": ["8 GB RAM, 256 GB ROM", "12GB RAM, 512GB Storage", "Invalid string"]
    })
    
    engineer = FeatureEngineer(df).extract_hardware_capacities()
    result = engineer.get_data()
    
    assert "ram_gb" in result.columns
    assert "storage_gb" in result.columns
    
    assert result["ram_gb"][0] == 8
    assert result["storage_gb"][0] == 256
    
    assert result["ram_gb"][1] == 12
    assert result["storage_gb"][1] == 512
    
    assert result["ram_gb"][2] is None
    assert result["storage_gb"][2] is None
