"""
Feature Engineering module utilizing Polars.

Focuses on extracting critical numerical metrics (e.g. RAM, Storage capacity)
from raw unstructured specification strings.
"""
import polars as pl
from src.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """
    Engineers discrete, highly predictive features from cleaned smartphone datasets.
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize the engineer with a prepared dataframe.
        """
        self.df = df
        logger.info(f"Initialized FeatureEngineer with {len(self.df)} records.")

    def extract_hardware_capacities(self) -> 'FeatureEngineer':
        """
        Parse combined specification strings into discrete RAM and Storage features.
        
        Returns:
            FeatureEngineer: Self-reference for method chaining.
            
        Time Complexity: O(N) mapping Regex extractions across string columns.
        Space Complexity: O(N) generating two new numerical column dimensions.
        """
        try:
            if 'specs' in self.df.columns:
                self.df = self.df.with_columns(
                    # Example Extract: "8 GB RAM, 256 GB ROM" -> ram_gb: 8, storage_gb: 256
                    pl.col('specs').str.extract(r'(?i)(\d+)\s*GB\s*RAM').cast(pl.Int32, strict=False).alias('ram_gb'),
                    pl.col('specs').str.extract(r'(?i)(\d+)\s*GB\s*(?:ROM|Storage)').cast(pl.Int32, strict=False).alias('storage_gb')
                )
                logger.debug("Successfully extracted ram_gb and storage_gb features.")
            return self
        except Exception as e:
            logger.error(f"Error engineering hardware capacity features: {e}")
            raise

    def get_data(self) -> pl.DataFrame:
        """
        Retrieve the heavily engineered Polars DataFrame ready for modeling.
        """
        return self.df
