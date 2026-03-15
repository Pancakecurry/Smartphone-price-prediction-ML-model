"""
Data Cleaner module utilizing Polars for high-speed preprocessing.

Handles type casting, null value imputation, and dropping structurally invalid records.
"""
import polars as pl
from src.logger import get_logger

logger = get_logger(__name__)

class DataCleaner:
    """
    Polars-driven data cleaning pipeline to standardize raw smartphone records.
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize the cleaner with a specific dataset.
        
        Args:
            df (pl.DataFrame): The raw dataframe to be cleaned.
        """
        self.df = df
        logger.info(f"Initialized DataCleaner with {len(self.df)} records.")

    def cast_types(self) -> 'DataCleaner':
        """
        Cast explicit columns to robust data types (e.g., pricing to Float).
        
        Returns:
            DataCleaner: Self-reference for method chaining.
            
        Time Complexity: O(N) where N is the number of rows (highly optimized in Rust).
        Space Complexity: O(N) yielding new contiguous column allocations in memory.
        """
        try: # Utilizing try-except as specified by quality rules
            if 'price' in self.df.columns:
                self.df = self.df.with_columns(
                    pl.col('price')
                    .cast(pl.Utf8) # Ensure format stability before regex
                    .str.replace(r'[^\d.]', '', literal=False) # Strip out non-numeric characters like '$'
                    .cast(pl.Float64, strict=False)
                )
                logger.debug("Successfully cast 'price' column to Float64.")
            return self
        except Exception as e:
            logger.error(f"Error during type casting: {e}")
            raise # Re-raise for pipeline abort safety

    def handle_missing_values(self) -> 'DataCleaner':
        """
        Impute or drop critical missing data.
        
        Returns:
            DataCleaner: Self-reference for method chaining.
            
        Time Complexity: O(N) to scan and drop null sets.
        Space Complexity: O(N) where the mask vector requires allocation proportional to row count.
        """
        try:
            initial_count = len(self.df)
            # Ensure records have the minimal required prediction fields
            critical_cols = [c for c in ['model', 'price'] if c in self.df.columns]
            
            if critical_cols:
                self.df = self.df.drop_nulls(subset=critical_cols)
                
            dropped_count = initial_count - len(self.df)
            if dropped_count > 0:
                logger.debug(f"Dropped {dropped_count} rows containing critical nulls.")
                
            return self
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise

    def get_data(self) -> pl.DataFrame:
        """
        Retrieve the fully cleaned Polars DataFrame.
        """
        return self.df
