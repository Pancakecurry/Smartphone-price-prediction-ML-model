"""
Data Preprocessing script utilizing Polars.

Contains the DataTransformer class responsible for cleaning
ingested smartphone data, handling null targets, imputing capacities,
and exporting a final Parquet file for ML inference operations.
"""
import polars as pl
from src.logger import get_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)

class DataTransformer:
    """
    Polars-driven data transformation pipeline.
    
    Accepts raw scraped smartphone variables and normalizes
    battery, RAM, and price fields via vectorized queries.
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize the transformer with a specific dataframe.
        
        Args:
            df (pl.DataFrame): The raw dataframe to be cleaned.
        """
        self.df = df
        logger.info(f"Initialized DataTransformer with {len(self.df)} records.")

    def drop_missing_price(self) -> 'DataTransformer':
        """
        Drop any rows that do not contain a target variable (Price).
        
        Returns:
            DataTransformer: Self-reference for method chaining.
            
        Time Complexity: O(N) to evaluate and drop null masks across rows.
        Space Complexity: O(N) where a new subset DataFrame is allocated.
        """
        initial_len = len(self.df)
        self.df = self.df.drop_nulls(subset=["Price"])
        dropped = initial_len - len(self.df)
        
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows due to lack of an explicit Price target.")
            
        return self

    def normalize_currency(self) -> 'DataTransformer':
        """
        Identify currency symbols and mathematically exchange everything directly to USD Float64.
        Converts ₹ (INR), € (EUR), and £ (GBP) natively.
        """
        try:
            self.df = self.df.with_columns(
                pl.col("Price").str.replace_all(",", "").alias("_raw_price_str")
            ).with_columns(
                pl.col("_raw_price_str").str.extract(r"([\d\.]+)").cast(pl.Float64, strict=False).alias("_raw_amt"),
                pl.col("_raw_price_str").str.contains(r"₹").alias("_is_inr"),
                pl.col("_raw_price_str").str.contains(r"€").alias("_is_eur"),
                pl.col("_raw_price_str").str.contains(r"£").alias("_is_gbp"),
            ).with_columns(
                pl.when(pl.col("_is_inr")).then(pl.col("_raw_amt") / 83.0)
                  .when(pl.col("_is_eur")).then(pl.col("_raw_amt") * 1.1)
                  .when(pl.col("_is_gbp")).then(pl.col("_raw_amt") * 1.25)
                  .otherwise(pl.col("_raw_amt"))
                  .alias("Price")
            ).drop(["_raw_price_str", "_raw_amt", "_is_inr", "_is_eur", "_is_gbp"])
            
            logger.info("Successfully normalized currencies to standardized USD base.")
            return self
        except Exception as e:
            logger.error(f"Failed normalizing currency values: {e}")
            raise

    def remove_price_outliers(self) -> 'DataTransformer':
        """
        Clip massive outliers caused by failed string conversion rates.
        Fallback INR conversions for Price > 2500, then drop using IQR and Min Bounds ($50).
        """
        try:
            # 1. Aggressive Currency Fallback
            self.df = self.df.with_columns(
                pl.when(pl.col("Price") > 2500)
                .then(pl.col("Price") / 83.0)
                .otherwise(pl.col("Price"))
                .alias("Price")
            )
            
            # 2. Strict Statistical Clipping (IQR)
            q1 = self.df.select(pl.col("Price").quantile(0.25)).item()
            q3 = self.df.select(pl.col("Price").quantile(0.75)).item()
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            
            initial_len = len(self.df)
            
            # 3. Drop Outliers
            self.df = self.df.filter(
                (pl.col("Price") <= upper_bound) & (pl.col("Price") >= 50.0)
            )
            
            dropped = initial_len - len(self.df)
            logger.info(f"Removed {dropped} price outliers. Min: ${self.df['Price'].min():.2f}, Max: ${self.df['Price'].max():.2f}")
            
            return self
        except Exception as e:
            logger.error(f"Failed removing price outliers: {e}")
            raise

    def standardize_features(self) -> 'DataTransformer':
        """
        Standardize string attributes into clean numerical integer/float columns.
        e.g., RAM ("8GB") -> 8
        e.g., Camera ("48 MP") -> 48
        e.g., Battery ("5000mAh") -> 5000
        
        Returns:
            DataTransformer: Self-reference for method chaining.
            
        Time Complexity: O(N) mapping Regex execution across dimensions.
        Space Complexity: O(N) generating continuous scaled output arrays.
        """
        try:
            self.df = self.df.with_columns(
                # Extract first integer matching block natively utilizing rust regex.
                pl.col("RAM").str.extract(r"(\d+)").cast(pl.Int32, strict=False).alias("ram_gb"),
                pl.col("Camera").str.extract(r"(\d+)").cast(pl.Int32, strict=False).alias("camera_mp"),
                pl.col("Battery").str.extract(r"(\d+)").cast(pl.Int32, strict=False).alias("battery_mah")
            )
            logger.debug("Successfully appended structured continuous vectors (ram_gb, camera_mp, battery_mah).")
            return self
        except Exception as e:
            logger.error(f"Failed standardizing core integer features. Ensure column existence: {e}")
            raise

    def impute_missing_battery(self) -> 'DataTransformer':
        """
        Fill missing Battery capacity parameters intelligently.
        
        Leverages Polars `.over()` context groups to impute utilizing
        the `median()` score restricted only to that model's Brand.
        
        Returns:
            DataTransformer: Self-reference for method chaining.
            
        Time Complexity: O(N log N) scaling proportional to partitioning and calculating grouped medians.
        """
        try:
            self.df = self.df.with_columns(
                pl.col("battery_mah")
                .fill_null(pl.col("battery_mah").median().over("Brand"))
            )
            logger.info("Imputed missing batteries scaled by Brand Median groupings.")
            return self
        except Exception as e:
            logger.error(f"Error computing grouped median battery metrics: {e}")
            raise

    def get_data(self) -> pl.DataFrame:
        """
        Return the fully processed dataframe payload.
        """
        return self.df


if __name__ == "__main__":
    logger.info("Starting Data Preprocessing Phase...")
    
    input_path = RAW_DATA_DIR / "raw_smartphones.csv"
    output_path = PROCESSED_DATA_DIR / "clean_smartphones.parquet"
    
    try:
        # Load Raw Mock Extracted CSV Pipeline
        raw_df = pl.read_csv(input_path)
        
        # Instantiate chaining and execute transformations sequentially
        transformer = (
            DataTransformer(raw_df)
            .normalize_currency()
            .drop_missing_price()
            .remove_price_outliers()
            .standardize_features()
            .impute_missing_battery()
        )
        
        cleaned_df = transformer.get_data()
        
        # Serialize to Parquet for optimized downstream Machine Learning inference
        cleaned_df.write_parquet(output_path)
        
        logger.info(f"Preprocessing fully completed. Final Dimensions: {cleaned_df.shape}")
        logger.info(f"Deserialized and cached locally to: {output_path}")
        
    except FileNotFoundError:
        logger.error(f"Could not find input file: {input_path}. Has Data Ingestion executed yet?")
    except Exception as e:
        logger.critical(f"FATAL extraction disruption executing preprocessing runtime: {e}")
