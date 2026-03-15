"""
Data Ingestion Script for the Smartphone Price Prediction project.

Loads the production 'real_smartphones.csv' natively via Polars, sanitizes numeric 
features globally using Regex logic, and outputs the master dataset safely bypassing HTTP mocks.
"""
import polars as pl
from typing import List, Dict, Any
import os

from src.logger import get_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)


class SmartphoneDataIngestor:
    """
    Ingestor class parsing production CSV files into strict ML target schemas.
    """
    
    def __init__(self):
        """Bind global pathways natively to production files."""
        self.input_path = RAW_DATA_DIR / "real_smartphones.csv"
        self.output_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"

    def run(self) -> None:
        """
        Execute the synchronous production ingestion pipeline globally.
        """
        logger.info(f"Ingesting production dataset directly mapping to {self.input_path}")
        
        try:
            # 1. Load Raw Schema
            df = pl.read_csv(self.input_path)
            
            # Validate schema integrity globally
            expected_columns = ["Company Name", "Model Name", "Launched Price (USA)", "RAM", "Battery Capacity", "Back Camera", "Processor"]
            missing_cols = [col for col in expected_columns if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"CRITICAL FAULT: Production Schema rejected missing bound parameters -> {missing_cols}")
            
            logger.info("Schema integrity verified mathematically. Extracting Regex numerical values.")
            
            # 2. Extract strictly numeric limits natively scaling string properties
            regex_float_parser = r"([\d\,?\.]+)"
            
            df_cleaned = df.with_columns([
                # Target Identifiers
                pl.col("Company Name").alias("Brand"),
                pl.col("Model Name").alias("Smartphone_Name"),
                pl.col("Launched Price (USA)").str.replace_all(",", "").alias("_raw_price_str"),
                
                # Default ML Feature Extraction
                pl.col("RAM").str.extract(regex_float_parser).str.replace_all(",", "").cast(pl.Float64).alias("ram_gb"),
                pl.col("Battery Capacity").str.extract(regex_float_parser).str.replace_all(",", "").cast(pl.Float64).alias("battery_mah"),
                pl.col("Back Camera").str.extract(regex_float_parser).str.replace_all(",", "").cast(pl.Float64).alias("camera_mp"),
            ]).with_columns([
                # Explicit currency mapping logic identifying contaminating symbols natively
                pl.col("_raw_price_str").str.extract(regex_float_parser).cast(pl.Float64, strict=False).alias("_raw_amt"),
                pl.col("_raw_price_str").str.contains(r"₹").alias("_is_inr"),
                pl.col("_raw_price_str").str.contains(r"€").alias("_is_eur"),
                pl.col("_raw_price_str").str.contains(r"£").alias("_is_gbp"),
            ]).with_columns([
                # Apply exchange rates computationally 
                pl.when(pl.col("_is_inr")).then(pl.col("_raw_amt") / 83.0)
                  .when(pl.col("_is_eur")).then(pl.col("_raw_amt") * 1.1)
                  .when(pl.col("_is_gbp")).then(pl.col("_raw_amt") * 1.25)
                  .otherwise(pl.col("_raw_amt"))
                  .alias("Price")
            ]).select(
                # Enforce rigid mapping bounds discarding visual artifact columns 
                ["Brand", "Smartphone_Name", "Price", "ram_gb", "battery_mah", "camera_mp", "Processor"]
            )
            
            # 3. Clean Target Matrix Nulls structurally
            initial_count = len(df_cleaned)
            df_cleaned = df_cleaned.drop_nulls(subset=["Price"])
            
            # -----------------------------------------
            # Price Outlier Mitigation (Fallback & IQR)
            # -----------------------------------------
            # Currency Fallback
            df_cleaned = df_cleaned.with_columns(
                pl.when(pl.col("Price") > 2500)
                .then(pl.col("Price") / 83.0)
                .otherwise(pl.col("Price"))
                .alias("Price")
            )
            
            # IQR Calc
            q1 = df_cleaned.select(pl.col("Price").quantile(0.25)).item()
            q3 = df_cleaned.select(pl.col("Price").quantile(0.75)).item()
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            
            # Drop bounds outside realistic targets
            df_cleaned = df_cleaned.filter(
                (pl.col("Price") <= upper_bound) & (pl.col("Price") >= 50.0)
            )
            
            dropped = initial_count - len(df_cleaned)
            
            if dropped > 0:
                logger.warning(f"Discarded {dropped} invalid dataframe properties (Missing/Outlier Targets).")
            
            # Log Global Scale Extent
            metrics_min = df_cleaned['Price'].min()
            metrics_max = df_cleaned['Price'].max()
            logger.info(f"💰 Price Vector Fixed! Global Min: ${metrics_min:.2f} | Global Max: ${metrics_max:.2f}")
                
            # 4. Serialize to Parquet memory mapping
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            df_cleaned.write_parquet(self.output_path)
            
            logger.info(f"Successfully processed {len(df_cleaned)} valid smartphone records to {self.output_path}")

        except Exception as e:
            logger.critical(f"FATAL Unhandled Exception within Production Ingestion Engine bounds: {e}")
            raise RuntimeError(f"Data Ingestor Logical crash: {e}") from e


class LiveMarketScraper:
    """
    Scrapes live flagship smartphone prices from the web using requests + BeautifulSoup.

    Falls back gracefully to a hardcoded static dataset of current flagship models
    if the target site is unreachable, rate-limited, or blocks the request.
    """

    STATIC_FALLBACK: list = [
        # iPhone 16 series
        {"Brand": "Apple", "Smartphone_Name": "Apple iPhone 16 Pro Max 256GB", "Price": 1199.0, "ram_gb": 8.0, "battery_mah": 4685.0, "camera_mp": 48.0, "Processor": "A18 Pro"},
        {"Brand": "Apple", "Smartphone_Name": "Apple iPhone 16 Pro 256GB",     "Price": 999.0,  "ram_gb": 8.0, "battery_mah": 4422.0, "camera_mp": 48.0, "Processor": "A18 Pro"},
        {"Brand": "Apple", "Smartphone_Name": "Apple iPhone 16 128GB",          "Price": 799.0,  "ram_gb": 8.0, "battery_mah": 3561.0, "camera_mp": 48.0, "Processor": "A18"},
        {"Brand": "Apple", "Smartphone_Name": "Apple iPhone 15 Pro Max 256GB",  "Price": 1099.0, "ram_gb": 8.0, "battery_mah": 4422.0, "camera_mp": 48.0, "Processor": "A17 Pro"},
        # Samsung S24 series
        {"Brand": "Samsung", "Smartphone_Name": "Samsung Galaxy S24 Ultra 256GB", "Price": 1299.0, "ram_gb": 12.0, "battery_mah": 5000.0, "camera_mp": 200.0, "Processor": "Snapdragon 8 Gen 3"},
        {"Brand": "Samsung", "Smartphone_Name": "Samsung Galaxy S24+ 256GB",      "Price": 999.0,  "ram_gb": 12.0, "battery_mah": 4900.0, "camera_mp": 50.0,  "Processor": "Snapdragon 8 Gen 3"},
        {"Brand": "Samsung", "Smartphone_Name": "Samsung Galaxy S24 128GB",        "Price": 799.0,  "ram_gb": 8.0,  "battery_mah": 4000.0, "camera_mp": 50.0,  "Processor": "Snapdragon 8 Gen 3"},
        {"Brand": "Samsung", "Smartphone_Name": "Samsung Galaxy Z Fold 6 256GB",   "Price": 1899.0, "ram_gb": 12.0, "battery_mah": 4400.0, "camera_mp": 50.0,  "Processor": "Snapdragon 8 Gen 3"},
        # Google Pixel
        {"Brand": "Google", "Smartphone_Name": "Google Pixel 9 Pro XL 128GB", "Price": 1099.0, "ram_gb": 16.0, "battery_mah": 5060.0, "camera_mp": 50.0, "Processor": "Tensor G4"},
        {"Brand": "Google", "Smartphone_Name": "Google Pixel 9 Pro 128GB",    "Price": 999.0,  "ram_gb": 16.0, "battery_mah": 4700.0, "camera_mp": 50.0, "Processor": "Tensor G4"},
        {"Brand": "Google", "Smartphone_Name": "Google Pixel 9 128GB",        "Price": 799.0,  "ram_gb": 12.0, "battery_mah": 4700.0, "camera_mp": 50.0, "Processor": "Tensor G4"},
        # OnePlus
        {"Brand": "OnePlus", "Smartphone_Name": "OnePlus 12 256GB",  "Price": 799.0, "ram_gb": 12.0, "battery_mah": 5400.0, "camera_mp": 50.0, "Processor": "Snapdragon 8 Gen 3"},
        {"Brand": "OnePlus", "Smartphone_Name": "OnePlus 12R 256GB", "Price": 499.0, "ram_gb": 8.0,  "battery_mah": 5500.0, "camera_mp": 50.0, "Processor": "Snapdragon 8 Gen 1"},
        # Xiaomi
        {"Brand": "Xiaomi", "Smartphone_Name": "Xiaomi 14 Ultra 512GB", "Price": 999.0, "ram_gb": 16.0, "battery_mah": 5000.0, "camera_mp": 50.0, "Processor": "Snapdragon 8 Gen 3"},
        {"Brand": "Xiaomi", "Smartphone_Name": "Xiaomi 14 256GB",       "Price": 799.0, "ram_gb": 12.0, "battery_mah": 4610.0, "camera_mp": 50.0, "Processor": "Snapdragon 8 Gen 3"},
        # Sony
        {"Brand": "Sony", "Smartphone_Name": "Sony Xperia 1 VI 256GB",     "Price": 1299.0, "ram_gb": 12.0, "battery_mah": 5000.0, "camera_mp": 52.0, "Processor": "Snapdragon 8 Gen 3"},
        # Motorola
        {"Brand": "Motorola", "Smartphone_Name": "Motorola Edge 50 Pro 256GB", "Price": 599.0, "ram_gb": 12.0, "battery_mah": 4500.0, "camera_mp": 50.0, "Processor": "Snapdragon 7s Gen 2"},
        # Nothing
        {"Brand": "Nothing", "Smartphone_Name": "Nothing Phone 2a 256GB", "Price": 349.0, "ram_gb": 12.0, "battery_mah": 5000.0, "camera_mp": 50.0, "Processor": "MediaTek Dimensity 7200 Pro"},
        # Asus
        {"Brand": "Asus", "Smartphone_Name": "Asus ROG Phone 8 Pro 512GB", "Price": 1099.0, "ram_gb": 16.0, "battery_mah": 5500.0, "camera_mp": 50.0, "Processor": "Snapdragon 8 Gen 3"},
        # Vivo
        {"Brand": "Vivo", "Smartphone_Name": "Vivo X100 Pro 256GB", "Price": 899.0, "ram_gb": 16.0, "battery_mah": 5400.0, "camera_mp": 50.0, "Processor": "MediaTek Dimensity 9300"},
    ]

    TARGET_SCHEMA = ["Brand", "Smartphone_Name", "Price", "ram_gb", "battery_mah", "camera_mp", "Processor"]

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }

    def _build_fallback_df(self) -> pl.DataFrame:
        """Build a clean Polars DataFrame from the static fallback list."""
        return pl.DataFrame(self.STATIC_FALLBACK).with_columns([
            pl.col("Price").cast(pl.Float64),
            pl.col("ram_gb").cast(pl.Float64),
            pl.col("battery_mah").cast(pl.Float64),
            pl.col("camera_mp").cast(pl.Float64),
        ])

    def _attempt_live_scrape(self) -> pl.DataFrame:
        """
        Attempts to scrape GSMArena's top phones listing for names, then enriches
        with pricing cross-referenced from the static fallback where available.
        Returns a Polars DataFrame or raises an exception on failure.
        """
        import requests
        from bs4 import BeautifulSoup

        url = "https://www.gsmarena.com/search.php3?sAvailabilities=1&chk5G=selected"
        resp = requests.get(url, headers=self._headers, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        phones = soup.select("div.makers ul li")

        if not phones:
            raise ValueError("GSMArena returned no phone listings — page structure may have changed.")

        records = []
        for phone in phones[:20]:
            name_tag = phone.find("strong") or phone.find("span")
            if not name_tag:
                continue
            name = name_tag.get_text(strip=True)
            brand = name.split()[0] if name else "Unknown"
            records.append({
                "Brand": brand,
                "Smartphone_Name": name,
                "Price": None,
                "ram_gb": None,
                "battery_mah": None,
                "camera_mp": None,
                "Processor": "Unknown",
            })

        if not records:
            raise ValueError("Scraped 0 records — structure may have changed.")

        logger.info(f"LiveMarketScraper: Scraped {len(records)} phone names from GSMArena.")

        scraped_df = pl.DataFrame(records).with_columns([
            pl.col("Price").cast(pl.Float64),
            pl.col("ram_gb").cast(pl.Float64),
            pl.col("battery_mah").cast(pl.Float64),
            pl.col("camera_mp").cast(pl.Float64),
        ])

        # Enrich scraped names with prices from fallback where they match
        fallback_df = self._build_fallback_df()
        enriched = (
            scraped_df
            .join(
                fallback_df.select(["Smartphone_Name", "Price", "ram_gb", "battery_mah", "camera_mp"]),
                on="Smartphone_Name", how="left", suffix="_fb"
            )
            .with_columns([
                pl.coalesce(["Price", "Price_fb"]).alias("Price"),
                pl.coalesce(["ram_gb", "ram_gb_fb"]).alias("ram_gb"),
                pl.coalesce(["battery_mah", "battery_mah_fb"]).alias("battery_mah"),
                pl.coalesce(["camera_mp", "camera_mp_fb"]).alias("camera_mp"),
            ])
            .select(self.TARGET_SCHEMA)
            .drop_nulls(subset=["Price"])
        )

        if len(enriched) == 0:
            raise ValueError("Live scrape produced 0 priced records after enrichment — falling back.")

        return enriched

    def _sanitize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Enforces a strict data contract on any scraped or fallback DataFrame
        before it can be returned to callers or merged into the master dataset.

        Steps:
          1. Strip text suffixes ('GB', 'mAh', 'MP', whitespace) from numeric
             columns and cast to Float64.
          2. Cast Price to Utf8 so DataTransformer.normalize_currency() can
             safely run its symbol-detection regex (handles both string prices
             like '$1,199' and plain floats).
          3. Run DataTransformer.normalize_currency() to convert any remaining
             currency symbols (₹, €, £) to USD.
          4. Run DataTransformer.remove_price_outliers() for IQR clipping and
             the >$2500 INR fallback.
          5. Drop rows still missing a valid Price.
        """
        from src.data.data_preprocessing import DataTransformer

        logger.info("LiveMarketScraper: Enforcing strict data contract on scraped DataFrame...")

        # ── Step 1: Clean numeric columns ──────────────────────────────
        regex_num = r"([\d,\.]+)"
        for col_name in ["ram_gb", "battery_mah", "camera_mp"]:
            if col_name in df.columns:
                col_dtype = df[col_name].dtype
                if col_dtype == pl.Utf8 or col_dtype == pl.String:
                    df = df.with_columns(
                        pl.col(col_name)
                          .str.extract(regex_num)
                          .str.replace_all(",", "")
                          .cast(pl.Float64, strict=False)
                          .alias(col_name)
                    )
                else:
                    # Already numeric — just ensure it's Float64
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Float64, strict=False).alias(col_name)
                    )

        # ── Step 2: Cast Price to Utf8 so normalize_currency regex works ──
        if "Price" in df.columns:
            price_dtype = df["Price"].dtype
            if price_dtype != pl.Utf8 and price_dtype != pl.String:
                # Already a numeric — convert back to string representation
                df = df.with_columns(
                    pl.col("Price").cast(pl.Utf8).alias("Price")
                )

        # ── Steps 3–5: Run DataTransformer pipeline ────────────────────
        try:
            sanitized = (
                DataTransformer(df)
                .normalize_currency()
                .remove_price_outliers()
                .drop_missing_price()
                .get_data()
            )
            logger.info(
                f"LiveMarketScraper: Sanitization complete. "
                f"{len(sanitized)} clean records. "
                f"Price range: ${sanitized['Price'].min():.2f} — ${sanitized['Price'].max():.2f}"
            )
            return sanitized
        except Exception as e:
            logger.error(
                f"LiveMarketScraper: DataTransformer sanitization failed ({type(e).__name__}: {e}). "
                f"Returning raw DataFrame as-is to avoid data loss."
            )
            # Last-resort: at minimum ensure Price is Float64 and non-null
            return df.with_columns(
                pl.col("Price").cast(pl.Float64, strict=False).alias("Price")
            ).drop_nulls(subset=["Price"])

    def fetch_live_prices(self) -> pl.DataFrame:
        """
        Public method. Returns a sanitized Polars DataFrame of current flagship prices.

        Tries live web scraping first. Falls back to the hardcoded static dataset
        if scraping fails for any reason (network error, rate-limit, schema change).
        Both paths are passed through _sanitize() to enforce a strict data contract
        before the result can be merged into master_smartphones.parquet.
        """
        try:
            logger.info("LiveMarketScraper: Attempting live web data fetch from GSMArena...")
            df = self._attempt_live_scrape()
            logger.info(f"LiveMarketScraper: Live scrape succeeded — {len(df)} raw records fetched. Sanitizing...")
        except Exception as e:
            logger.warning(
                f"LiveMarketScraper: Live scrape failed ({type(e).__name__}: {e}). "
                f"Falling back to static flagship dataset ({len(self.STATIC_FALLBACK)} records)."
            )
            df = self._build_fallback_df()

        return self._sanitize(df)



if __name__ == "__main__":
    ingestor = SmartphoneDataIngestor()
    ingestor.run()
