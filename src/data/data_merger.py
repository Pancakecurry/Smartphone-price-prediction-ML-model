"""
Integration module merging disparate smartphone datasets using Fuzzy String Matching
and an Upsert strategy for live market data.

Utilizes RapidFuzz Levenshtein computations to join Amazon-style pricing data
with GSMArena-style technical specifications by intelligently scoring name mappings.
Supports a secondary upsert path that merges a live-scraped DataFrame into the
historical master, overwriting stale prices for known models and inserting new ones.
"""
import polars as pl
from rapidfuzz import process, fuzz
from typing import Optional
from src.logger import get_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)

class DatasetIntegrator:
    """
    Polars architecture executing high-speed approximate string joins and
    live-data upserts against the historical master dataset.
    """

    def __init__(self, df_specs: pl.DataFrame, df_prices: pl.DataFrame, match_threshold: float = 85.0):
        """
        Initialize the merger with separated domain Dataframes.

        Args:
            df_specs (pl.DataFrame): Technical specs containing 'Smartphone_Name'.
            df_prices (pl.DataFrame): E-commerce pricing containing 'Smartphone_Name' and 'Price'.
            match_threshold (float): Minimum Levenshtein proximity (0-100) to permit a join.
        """
        self.df_specs = df_specs
        self.df_prices = df_prices
        self.match_threshold = match_threshold

        # Verify required join keys exist
        for df, source in zip([self.df_specs, self.df_prices], ["Specs", "Prices"]):
            if "Smartphone_Name" not in df.columns:
                raise ValueError(f"Core join key 'Smartphone_Name' missing in {source} dataset.")

        if "Price" not in self.df_prices.columns:
            raise ValueError("Target entity 'Price' missing in Prices dataset.")

        logger.info(f"Initialized Integrator. Specs Length: {len(self.df_specs)} | Prices Length: {len(self.df_prices)}")

    def execute_fuzzy_join(self) -> pl.DataFrame:
        """
        Generate unified dataset mapping optimum Prices to Specs conditionally.

        Operates by looping the target spec names through rapidfuzz.process.extractOne
        against all e-commerce namings, injecting the mapped price if score > threshold.

        Returns:
            pl.DataFrame: Merged entity context cleanly typed.
        """
        spec_names = self.df_specs["Smartphone_Name"].to_list()
        price_names = self.df_prices["Smartphone_Name"].to_list()
        price_values = self.df_prices["Price"].to_list()

        matched_prices = []
        successful_merges = 0

        logger.info(f"Initiating Heavy C++ Levenshtein proximity calculations. Target Bounds: > {self.match_threshold}")

        for spec_name in spec_names:
            match = process.extractOne(
                query=spec_name,
                choices=price_names,
                scorer=fuzz.WRatio
            )

            if match and match[1] >= self.match_threshold:
                matched_prices.append(price_values[match[2]])
                successful_merges += 1
            else:
                matched_prices.append(None)

        merge_rate = (successful_merges / len(self.df_specs)) * 100
        logger.info(f"Fuzzy Compute Complete. Successfully merged {successful_merges} entities.")
        logger.info(f"Integration Yield Rate: {merge_rate:.2f}% mappings found above threshold.")

        merged_df = self.df_specs.with_columns(
            pl.Series(name="Price", values=matched_prices, dtype=pl.Float64)
        )

        return merged_df

    def upsert_live_data(
        self,
        historical_df: pl.DataFrame,
        live_df: pl.DataFrame,
        output_path=None
    ) -> pl.DataFrame:
        """
        Merges a live-scraped DataFrame into the historical master using Upsert logic.

        Strategy:
          - UPDATE: If a phone from live_df already exists in historical_df (exact
            Smartphone_Name match), overwrite its Price with the fresh live price.
          - INSERT: If the phone is entirely new, append it as a new row.

        Falls back gracefully to the unchanged historical_df if live_df is empty
        or any exception occurs during the merge.

        Args:
            historical_df (pl.DataFrame): Existing master dataset.
            live_df      (pl.DataFrame): New DataFrame from LiveMarketScraper.
            output_path  (Path | None):  If provided, writes result to Parquet.

        Returns:
            pl.DataFrame: Unified dataset after upsert.
        """
        if live_df is None or len(live_df) == 0:
            logger.warning("upsert_live_data: live_df is empty — returning historical data unchanged.")
            return historical_df

        try:
            logger.info(
                f"Upsert: historical={len(historical_df)} rows | "
                f"live={len(live_df)} rows incoming."
            )

            historical_names = set(historical_df["Smartphone_Name"].to_list())
            live_names       = set(live_df["Smartphone_Name"].to_list())

            # ── UPDATE existing rows ──────────────────────────────────────
            # For matching names, replace the Price in historical_df
            # We do a left-join on Smartphone_Name, then coalesce prices.
            live_price_map = live_df.select(["Smartphone_Name", "Price"]).rename({"Price": "Live_Price"})

            updated_historical = (
                historical_df
                .join(live_price_map, on="Smartphone_Name", how="left")
                .with_columns(
                    pl.coalesce(["Live_Price", "Price"]).alias("Price")
                )
                .drop("Live_Price")
            )

            updated_count = len(historical_names & live_names)
            logger.info(f"Upsert UPDATE: refreshed prices for {updated_count} existing models.")

            # ── INSERT entirely new phones ────────────────────────────────
            new_names   = live_names - historical_names
            new_rows_df = live_df.filter(pl.col("Smartphone_Name").is_in(list(new_names)))

            # Align schema: only keep columns present in historical_df
            common_cols = [c for c in historical_df.columns if c in new_rows_df.columns]
            new_rows_aligned = new_rows_df.select(common_cols)

            # Fill any columns missing from live_df with null
            for col in historical_df.columns:
                if col not in new_rows_aligned.columns:
                    new_rows_aligned = new_rows_aligned.with_columns(
                        pl.lit(None).cast(historical_df[col].dtype).alias(col)
                    )
            new_rows_aligned = new_rows_aligned.select(historical_df.columns)

            unified_df = pl.concat([updated_historical, new_rows_aligned], how="diagonal")
            logger.info(
                f"Upsert INSERT: added {len(new_rows_df)} new models. "
                f"Final dataset size: {len(unified_df)} rows."
            )

            if output_path is not None:
                unified_df.write_parquet(output_path)
                logger.info(f"Upsert result persisted to: {output_path}")

            return unified_df

        except Exception as e:
            logger.error(
                f"upsert_live_data failed ({type(e).__name__}: {e}). "
                f"Falling back to unmodified historical dataset."
            )
            return historical_df


if __name__ == "__main__":
    from src.data.data_ingestion import LiveMarketScraper

    output_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"

    logger.info("=== Multi-Source Data Merger Demo ===")

    # 1. Load historical master
    try:
        historical_df = pl.read_parquet(output_path)
        logger.info(f"Loaded historical dataset: {len(historical_df)} rows.")
    except FileNotFoundError:
        logger.error(f"master_smartphones.parquet not found. Run data_ingestion.py first.")
        raise

    # 2. Fetch live market prices (with graceful fallback built-in)
    scraper = LiveMarketScraper()
    live_df = scraper.fetch_live_prices()

    # 3. Upsert live data into historical master
    integrator = DatasetIntegrator(historical_df, historical_df)  # fuzzy join not needed here
    unified_df = integrator.upsert_live_data(historical_df, live_df, output_path=output_path)

    logger.info(f"Pipeline complete. Final master dataset: {len(unified_df)} rows.")
    logger.info(f"Price range: ${unified_df['Price'].min():.2f} — ${unified_df['Price'].max():.2f}")

