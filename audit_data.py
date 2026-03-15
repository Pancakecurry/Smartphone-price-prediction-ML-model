import polars as pl
from src.config import PROCESSED_DATA_DIR

def audit_dataset():
    parquet_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"
    print(f"Loading dataset from: {parquet_path}")
    
    df = pl.read_parquet(parquet_path)
    
    print("\n--- Rows containing 'S24' ---")
    s24_df = df.filter(pl.col("Smartphone_Name").str.to_lowercase().str.contains("s24"))
    print(s24_df)
    
    print("\n--- Rows containing 'S22' or 'iPhone 13' (Top 5) ---")
    s22_iphone13_df = df.filter(
        pl.col("Smartphone_Name").str.to_lowercase().str.contains("s22") | 
        pl.col("Smartphone_Name").str.to_lowercase().str.contains("iphone 13")
    ).head(5)
    print(s22_iphone13_df)

if __name__ == "__main__":
    audit_dataset()
