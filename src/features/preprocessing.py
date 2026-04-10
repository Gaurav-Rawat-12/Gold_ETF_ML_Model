import os
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataEngineering")

class DataPreprocessor:
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        self.df = df
        self.data_path = data_path
        
    def load_data(self):
        """Loads data from the ingestion module if dataframe not passed."""
        if self.df is None:
            from src.data.ingestion import DataIngestor
            logger.info("No DataFrame provided, initializing DataIngestor (fetching last 10 days to ensure enough window)...")
            ingestor = DataIngestor(lookback_days=10, interval="5m")
            self.df = ingestor.fetch_all()
        return self.df

    def process_currency_and_units(self):
        """
        Converts GC=F (USD/Ounce) to equivalent INR/Gram to naturally compare with Gold BeES.
        Conversion scale: 1 Troy Ounce = 31.1034768 Grams. 1/31.103 = 0.03215.
        Actually, the Master Outline precisely defined fair value scalar: 0.0311
        mcx_equivalent_inr_gram = GC=F_Close(USD/Oz) * USDINR_Close * 0.0311
        """
        if self.df is None or self.df.empty:
            logger.warning("Empty dataframe, skipping conversion.")
            return

        logger.info("Applying Unit & Currency conversions (USD/Ounce -> INR/Gram)...")
        # Ensure we have required columns. USD/INR mapped previously to USDINR
        req_cols = ["GC=F_Close", "USDINR_Close"]
        if all(c in self.df.columns for c in req_cols):
            self.df["GC=F_INR_Gram_Close"] = self.df["GC=F_Close"] * self.df["USDINR_Close"] * 0.0311
            self.df["GC=F_INR_Gram_Open"] = self.df["GC=F_Open"] * self.df["USDINR_Open"] * 0.0311
            self.df["GC=F_INR_Gram_High"] = self.df["GC=F_High"] * self.df["USDINR_High"] * 0.0311
            self.df["GC=F_INR_Gram_Low"] = self.df["GC=F_Low"] * self.df["USDINR_Low"] * 0.0311
            logger.info("Successfully calculated GC=F_INR_Gram metrics.")
        else:
            logger.warning(f"Missing one of {req_cols} for conversion. Check TwelveData API Key or yfinance state. Found cols: {self.df.columns.tolist()}")

    def rolling_z_score(self, columns: List[str], window: int = 288):
        """
        Applies strict rolling z-score normalization without look-ahead bias.
        288 periods @ 5 min = 24h equivalent.
        """
        logger.info(f"Applying rolling z-score normalization (window={window})...")
        for col in columns:
            if col in self.df.columns:
                # Calculate trailing statistics
                rolling_mean = self.df[col].rolling(window=window, min_periods=window//4).mean()
                rolling_std = self.df[col].rolling(window=window, min_periods=window//4).std()
                
                # Shift by 1 purely to be paranoid about intraday leakage (prevents current candle contaminating its own bound)
                rolling_mean = rolling_mean.shift(1)
                rolling_std = rolling_std.shift(1)
                
                # Calculate safe z-score with epsilon protecting flat markets
                z_score = (self.df[col] - rolling_mean) / (rolling_std + 1e-8)
                
                # Outlier clipping at +- 4 standard deviations securely integrated 
                self.df[f"{col}_zscore"] = z_score.clip(-4.0, 4.0)

    def save_parquet(self, filename: str = "data/processed/baseline_features.parquet"):
        """Saves output to parquet securely, creating directories if needed."""
        if self.df is None or self.df.empty:
            logger.error("Dataframe empty, skipping save.")
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # We explicitly enforce pyarrow engine as CSV loses high-precision datetime index structure.
        self.df.to_parquet(filename, engine='pyarrow', index=True)
        logger.info(f"Data reliably saved to {filename}")


class WalkForwardValidator:
    """Implement Purge and Embargo logic for ML cross-validation testing explicitly targeting leakage."""
    def __init__(self, purge_bars: int, embargo_bars: int):
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def generate_splits(self, total_len: int, n_splits: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates safely isolated (train_idx, test_idx) arrays.
        Enforces: Train -> [Purge Gap] -> [Embargo Gap] -> Test
        """
        indices = np.arange(total_len)
        fold_size = total_len // (n_splits + 1)
        
        for i in range(1, n_splits + 1):
            train_end = fold_size * i
            
            # The gaps drop indices functionally to prevent stationary crossover and label horizons
            test_start = train_end + self.purge_bars + self.embargo_bars
            test_end = test_start + fold_size
            
            if test_start >= total_len:
                break
                
            test_end = min(test_end, total_len)
            
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            
            yield train_idx, test_idx


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    
    if df is not None and not df.empty:
        preprocessor.process_currency_and_units()
        
        # Determine valid dynamic columns
        cols_to_zscore = ["GOLDBEES.NS_Close", "GOLDBEES.NS_Volume"]
        if "GC=F_INR_Gram_Close" in df.columns:
            cols_to_zscore.append("GC=F_INR_Gram_Close")
            
        # Standardizing (78 bars roughly = 1 typical trading day at 5-min tracking limits)
        preprocessor.rolling_z_score(columns=cols_to_zscore, window=78)
        
        preprocessor.save_parquet()
        
        print(f"\nFinal Engine Shape Produced: {preprocessor.df.shape}")
        
        print("\n--- Purge/Embargo Validation Split Generator Sim ---")
        validator = WalkForwardValidator(purge_bars=6, embargo_bars=12) # ~30 min purge, ~1 hr embargo
        splits = list(validator.generate_splits(total_len=len(df), n_splits=4))
        for idx, (tr, te) in enumerate(splits):
            print(f"Fold {idx+1}: Train Length: {len(tr)} | [Gap Size: {validator.purge_bars + validator.embargo_bars}] | Test Length: {len(te)}")
