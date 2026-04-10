import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TargetEngineering")

class TargetEngineer:
    def __init__(self, 
                 baseline_path: str = "data/processed/baseline_features.parquet",
                 features_path: str = "data/processed/engineered_features.parquet"):
        self.baseline_path = baseline_path
        self.features_path = features_path
        self.baseline_df = None
        self.features_df = None

    def load_data(self):
        """Loads both un-shifted truth data (to trace path) and shifted ML features."""
        if not os.path.exists(self.baseline_path) or not os.path.exists(self.features_path):
            logger.error(f"Missing data! Check {self.baseline_path} and {self.features_path}")
            return False
            
        self.baseline_df = pd.read_parquet(self.baseline_path)
        self.features_df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded Baseline OHLCV shape: {self.baseline_df.shape}")
        logger.info(f"Loaded ML Features shape: {self.features_df.shape}")
        return True

    def calculate_atr(self, window: int = 14):
        """
        Dynamically calculates Average True Range (ATR) to size the barriers.
        TR = max(H-L, |H-PrevC|, |L-PrevC|)
        """
        df = self.baseline_df
        logger.info(f"Calculating ATR (window={window}) for barrier scoping...")
        
        high_low = df['GOLDBEES.NS_High'] - df['GOLDBEES.NS_Low']
        high_close = np.abs(df['GOLDBEES.NS_High'] - df['GOLDBEES.NS_Close'].shift(1))
        low_close = np.abs(df['GOLDBEES.NS_Low'] - df['GOLDBEES.NS_Close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        # Expanding first, then roll to preserve early data sizing realistically
        self.baseline_df['ATR'] = true_range.rolling(window=window, min_periods=window//2).mean()

    def generate_triple_barrier_labels(self, pt_mul=2.0, sl_mul=1.0, horizon=12):
        """
        Scans future paths to resolve which barrier is pierced first.
        - pt_mul: Profit Target multiplier of ATR
        - sl_mul: Stop Loss multiplier of ATR
        - horizon: Side barrier limit (bars)
        """
        logger.info(f"Executing Triple Barrier scan (PT={pt_mul}x, SL={sl_mul}x, Horizon={horizon} bars)...")
        
        highs = self.baseline_df['GOLDBEES.NS_High'].values
        lows = self.baseline_df['GOLDBEES.NS_Low'].values
        closes = self.baseline_df['GOLDBEES.NS_Close'].values
        atrs = self.baseline_df['ATR'].values
        
        labels = np.zeros(len(closes), dtype=int)
        hit_times = np.zeros(len(closes), dtype=int)  # To track how fast things resolve
        
        # Numpy inner loop (fast execution for thousands of rows)
        for t in range(len(closes) - 1):
            if np.isnan(closes[t]) or np.isnan(atrs[t]) or atrs[t] <= 0:
                labels[t] = 0
                continue
                
            c_t = closes[t]
            pt_barrier = c_t + (pt_mul * atrs[t])
            sl_barrier = c_t - (sl_mul * atrs[t])
            
            end_idx = min(t + horizon + 1, len(closes))
            
            label = 0
            resolved_at = horizon
            
            for i in range(t + 1, end_idx):
                h_i = highs[i]
                l_i = lows[i]
                
                # Extreme volatility check — if both hit in same intraday bar, conservative stance is -1
                if l_i <= sl_barrier and h_i >= pt_barrier:
                    label = -1
                    resolved_at = i - t
                    break
                elif l_i <= sl_barrier:
                    label = -1
                    resolved_at = i - t
                    break
                elif h_i >= pt_barrier:
                    label = 1
                    resolved_at = i - t
                    break
                    
            labels[t] = label
            hit_times[t] = resolved_at
            
        self.baseline_df['Triple_Barrier_Signal'] = labels
        self.baseline_df['Barrier_Hit_Time'] = hit_times
        
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Signal Class Distribution: {dict(zip(unique, counts))}")

    def sync_and_export(self, output_path: str = "data/processed/model_ready_data.parquet"):
        """Joins the precise path-labels to the strictly shifted ML feature matrix."""
        logger.info("Merging Triple Barrier labels onto the look-ahead-protected feature matrix...")
        
        # We enforce a strict index join. 
        # features_df row T corresponds to features available to observe AT time T. 
        # baseline_df row T "Triple_Barrier_Signal" corresponds to exactly what happens entering AT time T's close.
        # This perfectly maps known-state -> forward outcome without leakage.
        
        cols_to_pull = ['Triple_Barrier_Signal', 'Barrier_Hit_Time', 'ATR']
        final_df = self.features_df.join(self.baseline_df[cols_to_pull], how='inner')
        
        # Drop boundary NaNs from early rolling windows
        final_df = final_df.dropna(subset=['Triple_Barrier_Signal'])
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_parquet(output_path, engine='pyarrow', index=True)
        
        logger.info(f"Fully assembled ML matrix seamlessly saved -> {output_path}")
        logger.info(f"Final combined shape for training: {final_df.shape}")
        
    def run_pipeline(self):
        if not self.load_data(): return
        self.calculate_atr()
        self.generate_triple_barrier_labels()
        self.sync_and_export()

if __name__ == "__main__":
    target_engineer = TargetEngineer()
    target_engineer.run_pipeline()
    
    # Just to provide a quick readout for the user when tested
    df = pd.read_parquet("data/processed/model_ready_data.parquet")
    print("\n--- Final Model-Ready Data (Sample) ---")
    print(df[['Returns_1_period', 'Triple_Barrier_Signal', 'Barrier_Hit_Time']].dropna().head(10))
