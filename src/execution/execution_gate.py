import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExecutionGate")

class ExecutionGate:
    def __init__(self, data_path="data/processed/test_predictions.parquet"):
        self.data_path = data_path
        self.df = None

    def load_signals(self):
        if not os.path.exists(self.data_path):
            logger.error("Test prediction matrix missing. Run the Alpha Model first.")
            return False
            
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.df)} predictive ticks for Gate verification.")
        return True

    def calculate_microstructure_filters(self):
        """Simulates Gate 1 (Spread) and Gate 2 (Volume)."""
        logger.info("Computing theoretical microstructure constraints over test window...")
        
        # 1. Spread Proxy (High - Low) / Close -> rolling 80th percentile threshold
        # In Indian markets, the initial/closing 15 mins have spread explosions.
        if "GOLDBEES.NS_High" in self.df.columns and "GOLDBEES.NS_Low" in self.df.columns:
            # We offset by 1e-8 to avoid pure zero spread
            spread_proxy = (self.df["GOLDBEES.NS_High"] - self.df["GOLDBEES.NS_Low"]) / self.df["GOLDBEES.NS_Close"].replace(0, 1e-8)
            # 36 bars = ~3 hours rolling history
            rolling_80_perc = spread_proxy.rolling(window=36, min_periods=10).quantile(0.80)
            
            # If current spread > recent 80th percentile constraint
            self.df["Spread_Violation"] = spread_proxy > rolling_80_perc
        else:
            self.df["Spread_Violation"] = False
            logger.warning("Missing OHLC components for Spread Calculation.")

        # 2. Volume Gate (Relative volume requirement for true trend initiation)
        if "Relative_Volume" in self.df.columns:
            self.df["Volume_Violation"] = self.df["Relative_Volume"] < 1.0  # Slightly relaxed from 1.25 to prevent complete blocking 
        else:
            self.df["Volume_Violation"] = False
            logger.warning("Missing Relative Volume components.")

    def execute_gate_logic(self):
        """Maps AI Signals strictly through the Go/No-Go conditions logging reason codes."""
        logger.info("Mapping Alpha signals against Microstructure conditions...")
        
        decisions = []
        
        for idx, row in self.df.iterrows():
            model_signal = row['Model_Signal']  # -1, 0, 1
            confidence = row['Model_Confidence']
            
            # 1. Zero signal or Zero confidence implies AI abstention instantly
            if model_signal == 0 or confidence == 0.0:
                decisions.append('NONE')
                continue
                
            # 2. Microstructure Safety Sequence
            if row.get('Spread_Violation', False):
                decisions.append('FAIL_SPREAD')
            elif row.get('Volume_Violation', False):
                decisions.append('FAIL_VOL')
            else:
                decisions.append('PASS')
                
        self.df['Gate_Decision'] = decisions
        
        distribution = self.df['Gate_Decision'].value_counts().to_dict()
        logger.info(f"Execution Gate Audit: {distribution}")

    def export_gated_signals(self, output_path="data/processed/gated_signals.parquet"):
        self.df.to_parquet(output_path, engine='pyarrow', index=True)
        logger.info(f"Safely secured strict execution directives -> {output_path}")

    def run(self):
        if self.load_signals():
            self.calculate_microstructure_filters()
            self.execute_gate_logic()
            self.export_gated_signals()

if __name__ == "__main__":
    gate = ExecutionGate()
    gate.run()
