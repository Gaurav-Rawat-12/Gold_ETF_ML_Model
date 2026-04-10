import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProfessionalBacktest")

class BacktestEngine:
    def __init__(self, data_path="data/processed/gated_signals.parquet"):
        self.data_path = data_path
        self.df = None
        self.slippage_tax_penalty = 0.0005  # 0.05% flat hit per execution round-trip

    def load_data(self):
        if not os.path.exists(self.data_path):
            logger.error("Gated Signals dataset is missing!")
            return False
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.df)} gated rows for sequence backtesting.")
        return True

    def _resolve_trade_return(self, row):
        """Determines the exact arithmetic geometric return tracking both Long and Short trades."""
        c_t = row['GOLDBEES.NS_Close']
        atr = row['ATR']
        truth_label = row['Triple_Barrier_Signal']
        timeout_ret = row['Forward_Return_12']
        trade_dir = row['Model_Signal']
        
        pt_scalar = (2.0 * atr) / c_t
        sl_scalar = -(1.0 * atr) / c_t
        
        if trade_dir == 1.0:  # LONG
            if truth_label == 1: raw_ret = pt_scalar
            elif truth_label == -1: raw_ret = sl_scalar
            else: raw_ret = timeout_ret
        else: # SHORT (-1.0)
            if truth_label == -1: raw_ret = abs(sl_scalar) # Direction was down -> Short Profit
            elif truth_label == 1: raw_ret = -abs(pt_scalar) # Direction was up -> Short Loss
            else: raw_ret = -timeout_ret
            
        net_ret = raw_ret - self.slippage_tax_penalty
        return net_ret, raw_ret

    def simulate(self):
        """Processes the true path execution filtering out blocked paths."""
        logger.info(f"Starting stringent PnL simulation (Penalty = {self.slippage_tax_penalty*100}% per execution)...")
        
        capital = 100000.0  # Baseline 1 Lakh INR
        equity_curve = [capital]
        timestamps = [self.df.index[0]]
        
        trade_logs = []
        
        for idx, row in self.df.iterrows():
            if row['Gate_Decision'] == 'PASS' and row['Model_Signal'] != 0.0: # Long & Short
                net_ret, raw_ret = self._resolve_trade_return(row)
                
                capital *= (1.0 + net_ret)
                decision_slippage = raw_ret - net_ret
                
                trade_logs.append({
                    'Timestamp': idx,
                    'Model_Signal': row['Model_Signal'],
                    'Gate_Decision': row['Gate_Decision'],
                    'Confidence': row['Model_Confidence'],
                    'Raw_Return': raw_ret,
                    'Net_Realized_Return': net_ret,
                    'Decision_Slippage_Sunk': decision_slippage,
                    'Capital_State': capital
                })
                
                equity_curve.append(capital)
                timestamps.append(idx)
        
        self.equity_series = pd.Series(equity_curve, index=timestamps)
        self.trade_logs_df = pd.DataFrame(trade_logs)
        
        logger.info(f"Simulation completed. {len(self.trade_logs_df)} trades fully allowed through structural gates.")
        if not self.trade_logs_df.empty:
            logger.info(f"Final Capital Balance: ₹{capital:,.2f}")

    def plot_equity_and_drawdown(self, out_path="artifacts/equity_curve.png"):
        """Maps out the physical reality of the trading path avoiding fantasy inflations."""
        if len(self.equity_series) <= 1:
            logger.warning("Not enough trades executed to construct a meaningful plot.")
            return
            
        sns.set_theme(style="darkgrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # 1. Equity Curve
        self.equity_series.plot(ax=ax1, color='green', linewidth=2)
        ax1.set_title("Strategy Realistic Equity Curve (Incl. Slippage & Taxes)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Portfolio Value (INR)")
        
        # 2. Drawdown (Underwater Profile)
        peak = self.equity_series.cummax()
        drawdown = (self.equity_series - peak) / peak
        drawdown.plot(ax=ax2, color='red', kind='area', alpha=0.3)
        ax2.set_title("Underwater Drawdown Profile", fontsize=12)
        ax2.set_ylabel("Drawdown %")
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        logger.info(f"Equity map rendered securely to -> {out_path}")

    def export_logs(self, out_path="data/logs/trading_log.parquet"):
        if self.trade_logs_df.empty: return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.trade_logs_df.to_parquet(out_path, engine='pyarrow', index=False)
        logger.info(f"Strict session audit logs decoupled -> {out_path}")

    def run(self):
        if not self.load_data(): return
        self.simulate()
        self.plot_equity_and_drawdown()
        self.export_logs()

if __name__ == "__main__":
    bt = BacktestEngine()
    bt.run()
