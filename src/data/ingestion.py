import os
import time
import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import yfinance as yf
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataIngestion")

class DataIngestor:
    """
    Handles data extraction, strict time alignment, rate limiting, and basic cleaning
    (missing value handling, IQR clipping) for the Gold ETF ML Trading System.
    """
    def __init__(self, 
                 twelvedata_api_key: Optional[str] = None,
                 interval: str = "5m", 
                 lookback_days: int = 7):
        """
        :param twelvedata_api_key: API key for twelvedata (can also use env TWELVE_DATA_API_KEY)
        :param interval: Granularity of data (e.g., '1m', '5m')
        :param lookback_days: Number of trailing days to fetch
        """
        self.interval = interval
        self.lookback_days = lookback_days
        self.twelvedata_api_key = twelvedata_api_key or os.environ.get("TWELVE_DATA_API_KEY")
        
        # Map yfinance intervals to twelvedata intervals
        self.td_interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min"
        }
        
    def fetch_yfinance_data(self, ticker: str, start_dt: datetime.datetime, end_dt: datetime.datetime) -> pd.DataFrame:
        """Fetches OHLCV data using yfinance."""
        logger.info(f"Fetching {ticker} from yfinance...")
        try:
            df = yf.download(ticker, start=start_dt, end=end_dt, interval=self.interval, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker} from yfinance.")
                return pd.DataFrame()
                
            # If multi-index columns (often returned by yf.download in recent versions)
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.levels[1]:
                    df = df.xs(ticker, axis=1, level=1)
                else:
                    df.columns = df.columns.droplevel(1)
                    
            df.index.name = "timestamp"
            
            # Standardize columns
            df = df.rename(columns={
                "Open": f"{ticker}_Open",
                "High": f"{ticker}_High",
                "Low": f"{ticker}_Low",
                "Close": f"{ticker}_Close",
                "Volume": f"{ticker}_Volume"
            }).drop(columns=["Adj Close"], errors='ignore')
            
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker} from yfinance: {e}")
            return pd.DataFrame()

    def fetch_twelvedata_data(self, symbol: str, start_dt: datetime.datetime, end_dt: datetime.datetime) -> pd.DataFrame:
        """Fetches Forex/Asset data from Twelve Data via REST API with strict rate limiting."""
        if not self.twelvedata_api_key:
            logger.error("Twelve Data API key is missing. Skipping Twelve Data fetch.")
            return pd.DataFrame()
            
        td_interval = self.td_interval_map.get(self.interval, "5min")
        logger.info(f"Fetching {symbol} from Twelve Data ({td_interval})...")
        
        # Convert times to string format expected by TD (YYYY-MM-DD HH:MM:SS)
        start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": td_interval,
            "start_date": start_str,
            "end_date": end_str,
            "apikey": self.twelvedata_api_key,
            "format": "JSON",
            "outputsize": 5000,
            "timezone": "UTC"
        }
        
        # Rate Limiting: 8 requests per minute on free tier limit. 
        # By sleeping here we ensure we play nicely regardless of loop frequency.
        logger.info("Applying rate limit delay for Twelve Data (8s)...")
        time.sleep(8) 
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "status" in data and data["status"] == "error":
                logger.error(f"Twelve Data error: {data.get('message')}")
                return pd.DataFrame()
                
            if "values" not in data:
                logger.warning(f"No values found for {symbol} in Twelve Data response.")
                return pd.DataFrame()
                
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            df.index.name = "timestamp"
            
            # Numeric conversion
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    
            prefix = symbol.replace("/", "")
            df = df.rename(columns={
                "open": f"{prefix}_Open",
                "high": f"{prefix}_High",
                "low": f"{prefix}_Low",
                "close": f"{prefix}_Close"
            })
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Twelve Data: {e}")
            return pd.DataFrame()

    def time_align_and_clean(self, df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Joins different dataframes based on strict time alignment, 
        handles missing values (via flags and ffill), and clips outliers contextually.
        """
        logger.info("Aligning dataframes on strict time index...")
        
        # 1. Standardize Timestamps (tz-naive UTC)
        aligned_dfs = {}
        for name, df in df_dict.items():
            if not df.empty:
                df_copy = df.copy()
                if df_copy.index.tz is not None:
                    df_copy.index = df_copy.index.tz_convert('UTC').tz_localize(None)
                aligned_dfs[name] = df_copy
            
        if not aligned_dfs:
            return pd.DataFrame()
            
        # 2. Outer Join on Timestamp
        merged_df = pd.concat(aligned_dfs.values(), axis=1, join="outer")
        merged_df = merged_df.sort_index()
        
        # Resample to strict intervals to ensure regular grid (e.g. strict 5-min intervals)
        pd_interval_map = {"1m": "1min", "5m": "5min", "15m": "15min"}
        resample_rule = pd_interval_map.get(self.interval, "5min")
        
        # Retain original range but ensure all intervals exist
        merged_df = merged_df.resample(resample_rule).asfreq()
        
        # 3. Missing Value Handling
        logger.info("Applying missing value flags and forward filling...")
        original_cols = merged_df.columns.tolist()
        
        for col in original_cols:
            # TODO: Currency & Unit Conversion Reminder
            # NOTE: GC=F is priced in USD/ounce, while Gold BeES is in INR/gram.
            # This conversion (USD-to-INR and Ounce-to-Gram) MUST be implemented in the 
            # downstream Feature Engineering stage before calculating Fair Value/Spreads.
            
            is_vol = "Volume" in col or "volume" in col.lower()
            
            # Apply missingness flag BEFORE fill
            merged_df[f"{col}_missing"] = merged_df[col].isna().astype(int)
            
            if not is_vol:
                # Forward fill price/indicator columns
                merged_df[col] = merged_df[col].ffill()
            else:
                # Missing volume indicates no trades -> pad with 0
                merged_df[col] = merged_df[col].fillna(0)
                
        # 4. Outlier Clipping (Rolling IQR-based, STRICTLY NO LOOK-AHEAD)
        logger.info("Applying rolling IQR clipping...")
        # Window size heuristic: ~1 day of data
        # For 5-min data (24h) = 288 bars. Using a more conservative 78 bars for typical trading day (6.5 hours).
        window = 78 if self.interval == "5m" else 390 
        
        for col in original_cols:
            if "volume" not in col.lower():
                rolling_q25 = merged_df[col].rolling(window=window, min_periods=max(10, window//4)).quantile(0.25)
                rolling_q75 = merged_df[col].rolling(window=window, min_periods=max(10, window//4)).quantile(0.75)
                iqr = rolling_q75 - rolling_q25
                upper_bound = rolling_q75 + 4.0 * iqr
                lower_bound = rolling_q25 - 4.0 * iqr
                
                # Use shift(1) to prevent using current candle to calculate its own bounds
                upper_bound = upper_bound.shift(1)
                lower_bound = lower_bound.shift(1)
                
                # Clip values contextually where bounds exist
                mask = ~(upper_bound.isna() | lower_bound.isna())
                merged_df.loc[mask, col] = merged_df.loc[mask, col].clip(lower=lower_bound[mask], upper=upper_bound[mask])

        # Drop entirely empty rows gracefully if any trailing
        return merged_df

    def fetch_all(self) -> pd.DataFrame:
        """Orchestrates pulling from all specific sources for the ETF."""
        end_dt = datetime.datetime.utcnow()
        start_dt = end_dt - datetime.timedelta(days=self.lookback_days)
        
        dfs = {}
        # Fetching yfinance Tickers
        dfs["golbees"] = self.fetch_yfinance_data("GOLDBEES.NS", start_dt, end_dt)
        dfs["mcx_proxy"] = self.fetch_yfinance_data("GC=F", start_dt, end_dt)
        
        # Fetching Twelve Data
        dfs["usdinr"] = self.fetch_twelvedata_data("USD/INR", start_dt, end_dt)
        
        # Time Aligned Return
        return self.time_align_and_clean(dfs)

if __name__ == "__main__":
    # Smoke Test logic
    print("Testing Ingestor Logic...")
    # NOTE: requires `TWELVE_DATA_API_KEY` exported in bash
    ingestor = DataIngestor(lookback_days=3, interval="5m")
    df_clean = ingestor.fetch_all()
    if not df_clean.empty:
        print(f"Final Data Shape: {df_clean.shape}")
        print("Data Snapshot:")
        print(df_clean.tail(2))
    else:
        print("No valid data produced (possibly missing API key or market closed).")
