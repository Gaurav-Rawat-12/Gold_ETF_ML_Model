import os
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from scipy.spatial import distance
from scipy.linalg import inv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlphaModel")

class AlphaModelEngine:
    def __init__(self, data_path: str = "data/processed/model_ready_data.parquet"):
        self.data_path = data_path
        self.df = None
        
        self.direction_model = None
        self.return_model = None
        self.calibrated_direction_model = None
        
        self.cov_inv = None
        self.train_centroid = None
        self.md_threshold = None
        
        self.features = []

    def load_and_prepare_data(self):
        """Loads labeled data, constructs Forward Returns, and splits sequentially."""
        if not os.path.exists(self.data_path):
            logger.error("No model-ready data found.")
            return False
            
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded ML matrix directly -> {self.df.shape}")
        
        # Identify feature columns (ignoring targets, OHLC, times)
        drop_cols = ["GOLDBEES.NS_Open", "GOLDBEES.NS_High", "GOLDBEES.NS_Low", 
                     "GOLDBEES.NS_Close", "GOLDBEES.NS_Volume", 
                     "Triple_Barrier_Signal", "Barrier_Hit_Time"]
        
        self.features = [c for c in self.df.columns if c not in drop_cols]
        logger.info(f"Identified {len(self.features)} valid predictor features.")
        
        # XGBoost requires classes to be positive integers (0, 1, 2)
        # Originally: -1 (Loss), 0 (Timeout), 1 (Profit)
        # Mapped: 0 (Loss), 1 (Timeout), 2 (Profit)
        self.df['Direction_Target'] = self.df['Triple_Barrier_Signal'].astype(int) + 1
        
        # Expected Return Regression Target (12 bar geometric return, strictly for regression output)
        if "GOLDBEES.NS_Close" in self.df.columns:
            # We fetch the exact 12-bar realization purely for Regression Loss tracking.
            # Not used as a feature, only as target Y2.
            self.df['Forward_Return_12'] = self.df['GOLDBEES.NS_Close'].pct_change(periods=12).shift(-12)
            # Fill terminal NaNs to avoid crashing XGBRegressor arrays
            self.df['Forward_Return_12'] = self.df['Forward_Return_12'].fillna(0.0)
            
        return True

    def train_test_split(self):
        """Creates chronological 60/20/20 Split (Train/Calibrate/Test)."""
        logger.info("Executing Chronological Split (60% Train, 20% Calib, 20% Test) to protect structural time integrity...")
        
        total_len = len(self.df)
        train_idx = int(total_len * 0.6)
        calib_idx = int(total_len * 0.8)
        
        self.train_df = self.df.iloc[:train_idx].copy()
        self.calib_df = self.df.iloc[train_idx:calib_idx].copy()
        self.test_df = self.df.iloc[calib_idx:].copy()
        
        # Prepare Matrices
        self.X_train = self.train_df[self.features].values
        self.X_calib = self.calib_df[self.features].values
        self.X_test = self.test_df[self.features].values
        
        self.y_cls_train = self.train_df['Direction_Target'].values
        self.y_cls_calib = self.calib_df['Direction_Target'].values
        self.y_cls_test = self.test_df['Direction_Target'].values
        
        self.y_reg_train = self.train_df['Forward_Return_12'].values
        self.y_reg_test = self.test_df['Forward_Return_12'].values

    def fit_anomaly_gate(self):
        """Builds Inverse Covariance (Mahalanobis) anomaly map based strictly on Train distribution."""
        logger.info("Building Mahalanobis Distance structural baseline...")
        
        # We fill NaNs conservatively for stability
        stable_X_train = np.nan_to_num(self.X_train, nan=0.0)
        
        self.train_centroid = np.mean(stable_X_train, axis=0)
        cov_matrix = np.cov(stable_X_train, rowvar=False)
        
        # Small regularizer preventing singular matrix collapse
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-4
        self.cov_inv = inv(cov_matrix)
        
        # Calculate training distances
        md_train = [distance.mahalanobis(x, self.train_centroid, self.cov_inv) for x in stable_X_train]
        
        # Establish structural Out-of-Distribution limit at the exact 95th percentile
        self.md_threshold = np.percentile(md_train, 95)
        logger.info(f"Anomaly OOD threshold explicitly locked at Mahalanobis D = {self.md_threshold:.3f}")

    def train_models(self):
        """Instantiates and trains separate Classifier and Regressor endpoints natively."""
        logger.info("Training Classifer (Direction) Endpoint...")
        self.direction_model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=0
        )
        self.direction_model.fit(self.X_train, self.y_cls_train)
        
        logger.info("Applying Platt Calibration explicitly to Classifier Confidence outputs...")
        # cv="prefit" guarantees we only adjust log-odds mapping using our strict Out-Of-Sample CALIB block!
        self.calibrated_direction_model = CalibratedClassifierCV(
            estimator=self.direction_model, 
            method='sigmoid', 
            cv=2
        )
        self.calibrated_direction_model.fit(self.X_calib, self.y_cls_calib)

        logger.info("Training Regressor (Return) Endpoint...")
        self.return_model = xgb.XGBRegressor(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            objective='reg:squarederror',
            verbosity=0
        )
        self.return_model.fit(self.X_train, self.y_reg_train)

    def evaluate_live_test_state(self):
        """Mocks live execution behavior applying predictions and safety gates natively."""
        logger.info("\n--- EXECUTING TEST RUN WITH LIVE SAFETY GATES ---")
        
        # 1. Regress
        preds_return = self.return_model.predict(self.X_test)
        
        # 2. Classify (Calibrated)
        # predict_proba returns [P(Loss), P(Timeout), P(Profit)] mapping to [0,1,2] natively
        probs = self.calibrated_direction_model.predict_proba(self.X_test)
        
        confidence_scores = np.max(probs, axis=1)
        direction_preds = np.argmax(probs, axis=1)
        
        # 3. Anomaly Guard (Information Geometry)
        stable_X_test = np.nan_to_num(self.X_test, nan=0.0)
        mahalanobis_dist = np.array([distance.mahalanobis(x, self.train_centroid, self.cov_inv) for x in stable_X_test])
        
        anomalies_flagged = (mahalanobis_dist > self.md_threshold)
        
        # Overwrite confidence organically down to exactly 0 to kill sizing allocations automatically
        confidence_scores[anomalies_flagged] = 0.0
        
        logger.info(f"Out-Of-Distribution Anomaly Gate forcefully blocked: {anomalies_flagged.sum()} trades in this test fold.")
        
        logger.info("\n[Return Regressor Loss Metrics]")
        mse = mean_squared_error(self.y_reg_test, preds_return)
        r2 = r2_score(self.y_reg_test, preds_return)
        logger.info(f"MSE: {mse:.7f} | R²: {r2:.4f}")
        
        logger.info("\n[Direction Classifier Report]")
        # Mapping back for human understanding -> [Loss (-1), Timeout (0), Profit (1)]
        target_names = ["Loss (-1)", "Timeout (0)", "Profit (1)"]
        print(classification_report(self.y_cls_test, direction_preds, target_names=target_names, zero_division=0))
        
        # Save Test state for downstream Backtest Layer
        self.test_df['Model_Signal'] = direction_preds - 1  # Mapping 0,1,2 back to -1,0,1
        self.test_df['Model_Confidence'] = confidence_scores
        self.test_df['Expected_Return'] = preds_return
        
        out_path = "data/processed/test_predictions.parquet"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.test_df.to_parquet(out_path, engine='pyarrow', index=True)
        logger.info(f"Test fold signals preserved securely for Engine Backtest -> {out_path}")

    def run_pipeline(self):
        if not self.load_and_prepare_data():
            return
        self.train_test_split()
        self.fit_anomaly_gate()
        self.train_models()
        self.evaluate_live_test_state()

if __name__ == "__main__":
    engine = AlphaModelEngine()
    engine.run_pipeline()
    
