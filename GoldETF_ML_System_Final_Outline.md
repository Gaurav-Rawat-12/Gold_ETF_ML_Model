# Gold ETF ML Trading System — Final Architecture Outline
### Nippon India ETF Gold BeES | Intraday + Short Horizon
> Compiled from full Karan × Sara design review session

---

## System Philosophy

```
Predict → Filter → Size → Execute → Control Risk → Observe → Retrain
```

This is a **decision system**, not a prediction system.
Core discipline: *Most profit comes from trades avoided.*

---

## 1. Data Layer

### 1.1 Primary Market Data (Kite Connect)
- ETF OHLCV — 1-min / 5-min candles
- MCX Gold Futures (spot-adjusted near expiry — see Fair Value note)
- Global Gold spot price (USD)

### 1.2 Cross-Asset Data
- USD/INR intraday
- NIFTY 50 (lead-lag signal)
- DXY index

### 1.3 Macro / Event Layer
- Federal Reserve meeting dates
- RBI policy announcement dates
- CPI / inflation releases
- **F&O expiry days (last Thursday of month)** ← Indian market specific
- Union Budget day
- All flagged as binary event features with time-to-event countdown

### 1.4 Microstructure (Live Only — NOT in model)
- Best bid/ask price + quantity (Kite WebSocket, mode: `full`)
- Top 5 depth levels (bid + ask)
- Used exclusively in Execution Gate (Stage 2)
- NOT available historically → NOT used in ML model training

---

## 2. Data Engineering

### 2.1 Standard Processing
- Strict time alignment across all data sources
- Missing value handling: forward fill + missingness flag feature
- Outlier clipping (rolling IQR-based)
- Rolling z-score normalization (no look-ahead)

### 2.2 Critical Additions
- **Purge + Embargo gaps in walk-forward validation**
  - Forward labels (5-min/15-min returns) create overlapping windows
  - Purge gap = label horizon; Embargo = additional buffer
  - Without this: validation Sharpe is optimistically inflated
- Feature drift tracking (distribution shift monitoring)
- Feature freshness checks (stale data detection)

---

## 3. Feature Engineering

### 3.1 Price & Technical
- SMA, EMA (multiple windows)
- RSI, MACD
- Bollinger Bands
- ATR
- VWAP

### 3.2 Statistical Features
- Rolling mean / std
- Skewness / kurtosis
- Z-score of returns

### 3.3 Volume Features
- Volume spikes
- Volume moving average
- Relative Volume: `RV = current_volume / avg_volume` (institutional activity proxy)
- Volume-weighted returns

### 3.4 Lag Features
- lag_1 … lag_n (returns, volume, indicators)

### 3.5 Time Features
- Hour / minute
- Session flags: open / midday / close
- Session weighting (morning vs afternoon behavioral difference)

### 3.6 Cross-Asset Features
- ETF vs Gold spread
- Gold × USDINR interaction
- NIFTY 50 lagged returns
- USDINR spike signals
- Rolling cross-asset correlation shifts

### 3.7 Fair Value Features (Core Alpha Source)
```
theoretical_price  = gold_price × usdinr × 0.0311
premium_discount   = (etf_price - theoretical_price) / theoretical_price
```
**MCX Roll Cost Correction (Critical):**
- If gold_price = MCX Futures, premium_discount drifts near expiry
- Either use spot-adjusted MCX price OR add `days_to_expiry` correction factor
- Without this: systematic false signals every monthly expiry cycle

Additional fair value features:
- Rolling z-score of mispricing
- Rolling tracking error (ETF price vs published NAV, 5-day window)

### 3.8 Volatility Features
- Standard close-to-close volatility
- **Parkinson Volatility (Advanced):**
  ```
  σ_park = sqrt( 1/(4n·ln2) · Σ (ln(H_i/L_i))² )
  ```
  Captures intraday hidden volatility within each candle.
  Used by Decision Engine to dynamically widen SL / reduce size.

### 3.9 Regime Features
- Volatility regime label
- Trend strength metric

### 3.10 Event Features
- Binary event flag
- Time-to-next-event (in minutes/hours)

### 3.11 Order Flow (Approximated, Historical-Safe)
- No real OFI without tick data
- Volume-based proxy: volume acceleration + relative volume

---

## 4. Target Engineering

### 4.1 Multi-Target Setup
```
Y1: Direction       (classification)
Y2: Expected Return (regression)
Y3: Volatility      (regression)
Y4: Trade Viability (classification)
```

### 4.2 Triple Barrier Method (Primary Labeling)
Labels based on whichever barrier is hit first:
- Profit target hit → positive label
- Stop loss hit → negative label
- Time expiry hit → neutral / discard

Captures path dependency — reflects real trading behavior.

### 4.3 Trade Horizons
- 5-minute forward return
- 15-minute forward return

### 4.4 Noise Filtering
- Ignore moves below minimum threshold
- Only label economically meaningful trades

---

## 5. Model Architecture

### 5.1 Core Model
- XGBoost or LightGBM (primary)
- Single multi-output model with shared feature representation

### 5.2 Output Heads
```
Shared Features
      ↓
  ┌───────────────────────────┐
  │ Direction Head            │ → BUY / SELL / NO TRADE
  │ Return Head               │ → Expected Return (%)
  │ Confidence Head           │ → Raw probability score
  │ Regime Head               │ → TREND / SIDEWAYS / VOLATILE
  └───────────────────────────┘
```

### 5.3 Confidence Calibration (Critical — Often Skipped)
- Raw XGBoost/LightGBM probabilities are NOT calibrated
- Apply **Platt Scaling** or **Isotonic Regression** post-training
- Without calibration: position sizing formula receives incorrect confidence values
- `position_size = capital × confidence × (1/volatility)` breaks if confidence is uncalibrated

### 5.4 Anomaly Detection: Mahalanobis Distance
- Measures how far current market state is from training distribution
- If distance exceeds threshold → confidence score forced to zero
- Prevents "hallucinated" trades during flash crashes or extreme news events
- Acts as out-of-distribution (OOD) gatekeeper

### 5.5 Regime Gatekeeper: Information Geometry
- Monitors correlation structure between Gold, USDINR, and NIFTY 50
- When correlations decouple (e.g., Gold and USDINR both falling simultaneously):
  - Regime Head flags VOLATILE/UNSTABLE
  - System restricts to high-conviction signals only

---

## 6. Training Strategy

### 6.1 Walk-Forward Validation (with Purge + Embargo)
```
Train Window → [Purge Gap] → [Embargo] → Test Window → roll forward
```
- Purge gap = label horizon (prevents label leakage)
- Embargo = additional buffer (prevents feature leakage from proximity)

### 6.2 Event-Aware Validation
- Validation folds explicitly include Fed, RBI, F&O expiry, Budget days
- System must prove robustness on event days specifically

### 6.3 Sample Weighting
- Higher weight to high-volume, high-volatility periods
- Regime-aware weighting (trend vs sideways behavior differs)

---

## 7. Two-Stage Production Architecture

### Stage 1 — Alpha Model (ML)
```
Inputs : OHLCV, technicals, cross-asset, fair value, regime features
Output : { signal, expected_return, confidence, regime }
Trained : on clean historical OHLCV data
Backtest: fully reproducible, no microstructure contamination
```

### Stage 2 — Execution Gate (Rule-Based)
```
Inputs : live bid/ask spread, OFI, depth imbalance (Kite WebSocket)
Logic  : rule-based only (NOT ML — insufficient labeled fill history)
Output : EXECUTE / HOLD / ADJUST SIZE
```

**One-Way Information Flow:**
```
Alpha Model output → Execution Gate   ✅ (allowed)
Execution Gate output → Alpha Model   ❌ (never)
```

**Two Alpha outputs passed downstream:**
1. Volatility regime → dynamically adjusts spread threshold
2. Confidence score → scales position size

### Stage 2 Gate Logic (Priority Order)
```python
# 1. Spread Gate (Highest Impact)
spread_bps = (ask - bid) / mid_price × 10000
spread_percentile = rolling percentile within last 30-min window
if spread_percentile > 80th:
    → HOLD

# 2. Flow Gate
ofi = (bid_qty_l1 - ask_qty_l1) / (bid_qty_l1 + ask_qty_l1)
if ofi < -0.3 and signal == BUY:
    → HOLD
if ofi > +0.3 and signal == SELL:
    → HOLD

# 3. Size Scaler
execution_size = model_size × (1 - spread_percentile_normalized)
```

**Execution Rule Impact Ranking (Retail Level):**
1. Spread filtering — highest impact (Gold BeES spread widens 3-5× at open/close)
2. Passive limit logic — second (if implemented cleanly)
3. Order imbalance — weakest with top-5 depth only; treat as soft filter

### 7.1 Passive Limit Logic
- Place orders at mid-price instead of crossing the spread
- Saves 0.10–0.20% per trade
- If unfilled within X seconds → chase by one tick or cancel
- Requires careful chase logic to avoid worse average entry than market orders

---

## 8. Position Sizing

```python
position_size = capital × calibrated_confidence × (1 / parkinson_volatility)
```

Regime-adjusted stop losses:
- TREND regime: wider SL (1.5× ATR)
- VOLATILE regime: tighter SL (0.8× ATR)
- SIDEWAYS regime: standard SL (1.0× ATR)

---

## 9. Risk Management

### 9.1 Trade-Level
- ATR-based stop loss (regime-adjusted via Parkinson Vol)
- Controlled take profit (triple barrier defines targets)

### 9.2 Portfolio-Level
- Max capital per trade
- Max daily loss

### 9.3 Equity Curve Circuit Breaker
- Standard max daily loss is insufficient
- Add: **rolling 10-trade win rate circuit breaker**
  ```
  if rolling_10_trade_win_rate < 40%:
      → pause trading
      → flag for manual review / retraining check
  ```
- Protects capital during model decay between retraining cycles

---

## 10. Backtesting Engine

### 10.1 Realistic Execution Simulation
- Transaction costs: brokerage + STT + slippage
- Bid/Ask spread reality: buy at ASK, sell at BID
- Latency simulation
- Partial fill handling

### 10.2 Stress Testing Grid
```
Volatility:  low / medium / high
Cost:        low / medium / high
Liquidity:   normal / stressed
```

### 10.3 Scenario Events
- INR spike
- Gold crash
- Liquidity drop

### 10.4 Backtest Integrity Check
- Backtest runs on OHLCV + cross-asset features ONLY
- Zero microstructure features in backtest (matches training)
- Backtest Sharpe = honest Sharpe

---

## 11. Slippage Measurement Framework

### 11.1 Dual Slippage Anchors
```
decision_slippage  = fill_price - mid_price at SIGNAL TIME
execution_slippage = fill_price - mid_price at ORDER PLACEMENT TIME
```
- Decision slippage measures model timing quality
- Execution slippage measures order handling quality
- Separating them identifies whether underperformance is signal or execution

### 11.2 Opportunity Cost Measurement
Do NOT simulate fills for blocked trades (path-dependent, biased).
Instead:
```
Gate quality  : blocked_expected_return vs allowed_expected_return
Model quality : allowed_expected_return vs allowed_realized_return
```
Isolates gate quality from model quality cleanly.

---

## 12. Logging Schema (Phase 1 — Non-Negotiable Fields)

```
timestamp          — millisecond precision (microsecond not reliable via Kite)
mid_price          — (bid + ask) / 2 at signal time
arrival_price      — mid_price at order placement time
spread_bps         — (ask - bid) / mid × 10000
model_signal       — BUY / SELL / NO TRADE
model_confidence   — calibrated confidence score (critical for sizing debug)
gate_decision      — EXECUTE / HOLD + reason code
fill_price         — actual fill price (NULL if blocked)
fill_latency_ms    — time(fill) - time(signal) in milliseconds
model_version      — version tag of model that generated the signal
```

**fill_latency_ms** is critical and often omitted:
- Diagnoses API delay, order queue delay, passive logic failures
- Without it: can't separate execution quality from signal quality

Store as flat Parquet files, partitioned by date.
Do not use a database in Phase 1.

### 12.1 Extended Logging (Add After 30 Days)
```
bid_qty_l1..l5     — top 5 bid levels
ask_qty_l1..l5     — top 5 ask levels
ofi                — (bid_qty_l1 - ask_qty_l1) / (bid_qty_l1 + ask_qty_l1)
depth_imbalance    — (Σbid_qty - Σask_qty) / (Σbid_qty + Σask_qty)
volume_since_open  — cumulative session volume
session_bucket     — open / mid / close
regime_at_signal   — TREND / SIDEWAYS / VOLATILE
parkinson_vol      — computed from last N candles
```

This dataset becomes a **proprietary microstructure dataset** for Gold BeES.
No vendor provides this. After 6 months, it is a genuine edge.

---

## 13. Post-Trade Analysis Layer

Run end-of-day. Separate from model and execution layers.

### 13.1 Per-Trade Metrics
- Spread at signal time vs fill time
- OFI direction at signal time
- Decision slippage and execution slippage in bps
- Gate decision correctness (blocked trade that would have been bad?)

### 13.2 Weekly Aggregates
```
Gate precision : trades blocked that would have had slippage > 10 bps
Gate recall    : trades allowed that had slippage > 10 bps anyway
Best/worst execution windows by time of day
Rolling 10-trade win rate
```

### 13.3 Why This Layer Exists
Without it: when the system underperforms in month 4, you cannot determine whether:
- Alpha model is decaying
- Execution quality has degraded
- A specific market regime is unfavorable

This layer is what makes the system **diagnosable** in production.

---

## 14. Execution Degradation Alerting

**Primary trigger:**
```
7-day rolling slippage bps > 2× baseline median
AND
trade count in window > 50 (avoid false alarms from low sample)
```

- Gate precision drop: lagging indicator, use as secondary confirmation
- Expected vs realized divergence: conflates alpha and execution decay
- Slippage bps: cleanest isolated signal for execution health

**Safe Mode (on alert):**
- Reduce position sizes by 50%
- Tighten spread gate threshold
- Flag for manual review before next session

---

## 15. Monitoring & Continuous Learning

### 15.1 Track
- Prediction vs actual (per output head)
- Model drift (feature distribution shift)
- Feature importance over time (which features driving which periods)
- **Feature importance drift** — detect model decay early

### 15.2 Retraining Trigger
- Weekly scheduled retraining (baseline)
- Drift-triggered retraining (when distribution shift detected)
- Equity curve circuit breaker trigger (win rate collapse)

### 15.3 When to Add Execution ML (Transition Criteria)
```
Minimum conditions:
  - 6+ months logged live fills
  - 10,000+ labeled execution events
  - Stable alpha model (not drifting)
  - R² of slippage ~ rules < 0.35 (residual structure exists)
  - Unexplained slippage economically significant vs expected return
```
Start with **supervised fill prediction**, not RL.
RL requires a fill simulator with enough fidelity — not achievable at retail level with Kite.

---

## 16. System Flow (Complete)

```
KITE CONNECT LIVE FEED
        │
        ├─── OHLCV + Cross-Asset ────► STAGE 1: ALPHA MODEL
        │    (historical-compatible)          │
        │                              { signal, return,
        │                                confidence, regime }
        │                                      │
        └─── Bid/Ask + Depth ─────────► STAGE 2: EXECUTION GATE
             (live microstructure)             │
                                    ┌──────────┴──────────┐
                                    │ Spread gate          │
                                    │ Flow gate            │
                                    │ Size scaler          │
                                    └──────────┬──────────┘
                                               │
                                        EXECUTE / HOLD
                                               │
                                        ORDER PLACEMENT
                                        (passive limit logic)
                                               │
                                          RISK ENGINE
                                        (SL / TP / sizing)
                                               │
                                     LOGGING (all 9 fields)
                                               │
                              ┌────────────────┴────────────────┐
                              │        POST-TRADE ANALYSIS       │
                              │  (slippage / gate audit / PnL)   │
                              └────────────────┬────────────────┘
                                               │
                                   MONITORING + RETRAINING
```

---

## 17. Operational Safeguards

### 17.1 Data Failure Fallback
```
If any primary feed is stale > 30 seconds → NO TRADE
If MCX/USDINR feed missing → Fair Value features unavailable → NO TRADE
If Kite WebSocket disconnected → Execution Gate blind → NO TRADE
```
Default behavior on any data uncertainty: **do nothing**.
Log every fallback trigger with reason code for post-session review.

### 17.3 Rolling Feature Warm-Up Guard
At session open, rolling features (Parkinson vol, z-score, session percentile) are unstable:
```
if bars_since_open < min_lookback_period → NO TRADE
```
Distinct from the execution-noise reason to avoid open — this is feature validity.
Typically 10–15 bars depending on shortest rolling window in feature set.
For the first 1–2 weeks of live deployment:
```
position_size = normal_size × 0.25
```
Rationale: model has not been validated on live fills yet.
Backtest performance does not guarantee live behavior.
Ramp up only after 2 weeks of observed slippage matching back

```
"This is a two-stage, single-model, multi-output trading system
designed around realistic backtesting and production reliability.

Stage 1 generates alpha from historical-compatible features only.
Stage 2 uses live microstructure exclusively for execution quality.

The separation is intentional: backtest numbers are honest,
live behavior matches backtest behavior,
and every layer is independently measurable."
```

---

## Appendix: Key Design Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Microstructure in model | ❌ No | Train/serve skew destroys backtest validity |
| Microstructure in execution | ✅ Yes | Right layer, right timescale |
| Confidence calibration | ✅ Platt/Isotonic | Raw probabilities break position sizing |
| Walk-forward validation | ✅ With purge+embargo | Prevents label leakage |
| Execution model type | Rule-based (Phase 1) | Insufficient labeled fill data |
| Slippage anchor | Mid-price at signal time | Cleanest decision-quality counterfactual |
| Spread threshold | Rolling 30-min percentile | Adapts to intraday regime shifts |
| Degradation trigger | Slippage bps + count gate | Isolates execution from alpha decay |
| Logging precision | Millisecond | Kite WebSocket limitation |
| Fill prediction (future) | Supervised, not RL | RL requires unavailable fill simulator |
