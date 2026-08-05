# Stat Arb JP Equities — User Guide

---

# 1. Overview

This project implements a complete statistical arbitrage research and execution pipeline for Japanese equities.

The current pipeline consists of:

* Market data acquisition
* Data preprocessing
* PCA and clustering
* Johansen cointegration analysis
* Spread construction
* Spread evaluation
* Candidate ranking
* Signal generation
* Backtesting
* Paper trading (IBKR / Alpaca)

Overall workflow:

```text
Price Data
    ↓
Preprocessing
    ↓
PCA / Clustering
    ↓
Johansen Cointegration
    ↓
Spread Evaluation
    ↓
Ranking
    ↓
Signal Generation
    ↓
Backtest
    ↓
Paper Trading
```

---

# 2. Configuration

All experiment settings are controlled through

```text
src/statarb/config/settings.py
```

Important parameters:

| Parameter               | Description                         |
| ----------------------- | ----------------------------------- |
| UNIVERSE                | TOPIX100 / TOPIX500                 |
| START_DATE              | Backtest start date                 |
| END_DATE                | Backtest end date                   |
| N_CLUSTERS              | Number of PCA clusters              |
| MIN_CLUSTER_SIZE        | Minimum cluster size                |
| MAX_CLUSTER_SIZE        | Maximum cluster size                |
| MIN_ASSETS              | Minimum assets for Johansen         |
| MAX_ASSETS              | Maximum assets for Johansen         |
| TOP_N_SPREADS           | Number of selected spreads          |
| ROLLING_WINDOW          | Rolling z-score window              |
| ENTRY_THRESHOLD         | Entry threshold                     |
| EXIT_THRESHOLD          | Exit threshold                      |
| THRESHOLD_METHOD        | fixed / gaussian / empirical        |
| SPREAD_METHOD           | static / kalman                     |
| BROKER                  | IBKR / ALPACA                       |
| PAPER_TRADING           | Enable paper trading                |
| TOP_N_EXECUTION_SPREADS | Number of spreads sent to execution |

---

# 3. Running the Research Pipeline

From the project root:

```bash
uv run python -m statarb.pipeline
```

After completion, the console prints a short summary of the pipeline.

Results are written into

```text
results/
```

---

# 4. Output Files

| File                  | Description                  |
| --------------------- | ---------------------------- |
| summary_fixed.csv     | Fixed threshold strategy     |
| summary_gaussian.csv  | Gaussian threshold strategy  |
| summary_empirical.csv | Empirical threshold strategy |
| summary_kalman.csv    | Kalman-filter strategy       |
| selected_spreads.pkl  | Selected spread candidates   |

---

# 5. Reading the Backtest Results

Primary performance metrics:

* Total Return
* Annualized Return
* Volatility
* Sharpe Ratio
* Maximum Drawdown
* Win Rate
* Number of Trades

Recommended evaluation order:

1. Sharpe Ratio
2. Maximum Drawdown
3. Annualized Return
4. Win Rate
5. Number of Trades

---

# 6. Threshold Methods

### Fixed

Uses manually specified thresholds.

```text
Entry = 2σ
Exit  = 0.5σ
```

### Gaussian

Optimizes thresholds under a Gaussian assumption.

### Empirical

Optimizes thresholds directly from historical observations.

---

# 7. Spread Construction

### Static

Uses fixed Johansen cointegration weights.

### Kalman

Uses time-varying hedge ratios estimated by a Kalman Filter.

---

# 8. Paper Trading

## Interactive Brokers

1. Launch Trader Workstation (Paper Account)
2. Enable API access
3. Execute

```bash
uv run python -m statarb.pipeline
```

Orders may also be submitted programmatically through

```python
IBKRExecution.submit_order(...)
```

Monitor:

* Paper Portfolio
* Open Orders
* Activity Log

---

## Alpaca

Configure API credentials in

```text
.env
```

```text
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

Orders are submitted through

```python
AlpacaExecution.submit_order(...)
```

---

# 9. Frequently Used Commands

Run research:

```bash
uv run python -m statarb.pipeline
```

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

Format:

```bash
uv run ruff format .
```

---

# 10. Typical Development Workflow

```text
Modify settings.py
        ↓
Run pipeline
        ↓
Review summary files
        ↓
Inspect selected spreads
        ↓
Adjust parameters
        ↓
Repeat
```

---

# 11. Paper Trading Checklist

Before sending any orders:

* Verify the selected universe
* Verify the backtest period
* Verify threshold settings
* Verify spread construction method
* Verify broker selection
* Confirm paper trading mode
* Review summary statistics
* Review selected spreads
* Submit paper orders
* Validate execution before moving to live trading
