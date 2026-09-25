# Statistical Arbitrage in Japanese Equities

An end-to-end research and execution framework for cointegration-based statistical arbitrage in Japanese equities.

## Overview

This project investigates whether stock screening methodology affects the out-of-sample performance of cointegration-based statistical arbitrage.

The system implements the complete research pipeline:

**Data → Screening → Cointegration Analysis → Signal Generation → Portfolio Construction → Backtesting → Paper Trading**

Three screening methodologies are evaluated:

* **Full** — direct candidate search
* **PCA** — low-dimensional representation of return patterns
* **FF3** — Fama-French three-factor exposures

The framework is designed to support systematic experimentation with alternative screening, spread construction, signal-generation, and portfolio configurations.

## Research Question

> **Does stock screening methodology affect the out-of-sample performance of cointegration-based statistical arbitrage?**

The experiment compares Full, PCA, and FF3 screening while keeping the main downstream strategy components fixed.

## Empirical Results

The final out-of-sample evaluation uses a TOPIX 50 universe with:

* **Training:** Jan 2022 – Dec 2025
* **Test:** Jan 2026 – Aug 2026
* **Candidate groups:** 2–3 assets
* **Selected spreads:** Top 5
* **Spread construction:** Kalman filter

### Out-of-Sample Performance

| Screening Method | Total Return | Annualized Return | Volatility | Sharpe | Max Drawdown | Win Rate | Trades |
| ---------------- | -----------: | ----------------: | ---------: | -----: | -----------: | -------: | -----: |
| Full             |       +7.33% |           +11.78% |     15.08% |   0.77 |       −6.75% |    52.5% |     96 |
| PCA              |      −31.28% |           −44.61% |     17.37% |  −2.84 |      −33.47% |    50.0% |     36 |
| FF3              |       −1.07% |            −1.67% |      7.46% |  −0.23 |       −8.19% |    55.0% |     55 |

The results illustrate that differences in the candidate-generation stage can propagate through cointegration testing, spread selection, and ultimately trading performance.

### Project Summaries

* [Research Summary](docs/research-summary.pdf)
* [System Summary](docs/system-summary.pdf)

The Research Summary provides the experimental design, empirical results, interpretation, limitations, and future work.

The System Summary provides an overview of the software architecture and research pipeline.

## System Architecture

The framework is organized as a modular research pipeline:

```text
Market Data
     │
     ▼
Stock Screening
     │
     ▼
Cointegration Analysis
     │
     ▼
Spread Construction
     │
     ▼
Signal Generation
     │
     ▼
Portfolio Construction
     │
     ▼
Backtesting
     │
     ▼
IBKR Paper Trading
```

The modular architecture allows individual components and strategy parameters to be configured and evaluated independently.

## Repository Structure

```text
src/statarb/

├── config/
├── core/
│   ├── data/
│   │   ├── api.py
│   │   ├── factors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── french.py
│   │   ├── instruments/
│   │   │   ├── __init__.py
│   │   │   └── japan_equity.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── yfinance.py
│   │   └── universe/
│   │       ├── __init__.py
│   │       └── universe.py
│   ├── screening/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── clustering.py
│   │   ├── ff3.py
│   │   ├── groups.py
│   │   ├── pca.py
│   │   └── subgroups.py
│   ├── cointegration/
│   │   ├── api.py
│   │   ├── distribution_evaluator.py
│   │   ├── johansen_tester.py
│   │   ├── mean_reversion_evaluator.py
│   │   ├── spread_creator.py
│   │   └── stationarity_evaluator.py
│   ├── signal/
│   │   ├── api.py
│   │   ├── generator.py
│   │   ├── kalman.py
│   │   ├── zscore.py
│   │   └── threshold/
│   │       ├── __init__.py
│   │       ├── empirical.py
│   │       ├── fixed.py
│   │       └── gaussian.py
│   ├── portfolio/
│   │   ├── api.py
│   │   ├── allocator.py
│   │   ├── leverage.py
│   │   └── selector.py
│   ├── backtest/
│   │   ├── api.py
│   │   ├── portfolio_return_calculator.py
│   │   └── return_calculator.py
│   ├── report/
│   │   ├── api.py
│   │   ├── export.py
│   │   ├── metrics.py
│   │   └── storage.py
│   └── live/
│       ├── __init__.py
│       ├── execution.py
│       ├── ibkr.py
│       └── storage.py
└── entry/
```

Key components include:

* **Screening** — generates candidate asset groups using alternative screening methodologies.
* **Cointegration** — evaluates candidate groups using cointegration tests and persistence criteria.
* **Signal** — constructs spreads and generates trading signals from standardized spread deviations.
* **Portfolio** — selects and allocates capital across candidate spreads.
* **Backtest** — evaluates portfolio performance over historical test periods.
* **Execution** — interfaces with IBKR for paper-trading execution.

## Reproducibility

The project uses `uv` for Python environment and dependency management.

```bash
uv sync
```

Run the backtest from the project root:

```bash
uv run python -m statarb.entry.backtest
```

Backtest outputs are saved under:

```text
results/
```

Configuration parameters are centralized in the project configuration and can be modified to run alternative experimental settings.

## Further Reading

The theoretical and technical foundations of the methods used in this project are documented separately.

Planned topics include:

* Cointegration and the Johansen test
* Kalman filtering
* Principal Component Analysis
* Fama-French factor models
* Mean reversion and statistical arbitrage
* Time-series analysis
* Backtesting and portfolio construction

Detailed mathematical explanations and implementation notes will be added as the corresponding articles are developed.
