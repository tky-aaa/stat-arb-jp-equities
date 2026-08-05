"""
Global project settings.

This file centralizes parameters used across the whole project.
"""

# ======================================================
# Universe
# ======================================================

UNIVERSE = "topix500"
# "topix100"
# "topix500"

START_DATE = "2025-06-01"
END_DATE = "2026-01-01"

# ======================================================
# Screening
# ======================================================

N_CLUSTERS = 20

MIN_CLUSTER_SIZE = 5
MAX_CLUSTER_SIZE = 50

# ======================================================
# Johansen
# ======================================================

MIN_ASSETS = 2
MAX_ASSETS = 3

TOP_N_SPREADS = 10

# ======================================================
# Backtest
# ======================================================

ROLLING_WINDOW = 60

ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5

THRESHOLD_METHOD = "fixed"
# "fixed"
# "gaussian"
# "empirical"

SPREAD_METHOD = "static"
# "static"
# "kalman"

# ======================================================
# Execution
# ======================================================

BROKER = "IBKR"
# "IBKR"
# "ALPACA"

PAPER_TRADING = True

TOP_N_EXECUTION_SPREADS = 3

# ======================================================
# Portfolio
# ======================================================

INITIAL_CAPITAL = 1.0
