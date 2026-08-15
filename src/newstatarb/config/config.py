from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    universe: str = "topix100"

    training_start: str = "2025-06-01"
    training_end: str = "2025-10-01"
    test_start: str = "2025-10-01"
    test_end: str = "2026-01-01"

    screening_method: str = "full"  # ("full", "pca", "ff3")
    n_clusters: int = 20
    min_cluster_size: int = 5
    max_cluster_size: int = 50
    min_assets: int = 2
    max_assets: int = 3
    pca_components: int = 5

    persistence_days: int = 60

    zscore_window: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    threshold_method: str = "fixed"  # ("fixed", "gaussian", "empirical")
    spread_method: str = "fixed"  # ("static", "kalman")

    top_n_spreads: int = 10

    initial_capital: float = 1.0
    broker: str = "IBKR"
    paper_trading: bool = True


@dataclass(frozen=True)
class LiveConfig:
    universe: str = "topix100"

    training_start: str = "2025-06-01"
    training_end: str = "2025-10-01"

    screening_method: str = "full"
    n_clusters: int = 20
    min_cluster_size: int = 5
    max_cluster_size: int = 50
    min_assets: int = 2
    max_assets: int = 3
    pca_components: int = 5

    persistence_days: int = 60

    zscore_window: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    threshold_method: str = "fixed"
    spread_method: str = "fixed"

    top_n_spreads: int = 10

    strategy_path: str = "data/live/strategy.pkl"

    broker: str = "IBKR"
    paper_trading: bool = True
