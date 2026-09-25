from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    universe: str = "topix50"

    training_start: str = "2022-01-01"
    training_end: str = "2025-12-31"
    test_start: str = "2026-01-01"
    test_end: str = "2026-08-31"

    screening_method: str = "ff3"  # ("full", "pca", "ff3")
    n_clusters: int = 10
    min_cluster_size: int = 3
    max_cluster_size: int = 20
    min_assets: int = 2
    max_assets: int = 3
    pca_components: int = 5

    persistence_days: int = 60

    zscore_window: int = 80  # to be validated
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5  # to be validated
    threshold_method: str = "gaussian"  # ("fixed", "gaussian", "empirical")
    spread_method: str = "kalman"  # ("static", "kalman")

    top_n_spreads: int = 5

    results_path: str = "results"
    initial_capital: float = 1.0
    broker: str = "IBKR"
    paper_trading: bool = True


@dataclass(frozen=True)
class LiveConfig:
    universe: str = "topix50"

    training_start: str = "2024-01-01"
    training_end: str = "2025-12-31"

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
    threshold_method: str = "gaussian"
    spread_method: str = "kalman"

    top_n_spreads: int = 5

    strategy_path: str = "data/live/strategy.pkl"

    broker: str = "IBKR"
    paper_trading: bool = True
