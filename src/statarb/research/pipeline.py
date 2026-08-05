from pathlib import Path

import pandas as pd

from statarb.backtest.pipeline import run_spread_backtest
from statarb.backtest.report import evaluate_backtest
from statarb.cointegration.evaluation_pipeline import evaluation_pipeline
from statarb.cointegration.ranking import rank_spreads
from statarb.cointegration.selection import select_top_spreads
from statarb.config.universe import get_topix100, get_topix500
from statarb.data.instruments.japan_equity import JapanEquity
from statarb.data.loaders.yfinance_loader import YahooFinanceSource
from statarb.data.transform.transform import preprocess_prices
from statarb.execution.paper import simulate_paper_execution
from statarb.reports.summary import create_backtest_summary
from statarb.screening.candidate import CandidateGroup
from statarb.screening.clustering import cluster_by_pca
from statarb.screening.pca import compute_pca_features


def _run_one_configuration(
    selected,
    log_prices,
    *,
    threshold_method,
    spread_method,
):
    metrics = []
    spread_results = []
    paper_results = []

    for spread in selected:
        bt = run_spread_backtest(
            spread,
            log_prices,
            spread_method=spread_method,
            threshold_method=threshold_method,
        )

        spread_results.append(bt)

        metrics.append(
            evaluate_backtest(
                bt.backtest,
                bt.signal,
            )
        )

        paper_results.append(
            simulate_paper_execution(
                bt.spread,
                bt.signal,
            )
        )

    summary = create_backtest_summary(
        selected,
        metrics,
    )

    return {
        "summary": summary,
        "spread_results": spread_results,
        "paper_results": paper_results,
    }


def run_research_pipeline(
    *,
    universe="topix100",
    start="2025-06-01",
    end="2026-01-01",
    n_clusters=20,
    min_cluster_size=5,
    max_cluster_size=50,
    min_assets=2,
    max_assets=3,
    top_n=10,
    max_groups=5,
):
    if universe == "topix100":
        tickers = get_topix100()
    elif universe == "topix500":
        tickers = get_topix500()
    else:
        raise ValueError

    instruments = [JapanEquity(t) for t in tickers]

    source = YahooFinanceSource()

    prices = source.get_prices(
        instruments=instruments,
        start=start,
        end=end,
    )

    clean_prices = prices.dropna(axis=1)

    log_prices = preprocess_prices(clean_prices)

    returns = log_prices.diff().dropna()

    pca_features = compute_pca_features(
        returns,
        n_components=10,
    )

    clusters = cluster_by_pca(
        pca_features,
        n_clusters=n_clusters,
    )

    candidate_groups = []

    for cluster_id in sorted(clusters.unique()):
        members = clusters[clusters == cluster_id].index.tolist()

        if min_cluster_size <= len(members) <= max_cluster_size:
            candidate_groups.append(
                CandidateGroup(
                    tickers=members,
                )
            )

    if max_groups is not None:
        candidate_groups = candidate_groups[:max_groups]

    evaluation_results = []

    for group in candidate_groups:
        result = evaluation_pipeline(
            group,
            log_prices,
            min_assets=min_assets,
            max_assets=max_assets,
        )

        if not result.empty:
            evaluation_results.append(result)

    evaluation = (
        pd.concat(
            evaluation_results,
            ignore_index=True,
        )
        if evaluation_results
        else pd.DataFrame()
    )

    ranking = rank_spreads(evaluation)

    selected = select_top_spreads(
        ranking,
        n_spreads=top_n,
    )

    fixed = _run_one_configuration(
        selected,
        log_prices,
        threshold_method="fixed",
        spread_method="static",
    )

    gaussian = _run_one_configuration(
        selected,
        log_prices,
        threshold_method="gaussian",
        spread_method="static",
    )

    empirical = _run_one_configuration(
        selected,
        log_prices,
        threshold_method="empirical",
        spread_method="static",
    )

    kalman = _run_one_configuration(
        selected,
        log_prices,
        threshold_method="fixed",
        spread_method="kalman",
    )

    Path("results").mkdir(exist_ok=True)

    fixed["summary"].to_csv(
        "results/summary_fixed.csv",
        index=False,
    )

    gaussian["summary"].to_csv(
        "results/summary_gaussian.csv",
        index=False,
    )

    empirical["summary"].to_csv(
        "results/summary_empirical.csv",
        index=False,
    )

    kalman["summary"].to_csv(
        "results/summary_kalman.csv",
        index=False,
    )

    return {
        "prices": prices,
        "clean_prices": clean_prices,
        "log_prices": log_prices,
        "candidate_groups": candidate_groups,
        "evaluation": evaluation,
        "ranking": ranking,
        "selected": selected,
        "summary_fixed": fixed["summary"],
        "summary_gaussian": gaussian["summary"],
        "summary_empirical": empirical["summary"],
        "summary_kalman": kalman["summary"],
        "spread_fixed": fixed["spread_results"],
        "spread_gaussian": gaussian["spread_results"],
        "spread_empirical": empirical["spread_results"],
        "spread_kalman": kalman["spread_results"],
        "paper_fixed": fixed["paper_results"],
        "paper_gaussian": gaussian["paper_results"],
        "paper_empirical": empirical["paper_results"],
        "paper_kalman": kalman["paper_results"],
    }
