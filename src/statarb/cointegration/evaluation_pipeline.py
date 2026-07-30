import pandas as pd

from statarb.screening.candidate import CandidateGroup

from .evaluate_all import evaluate_all_candidates
from .search import search_cointegrated_subgroups


def evaluation_pipeline(
    group: CandidateGroup,
    log_prices: pd.DataFrame,
    *,
    min_assets: int = 2,
    max_assets: int = 5,
    persistence_window: int = 120,
) -> pd.DataFrame:
    """
    Complete cointegration evaluation pipeline.
    """

    johansen_results = search_cointegrated_subgroups(
        group,
        log_prices,
        min_assets=min_assets,
        max_assets=max_assets,
    )

    if len(johansen_results) == 0:
        return pd.DataFrame()

    johansen_results = pd.DataFrame(johansen_results)

    return evaluate_all_candidates(
        johansen_results,
        log_prices,
        persistence_window=persistence_window,
    )
