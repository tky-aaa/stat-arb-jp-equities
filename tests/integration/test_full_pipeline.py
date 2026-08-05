import pandas as pd

from statarb.backtest.engine import (
    run_backtest,
)
from statarb.cointegration.evaluation_pipeline import (
    evaluation_pipeline,
)
from statarb.cointegration.ranking import (
    rank_spreads,
)
from statarb.cointegration.selection import (
    select_top_spreads,
)
from statarb.cointegration.spread import (
    create_spread,
)
from statarb.screening.candidate import (
    CandidateGroup,
)
from statarb.signals.zscore import (
    generate_zscore_signal,
)


def test_full_cointegration_pipeline():

    prices = pd.read_pickle(
        "tests/integration/data/cement_prices.pkl",
    )

    group = CandidateGroup(
        tickers=list(prices.columns),
    )

    evaluation = evaluation_pipeline(
        group,
        prices,
        min_assets=2,
        max_assets=5,
    )

    assert len(evaluation) > 0

    ranked = rank_spreads(
        evaluation,
    )

    assert len(ranked) > 0

    selected = select_top_spreads(
        ranked,
        n_spreads=1,
    )

    assert len(selected) == 1

    selected_spread = selected[0]

    prices_selected = prices[selected_spread.tickers]

    signal = generate_zscore_signal(
        create_spread(
            prices_selected,
            selected_spread.beta,
        )
    )

    backtest = run_backtest(
        prices_selected,
        selected_spread.beta,
        signal.position,
    )

    assert backtest.cumulative_pnl.notna().all()

    assert len(backtest.cumulative_pnl) == len(prices)
