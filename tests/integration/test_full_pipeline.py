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

    spread = create_spread(
        prices[selected[0].tickers],
        selected[0].beta,
    )

    signal = generate_zscore_signal(
        spread,
    )

    backtest = run_backtest(
        spread,
        signal.position,
    )

    assert backtest.equity.notna().all()

    assert len(backtest.equity) == len(prices)
