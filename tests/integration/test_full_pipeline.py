import numpy as np
import pandas as pd

from statarb.backtest.engine import (
    run_backtest,
)
from statarb.cointegration.evaluate_all import (
    evaluate_all_candidates,
)
from statarb.cointegration.johansen import (
    estimate_cointegration,
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
from statarb.signals.zscore import (
    generate_zscore_signal,
)


def test_full_cointegration_pipeline():

    prices = pd.read_pickle("tests/integration/data/cement_prices.pkl")

    johansen = estimate_cointegration(
        prices,
    )

    assert johansen.rank == 1

    johansen_results = pd.DataFrame(
        [
            {
                "tickers": johansen.tickers,
                "rank": johansen.rank,
                "beta": johansen.beta[:, [0]],
                "error": np.nan,
            }
        ]
    )

    evaluation = evaluate_all_candidates(
        johansen_results,
        prices,
    )

    assert len(evaluation) == 1

    ranked = rank_spreads(
        evaluation,
    )

    assert len(ranked) == 1

    selected = select_top_spreads(
        ranked,
        n_spreads=1,
    )

    assert len(selected) == 1

    spread = create_spread(
        prices[selected[0].tickers],
        np.array(selected[0].beta),
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
