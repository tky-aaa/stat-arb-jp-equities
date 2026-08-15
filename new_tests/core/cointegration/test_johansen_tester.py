import numpy as np
import pandas as pd

from newstatarb.core.cointegration.johansen_tester import (
    JohansenTester,
)


def test_estimate_cointegration_detects_cointegration() -> None:
    rng = np.random.default_rng(42)

    n = 200

    common = np.cumsum(
        rng.normal(0.0, 1.0, n),
    )

    spread = np.zeros(n)

    for i in range(1, n):
        spread[i] = 0.5 * spread[i - 1] + rng.normal(0.0, 0.2)

    log_prices = pd.DataFrame(
        {
            "AAA": common,
            "BBB": common + spread,
            "CCC": (2.0 * common + 1.5 * spread + rng.normal(0.0, 0.05, n)),
        }
    )

    tester = JohansenTester()

    result = tester.estimate_cointegration(
        log_prices,
    )

    assert result.rank >= 1
    assert result.beta.shape[0] == 3
    assert result.beta.shape[1] == result.rank

    assert len(result.tickers) == 3
    assert result.var_lag >= 1
    assert result.k_ar_diff >= 0
