import numpy as np
import pandas as pd

from newstatarb.config.contract import (
    CointegrationAnalysis,
    Prices,
    Spread,
    SpreadEvaluation,
)
from newstatarb.core.signal.api import SignalAPI


def test_signal_api_generates_kalman_signal() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=120,
        freq="D",
    )

    x1 = 100.0 + np.linspace(
        0.0,
        20.0,
        120,
    )

    x2 = 80.0 + 10.0 * np.sin(
        np.linspace(
            0.0,
            4.0 * np.pi,
            120,
        )
    )

    y = 2.0 + 0.5 * x1 - 0.3 * x2

    prices = Prices(
        training=pd.DataFrame(
            {
                "Y": y[:80],
                "X1": x1[:80],
                "X2": x2[:80],
            },
            index=index[:80],
        ),
        test=pd.DataFrame(
            {
                "Y": y[80:],
                "X1": x1[80:],
                "X2": x2[80:],
            },
            index=index[80:],
        ),
    )

    spread = Spread(
        tickers=["Y", "X1", "X2"],
        beta=np.array(
            [
                1.0,
                0.5,
                -0.3,
            ]
        ),
        beta_index=0,
        values=pd.Series(
            dtype=float,
        ),
    )

    evaluation = SpreadEvaluation(
        adf_stat=0.0,
        adf_pvalue=0.01,
        kpss_stat=0.0,
        kpss_pvalue=0.10,
        rho1=0.0,
        phi=0.0,
        half_life=1.0,
        persistence=0.0,
        mean=0.0,
        variance=1.0,
        std=1.0,
        portmanteau=0.0,
    )

    analysis = CointegrationAnalysis(
        tickers=["Y", "X1", "X2"],
        rank=1,
        beta_index=0,
        beta=np.array(
            [
                1.0,
                0.5,
                -0.3,
            ]
        ),
        spread=spread,
        evaluation=evaluation,
    )

    api = SignalAPI(
        zscore_window=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        spread_method="kalman",
        kalman_alpha=1e-5,
    )

    signals = api.service(
        prices,
        [analysis],
    )
    assert len(signals) == 1

    signal = signals[0]

    assert isinstance(signal.spread, pd.Series)
    assert isinstance(signal.zscore, pd.Series)
    assert isinstance(signal.position, pd.Series)

    assert len(signal.spread) == len(prices.test)
    assert len(signal.zscore) == len(prices.test)
    assert len(signal.position) == len(prices.test)

    pd.testing.assert_index_equal(
        signal.spread.index,
        prices.test.index,
    )

    pd.testing.assert_index_equal(
        signal.zscore.index,
        prices.test.index,
    )

    pd.testing.assert_index_equal(
        signal.position.index,
        prices.test.index,
    )

    assert np.isfinite(signal.spread.to_numpy()).all()

    assert signal.position.isin([-1, 0, 1]).all()


def test_signal_api_generates_fixed_signal() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=120,
        freq="D",
    )

    x1 = 100.0 + np.linspace(
        0.0,
        20.0,
        120,
    )

    x2 = 80.0 + 10.0 * np.sin(
        np.linspace(
            0.0,
            4.0 * np.pi,
            120,
        )
    )

    y = 2.0 + 0.5 * x1 - 0.3 * x2

    prices = Prices(
        training=pd.DataFrame(
            {
                "Y": y[:80],
                "X1": x1[:80],
                "X2": x2[:80],
            },
            index=index[:80],
        ),
        test=pd.DataFrame(
            {
                "Y": y[80:],
                "X1": x1[80:],
                "X2": x2[80:],
            },
            index=index[80:],
        ),
    )

    spread = Spread(
        tickers=["Y", "X1", "X2"],
        beta=np.array(
            [
                1.0,
                0.5,
                -0.3,
            ]
        ),
        beta_index=0,
        values=pd.Series(
            dtype=float,
        ),
    )

    evaluation = SpreadEvaluation(
        adf_stat=0.0,
        adf_pvalue=0.01,
        kpss_stat=0.0,
        kpss_pvalue=0.10,
        rho1=0.0,
        phi=0.0,
        half_life=1.0,
        persistence=0.0,
        mean=0.0,
        variance=1.0,
        std=1.0,
        portmanteau=0.0,
    )

    analysis = CointegrationAnalysis(
        tickers=["Y", "X1", "X2"],
        rank=1,
        beta_index=0,
        beta=np.array(
            [
                1.0,
                0.5,
                -0.3,
            ]
        ),
        spread=spread,
        evaluation=evaluation,
    )

    api = SignalAPI(
        zscore_window=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        spread_method="fixed",
    )

    signals = api.service(
        prices,
        [analysis],
    )

    assert len(signals) == 1

    signal = signals[0]

    assert isinstance(signal.spread, pd.Series)
    assert isinstance(signal.zscore, pd.Series)
    assert isinstance(signal.position, pd.Series)

    assert len(signal.spread) == len(prices.test)
    assert len(signal.zscore) == len(prices.test)
    assert len(signal.position) == len(prices.test)

    pd.testing.assert_index_equal(
        signal.spread.index,
        prices.test.index,
    )

    pd.testing.assert_index_equal(
        signal.zscore.index,
        prices.test.index,
    )

    pd.testing.assert_index_equal(
        signal.position.index,
        prices.test.index,
    )

    assert np.isfinite(signal.spread.to_numpy()).all()

    assert signal.position.isin([-1, 0, 1]).all()


def test_signal_api_rejects_unknown_spread_method() -> None:
    prices = Prices(
        training=pd.DataFrame(
            {
                "Y": [1.0, 2.0, 3.0],
                "X1": [1.0, 1.0, 1.0],
            }
        ),
        test=pd.DataFrame(
            {
                "Y": [4.0, 5.0, 6.0],
                "X1": [1.0, 1.0, 1.0],
            }
        ),
    )

    spread = Spread(
        tickers=["Y", "X1"],
        beta=np.array([1.0, -1.0]),
        beta_index=0,
        values=pd.Series(dtype=float),
    )

    evaluation = SpreadEvaluation(
        adf_stat=0.0,
        adf_pvalue=0.01,
        kpss_stat=0.0,
        kpss_pvalue=0.10,
        rho1=0.0,
        phi=0.0,
        half_life=1.0,
        persistence=0.0,
        mean=0.0,
        variance=1.0,
        std=1.0,
        portmanteau=0.0,
    )

    analysis = CointegrationAnalysis(
        tickers=["Y", "X1"],
        rank=1,
        beta_index=0,
        beta=np.array([1.0, -1.0]),
        spread=spread,
        evaluation=evaluation,
    )

    api = SignalAPI(
        zscore_window=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        spread_method="unknown",
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="Unknown spread_method",
    ):
        api.service(
            prices,
            [analysis],
        )


def test_signal_api_uses_gaussian_threshold() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=120,
        freq="D",
    )

    x1 = 100.0 + np.linspace(
        0.0,
        20.0,
        120,
    )

    x2 = 80.0 + 10.0 * np.sin(
        np.linspace(
            0.0,
            4.0 * np.pi,
            120,
        )
    )

    y = 2.0 + 0.5 * x1 - 0.3 * x2

    prices = Prices(
        training=pd.DataFrame(
            {
                "Y": y[:80],
                "X1": x1[:80],
                "X2": x2[:80],
            },
            index=index[:80],
        ),
        test=pd.DataFrame(
            {
                "Y": y[80:],
                "X1": x1[80:],
                "X2": x2[80:],
            },
            index=index[80:],
        ),
    )

    spread = Spread(
        tickers=["Y", "X1", "X2"],
        beta=np.array(
            [
                1.0,
                0.5,
                -0.3,
            ]
        ),
        beta_index=0,
        values=pd.Series(
            dtype=float,
        ),
    )

    evaluation = SpreadEvaluation(
        adf_stat=0.0,
        adf_pvalue=0.01,
        kpss_stat=0.0,
        kpss_pvalue=0.10,
        rho1=0.0,
        phi=0.0,
        half_life=1.0,
        persistence=0.0,
        mean=0.0,
        variance=1.0,
        std=1.0,
        portmanteau=0.0,
    )

    analysis = CointegrationAnalysis(
        tickers=["Y", "X1", "X2"],
        rank=1,
        beta_index=0,
        beta=np.array(
            [
                1.0,
                0.5,
                -0.3,
            ]
        ),
        spread=spread,
        evaluation=evaluation,
    )

    api = SignalAPI(
        zscore_window=20,
        entry_threshold=999.0,
        exit_threshold=0.5,
        spread_method="fixed",
        threshold_method="gaussian",
    )

    signals = api.service(
        prices,
        [analysis],
    )

    assert len(signals) == 1
