import numpy as np
import pandas as pd

from statarb.core.screening.ff3 import FF3Estimator


def test_ff3_returns_factor_exposures() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    returns = pd.DataFrame(
        {
            "AAA": np.linspace(0.01, 0.20, 20),
            "BBB": np.linspace(0.02, 0.30, 20),
        },
        index=index,
    )

    factors = pd.DataFrame(
        {
            "MKT": np.linspace(0.01, 0.05, 20),
            "SMB": np.linspace(-0.02, 0.02, 20),
            "HML": np.linspace(0.03, -0.01, 20),
            "RF": np.full(20, 0.001),
        },
        index=index,
    )

    ff3 = FF3Estimator()

    exposures = ff3.estimate_exposure(
        returns,
        factors,
    )

    assert isinstance(exposures, pd.DataFrame)
    assert exposures.index.tolist() == ["AAA", "BBB"]
    assert exposures.columns.tolist() == [
        "alpha",
        "beta_mkt",
        "beta_smb",
        "beta_hml",
    ]
    assert np.isfinite(exposures.to_numpy()).all()
