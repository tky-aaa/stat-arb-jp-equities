import pandas as pd
import statsmodels.api as sm


def estimate_ff3_exposure(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate Fama-French 3 factor exposures.

    Parameters
    ----------
    returns:
        Asset returns.

        index:
            datetime

        columns:
            ticker

    factors:
        Fama-French factors.

        columns:
            MKT
            SMB
            HML
            RF

    Returns
    -------
    pd.DataFrame

        index:
            ticker

        columns:
            alpha
            beta_mkt
            beta_smb
            beta_hml
    """

    exposures = {}

    for ticker in returns.columns:
        df = pd.concat(
            [
                returns[ticker],
                factors,
            ],
            axis=1,
            join="inner",
        ).dropna()

        df.columns = [
            "return",
            "MKT",
            "SMB",
            "HML",
            "RF",
        ]

        # excess return
        y = df["return"] - df["RF"]

        X = df[
            [
                "MKT",
                "SMB",
                "HML",
            ]
        ]

        X = sm.add_constant(X)

        model = sm.OLS(
            y,
            X,
        ).fit()

        exposures[ticker] = {
            "alpha": model.params["const"],
            "beta_mkt": model.params["MKT"],
            "beta_smb": model.params["SMB"],
            "beta_hml": model.params["HML"],
        }

    return pd.DataFrame(exposures).T
