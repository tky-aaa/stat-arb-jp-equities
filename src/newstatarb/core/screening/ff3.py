import pandas as pd
import statsmodels.api as sm


class FF3Estimator:
    def estimate_exposure(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
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
