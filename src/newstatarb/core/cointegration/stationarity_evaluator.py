import warnings

from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from newstatarb.config.contract import Spread


class StationarityEvaluator:
    def evaluate(
        self,
        spread: Spread,
    ) -> dict[str, float]:
        values = spread.values.dropna().astype(float)

        if len(values) < 50:
            raise ValueError("Too few observations.")

        adf_result = adfuller(
            values,
            autolag="AIC",
        )

        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                InterpolationWarning,
            )

            kpss_result = kpss(
                values,
                regression="c",
                nlags="auto",
            )

        kpss_stat = float(kpss_result[0])
        kpss_pvalue = float(kpss_result[1])

        return {
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "kpss_stat": kpss_stat,
            "kpss_pvalue": kpss_pvalue,
        }
