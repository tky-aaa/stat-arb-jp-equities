import numpy as np

from statarb.config.contract import CointegrationAnalysis


class PortfolioSelector:
    def select(
        self,
        analyses: list[CointegrationAnalysis],
        *,
        top_n: int,
        adf_threshold: float = 0.05,
        kpss_threshold: float = 0.05,
    ) -> list[CointegrationAnalysis]:

        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        filtered = []

        for analysis in analyses:
            evaluation = analysis.evaluation

            if evaluation.adf_pvalue >= adf_threshold:
                continue

            if evaluation.kpss_pvalue <= kpss_threshold:
                continue

            if not np.isfinite(evaluation.half_life):
                continue

            if not np.isfinite(evaluation.portmanteau):
                continue

            filtered.append(analysis)

        filtered.sort(
            key=lambda analysis: (
                analysis.evaluation.half_life,
                -analysis.evaluation.portmanteau,
            )
        )

        return filtered[:top_n]
