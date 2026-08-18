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
            if not np.isfinite(evaluation.persistence):
                continue
            if not np.isfinite(evaluation.portmanteau):
                continue
            filtered.append(analysis)

        half_life_values = sorted(
            {analysis.evaluation.half_life for analysis in filtered}
        )
        half_life_rank = {
            value: rank
            for rank, value in enumerate(
                half_life_values,
                start=1,
            )
        }

        persistence_values = sorted(
            {analysis.evaluation.persistence for analysis in filtered},
            reverse=True,
        )
        persistence_rank = {
            value: rank
            for rank, value in enumerate(
                persistence_values,
                start=1,
            )
        }

        portmanteau_values = sorted(
            {analysis.evaluation.portmanteau for analysis in filtered},
            reverse=True,
        )
        portmanteau_rank = {
            value: rank
            for rank, value in enumerate(
                portmanteau_values,
                start=1,
            )
        }

        filtered.sort(
            key=lambda analysis: (
                half_life_rank[analysis.evaluation.half_life]
                + persistence_rank[analysis.evaluation.persistence]
                + portmanteau_rank[analysis.evaluation.portmanteau],
            )
        )

        return filtered[:top_n]
