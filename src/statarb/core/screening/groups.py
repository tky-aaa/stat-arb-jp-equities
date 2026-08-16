import pandas as pd

from statarb.config.contract import Group


class GroupGenerator:
    def generate(
        self,
        clusters: pd.Series,
        *,
        min_size: int,
        max_size: int,
    ) -> list[Group]:
        groups = []

        for cluster_id in clusters.unique():
            tickers = clusters[clusters == cluster_id].index.tolist()

            size = len(tickers)

            if min_size <= size <= max_size:
                groups.append(
                    Group(
                        tickers=tickers,
                    )
                )

        return groups
