from itertools import combinations

from statarb.config.contract import Group, SubGroup


class SubgroupGenerator:
    def generate(
        self,
        groups: list[Group],
        *,
        min_assets: int,
        max_assets: int,
    ) -> list[SubGroup]:
        subgroups = []

        for group in groups:
            tickers = group.tickers

            upper = min(
                max_assets,
                len(tickers),
            )

            for size in range(
                min_assets,
                upper + 1,
            ):
                for subset in combinations(
                    tickers,
                    size,
                ):
                    subgroups.append(
                        SubGroup(
                            tickers=list(subset),
                        )
                    )

        return subgroups
