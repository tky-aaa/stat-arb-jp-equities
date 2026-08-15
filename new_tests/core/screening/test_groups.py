import pandas as pd

from newstatarb.config.contract import Group
from newstatarb.core.screening.groups import GroupGenerator


def test_generate_groups_filters_by_cluster_size() -> None:
    clusters = pd.Series(
        [0, 0, 1, 1, 1, 2],
        index=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        name="cluster",
    )

    generator = GroupGenerator()

    groups = generator.generate(
        clusters,
        min_size=2,
        max_size=3,
    )

    assert groups == [
        Group(tickers=["AAA", "BBB"]),
        Group(tickers=["CCC", "DDD", "EEE"]),
    ]
