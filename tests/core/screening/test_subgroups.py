from statarb.config.contract import Group, SubGroup
from statarb.core.screening.subgroups import SubgroupGenerator


def test_generate_subgroups_returns_all_combinations() -> None:
    groups = [
        Group(
            tickers=["AAA", "BBB", "CCC"],
        )
    ]

    generator = SubgroupGenerator()

    subgroups = generator.generate(
        groups,
        min_assets=2,
        max_assets=2,
    )

    assert subgroups == [
        SubGroup(tickers=["AAA", "BBB"]),
        SubGroup(tickers=["AAA", "CCC"]),
        SubGroup(tickers=["BBB", "CCC"]),
    ]
