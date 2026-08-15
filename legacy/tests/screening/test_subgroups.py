from statarb.screening.candidate import CandidateGroup
from statarb.screening.subgroups import enumerate_subgroups


def test_enumerate_subgroups():
    group = CandidateGroup(
        tickers=[
            "A",
            "B",
            "C",
            "D",
        ]
    )

    result = enumerate_subgroups(
        group,
        min_assets=2,
        max_assets=3,
    )

    assert len(result) == 10

    assert (
        CandidateGroup(
            tickers=["A", "B"],
        )
        in result
    )

    assert (
        CandidateGroup(
            tickers=["A", "B", "C"],
        )
        in result
    )
