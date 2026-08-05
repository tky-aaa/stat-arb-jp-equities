import random

from .candidate import CandidateGroup


def generate_random_candidate_groups(
    tickers: list[str],
    *,
    group_size: int = 20,
    n_groups: int = 100,
    seed: int = 42,
):
    rng = random.Random(seed)

    groups = []

    for _ in range(n_groups):
        sample = rng.sample(
            tickers,
            group_size,
        )

        groups.append(
            CandidateGroup(
                tickers=sample,
            )
        )

    return groups
