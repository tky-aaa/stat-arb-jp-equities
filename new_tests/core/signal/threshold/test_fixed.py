import pytest

from newstatarb.core.signal.threshold.fixed import FixedThreshold


def test_optimize_returns_configured_threshold() -> None:
    optimizer = FixedThreshold(
        entry_threshold=2.0,
    )

    threshold = optimizer.optimize()

    assert threshold == 2.0


def test_rejects_non_positive_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="entry_threshold must be positive",
    ):
        FixedThreshold(
            entry_threshold=0.0,
        )
