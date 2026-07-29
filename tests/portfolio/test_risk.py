import pytest

from statarb.portfolio.risk import (
    apply_leverage_limit,
    cap_weights,
)


def test_cap_weights():

    weights = [
        0.6,
        0.3,
        0.1,
    ]

    result = cap_weights(
        weights,
        max_weight=0.5,
    )

    assert max(result) <= 0.5

    assert sum(result) == pytest.approx(1.0)


def test_apply_leverage_limit():

    weights = [
        0.8,
        -0.8,
    ]

    result = apply_leverage_limit(
        weights,
        max_leverage=1.0,
    )

    assert sum(abs(x) for x in result) == pytest.approx(1.0)
