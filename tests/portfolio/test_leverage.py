import pytest

from statarb.portfolio.leverage import (
    gross_leverage,
    net_exposure,
    scale_to_leverage,
)


def test_gross_leverage():

    weights = [
        0.5,
        -0.5,
    ]

    assert gross_leverage(weights) == pytest.approx(1.0)


def test_net_exposure():

    weights = [
        0.5,
        -0.5,
    ]

    assert net_exposure(weights) == pytest.approx(0.0)


def test_scale_to_leverage():

    weights = [
        2.0,
        -1.0,
    ]

    result = scale_to_leverage(
        weights,
        target_leverage=1.0,
    )

    assert gross_leverage(result) == pytest.approx(1.0)
