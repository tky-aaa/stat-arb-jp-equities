import pytest

from statarb.core.portfolio.leverage import LeverageCalculator


def test_gross_leverage() -> None:
    calculator = LeverageCalculator()

    result = calculator.gross_leverage(
        [0.2, -0.3, 0.5],
    )

    assert result == pytest.approx(1.0)


def test_net_exposure() -> None:
    calculator = LeverageCalculator()

    result = calculator.net_exposure(
        [0.2, -0.3, 0.5],
    )

    assert result == pytest.approx(0.4)


def test_scale_to_leverage() -> None:
    calculator = LeverageCalculator()

    result = calculator.scale_to_leverage(
        [0.2, -0.3, 0.5],
        target_leverage=2.0,
    )

    assert result == pytest.approx(
        [0.4, -0.6, 1.0],
    )


def test_scale_to_leverage_preserves_relative_weights() -> None:
    calculator = LeverageCalculator()

    weights = [0.1, -0.2, 0.3]

    result = calculator.scale_to_leverage(
        weights,
        target_leverage=1.5,
    )

    ratio = [result[i] / weights[i] for i in range(len(weights))]

    assert ratio == pytest.approx(
        [2.5, 2.5, 2.5],
    )


def test_scale_to_leverage_rejects_nonpositive_target() -> None:
    calculator = LeverageCalculator()

    with pytest.raises(
        ValueError,
        match="target_leverage must be positive",
    ):
        calculator.scale_to_leverage(
            [0.2, -0.3],
            target_leverage=0.0,
        )


def test_scale_to_leverage_rejects_zero_portfolio() -> None:
    calculator = LeverageCalculator()

    with pytest.raises(
        ValueError,
        match="Cannot scale zero portfolio",
    ):
        calculator.scale_to_leverage(
            [0.0, 0.0],
            target_leverage=1.0,
        )


def test_gross_and_net_leverage_for_empty_weights() -> None:
    calculator = LeverageCalculator()

    assert calculator.gross_leverage([]) == pytest.approx(0.0)
    assert calculator.net_exposure([]) == pytest.approx(0.0)
