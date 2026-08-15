from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    equity: pd.Series,
    *,
    output_path: str = "results/equity_curve.png",
) -> Path:
    """
    Plot portfolio equity curve.
    """

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 4),
    )

    equity.plot()

    plt.title(
        "Equity Curve",
    )

    plt.xlabel(
        "Date",
    )

    plt.ylabel(
        "Equity",
    )

    plt.grid(
        True,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


def plot_spread(
    spread: pd.Series,
    *,
    output_path: str = "results/spread.png",
) -> Path:
    """
    Plot spread series.
    """

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 4),
    )

    spread.plot()

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.title(
        "Spread",
    )

    plt.xlabel(
        "Date",
    )

    plt.ylabel(
        "Spread",
    )

    plt.grid(
        True,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path
