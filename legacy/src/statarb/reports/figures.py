from pathlib import Path

import matplotlib.figure


def save_figure(
    fig: matplotlib.figure.Figure,
    path: str,
) -> None:
    """
    Save matplotlib figure.
    """

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        bbox_inches="tight",
    )
