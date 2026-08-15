from pathlib import Path

import matplotlib.pyplot as plt

from statarb.reports.figures import (
    save_figure,
)


def test_save_figure(tmp_path):

    fig, ax = plt.subplots()

    ax.plot(
        [0, 1],
        [0, 1],
    )

    path = tmp_path / "test.png"

    save_figure(
        fig,
        str(path),
    )

    assert Path(path).exists()
