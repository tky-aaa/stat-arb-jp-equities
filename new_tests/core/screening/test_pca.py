import pandas as pd

from newstatarb.core.screening.pca import PCAComputer


def test_pca_returns_exposures() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.01, 0.03],
            "BBB": [0.02, 0.01, 0.03, 0.02],
            "CCC": [0.03, 0.02, 0.04, 0.01],
        }
    )

    pca = PCAComputer()

    exposures = pca.compute_features(
        returns,
        n_components=2,
    )

    assert isinstance(exposures, pd.DataFrame)
    assert exposures.index.tolist() == [
        "AAA",
        "BBB",
        "CCC",
    ]
    assert exposures.columns.tolist() == [
        "PC1",
        "PC2",
    ]
    assert exposures.shape == (3, 2)
