import numpy as np
import pandas as pd

from statarb.screening.pca import compute_pca_features


def test_compute_pca_features():

    np.random.seed(0)

    returns = pd.DataFrame(
        np.random.normal(
            size=(200, 20),
        ),
        columns=[f"T{i}" for i in range(20)],
    )

    features = compute_pca_features(
        returns,
        n_components=5,
    )

    assert features.shape == (20, 5)

    assert list(features.columns) == [
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
    ]
