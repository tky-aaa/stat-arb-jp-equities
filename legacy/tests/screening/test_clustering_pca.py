import numpy as np
import pandas as pd

from statarb.screening.clustering import cluster_by_pca


def test_cluster_by_pca():

    np.random.seed(0)

    features = pd.DataFrame(
        np.random.normal(size=(30, 5)),
        index=[f"T{i}" for i in range(30)],
        columns=[
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "PC5",
        ],
    )

    labels = cluster_by_pca(
        features,
        n_clusters=5,
    )

    assert len(labels) == 30

    assert labels.index.equals(features.index)

    assert labels.nunique() == 5
