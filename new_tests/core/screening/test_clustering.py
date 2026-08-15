import pandas as pd

from newstatarb.core.screening.clustering import Clusterizer


def test_clusterizer_returns_cluster_labels() -> None:
    exposures = pd.DataFrame(
        {
            "PC1": [0.0, 0.1, 10.0, 10.1],
            "PC2": [0.0, 0.1, 10.0, 10.1],
        },
        index=["AAA", "BBB", "CCC", "DDD"],
    )

    clusterizer = Clusterizer()

    clusters = clusterizer.cluster(
        exposures,
        n_clusters=2,
    )

    assert isinstance(clusters, pd.Series)
    assert clusters.index.tolist() == [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
    ]
    assert clusters.name == "cluster"
    assert clusters.nunique() == 2
