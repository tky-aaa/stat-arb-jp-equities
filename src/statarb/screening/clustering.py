import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_by_factor_exposure(
    exposures: pd.DataFrame,
    n_clusters: int,
) -> pd.Series:
    """
    Cluster instruments by Fama-French factor exposure.

    Parameters
    ----------
    exposures:
        Output from estimate_ff3_exposure.

        index:
            ticker

        columns:
            alpha
            beta_mkt
            beta_smb
            beta_hml

    n_clusters:
        Number of clusters.

    Returns
    -------
    pd.Series

        index:
            ticker

        values:
            cluster label
    """

    features = exposures[
        [
            "beta_mkt",
            "beta_smb",
            "beta_hml",
        ]
    ]

    # scale features
    scaler = StandardScaler()

    X = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
    )

    labels = model.fit_predict(X)

    return pd.Series(
        labels,
        index=features.index,
        name="cluster",
    )
