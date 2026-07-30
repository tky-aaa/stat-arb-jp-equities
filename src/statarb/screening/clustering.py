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


def cluster_by_pca(
    pca_features,
    *,
    n_clusters: int = 20,
):
    """
    Cluster assets using PCA features.

    Parameters
    ----------
    pca_features
        DataFrame

        index:
            ticker

        columns:
            PC1, PC2, ...

    Returns
    -------
    pd.Series

    index:
        ticker

    values:
        cluster label
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init="auto",
    )

    labels = model.fit_predict(
        pca_features.values,
    )

    return pca_features.index.to_series().map(dict(zip(pca_features.index, labels)))
