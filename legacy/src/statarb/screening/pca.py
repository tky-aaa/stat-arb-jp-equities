from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_features(
    returns: pd.DataFrame,
    *,
    n_components: int = 10,
) -> pd.DataFrame:
    """
    Compute PCA factor exposure for each asset.

    Parameters
    ----------
    returns
        DataFrame
        index:
            datetime

        columns:
            ticker

    n_components
        Number of principal components.

    Returns
    -------
    DataFrame

    index:
        ticker

    columns:
        PC1
        PC2
        ...
    """

    returns = returns.dropna()

    X = returns.T

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=min(
            n_components,
            X_scaled.shape[1],
        )
    )

    exposure = pca.fit_transform(X_scaled)

    columns = [f"PC{i + 1}" for i in range(exposure.shape[1])]

    return pd.DataFrame(
        exposure,
        index=X.index,
        columns=columns,
    )
