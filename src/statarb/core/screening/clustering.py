import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Clusterizer:
    def cluster(
        self,
        features: pd.DataFrame,
        n_clusters: int,
        *,
        scale: bool = False,
    ) -> pd.Series:
        X = features

        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

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
