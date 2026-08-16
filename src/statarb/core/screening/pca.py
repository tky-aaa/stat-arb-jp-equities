import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAComputer:
    def compute_features(
        self,
        returns: pd.DataFrame,
        n_components: int,
    ) -> pd.DataFrame:
        returns = returns.dropna()

        X = returns.T

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(
            n_components,
            X_scaled.shape[1],
        )

        pca = PCA(
            n_components=n_components,
        )

        exposure = pca.fit_transform(X_scaled)

        columns = [f"PC{i + 1}" for i in range(exposure.shape[1])]

        return pd.DataFrame(
            exposure,
            index=X.index,
            columns=columns,
        )
