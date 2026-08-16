from statarb.config.contract import Spread


class DistributionEvaluator:
    def evaluate(
        self,
        spread: Spread,
    ) -> dict[str, float]:
        values = spread.values.dropna().astype(float)

        if len(values) < 2:
            raise ValueError("Too few observations.")

        return {
            "mean": float(values.mean()),
            "variance": float(values.var()),
            "std": float(values.std()),
        }
