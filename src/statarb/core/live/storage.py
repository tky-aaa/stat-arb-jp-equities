import pickle
from pathlib import Path

from statarb.config.contract import LiveStrategy


class LiveStrategyStorage:
    def save(
        self,
        strategy: LiveStrategy,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("wb") as file:
            pickle.dump(strategy, file)

    def load(
        self,
        path: Path,
    ) -> LiveStrategy:
        with path.open("rb") as file:
            strategy = pickle.load(file)

        if not isinstance(strategy, LiveStrategy):
            raise TypeError("Invalid live strategy.")

        return strategy
