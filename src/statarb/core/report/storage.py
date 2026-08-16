import pickle
from pathlib import Path


class ReportStorage:
    def save(
        self,
        output: object,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("wb") as file:
            pickle.dump(output, file)

    def load(
        self,
        path: Path,
    ) -> object:
        with path.open("rb") as file:
            return pickle.load(file)
