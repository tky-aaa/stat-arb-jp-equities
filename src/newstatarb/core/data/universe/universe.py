from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UNIVERSE_PATH = Path(__file__).resolve().parents[3] / "config" / "universe"


@dataclass(frozen=True)
class Universe:
    name: str

    def get_tickers(self) -> list[str]:
        match self.name:
            case "topix10":
                return self._load("topix10.csv")
            case "topix100":
                return self._load("topix100.csv")
            case "topix500":
                return self._load("topix500.csv")
            case _:
                raise ValueError(f"Unknown universe: {self.name}")

    def _load(self, filename: str) -> list[str]:
        path = UNIVERSE_PATH / filename

        df = pd.read_csv(path)

        return df["ticker"].astype(str).tolist()
