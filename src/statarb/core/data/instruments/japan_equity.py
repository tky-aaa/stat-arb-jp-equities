from dataclasses import dataclass


@dataclass(frozen=True)
class JapanEquity:
    ticker: str

    def yahoo_symbol(self) -> str:
        return f"{self.ticker}.T"
