from dataclasses import dataclass


@dataclass(frozen=True)
class JapanEquity:
    """
    Japanese equity instrument.
    """

    ticker: str

    def yahoo_contract(self) -> str:
        return f"{self.ticker}.T"

    def ibkr_contract(self):
        from ib_insync import Stock

        return Stock(
            self.ticker,
            "TSEJ",
            "JPY",
        )
