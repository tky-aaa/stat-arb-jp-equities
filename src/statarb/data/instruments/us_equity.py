from dataclasses import dataclass


@dataclass(frozen=True)
class USEquity:
    """
    U.S. equity instrument.
    """

    ticker: str

    def yahoo_contract(self) -> str:
        return self.ticker

    def alpaca_contract(self) -> str:
        return self.ticker

    def ibkr_contract(self):
        from ib_insync import Stock

        return Stock(
            self.ticker,
            "SMART",
            "USD",
        )
