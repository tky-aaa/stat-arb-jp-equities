from dataclasses import dataclass

from newstatarb.config.contract import (
    CointegrationAnalysis,
    PortfolioDecision,
    Prices,
)
from newstatarb.core.portfolio.allocator import PortfolioAllocator
from newstatarb.core.portfolio.leverage import LeverageCalculator
from newstatarb.core.portfolio.selector import PortfolioSelector
from newstatarb.core.signal.api import SignalAPI


@dataclass(frozen=True)
class PortfolioAPI:
    signal_api: SignalAPI
    top_n_spreads: int
    max_weight: float = 0.2
    max_leverage: float = 1.0

    def service(
        self,
        prices: Prices,
        analyses: list[CointegrationAnalysis],
    ) -> PortfolioDecision:

        selector = PortfolioSelector()
        allocator = PortfolioAllocator()
        leverage = LeverageCalculator()

        selected = selector.select(
            analyses,
            top_n=self.top_n_spreads,
        )

        signals = self.signal_api.service(
            prices,
            selected,
        )

        weights = allocator.allocate(
            selected,
            max_weight=self.max_weight,
        )

        weights = leverage.scale_to_leverage(
            weights,
            target_leverage=min(
                self.max_leverage,
                sum(abs(weight) for weight in weights),
            ),
        )

        return PortfolioDecision(
            analyses=selected,
            signals=signals,
            weights=weights,
        )
