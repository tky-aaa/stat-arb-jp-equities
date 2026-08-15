from newstatarb.config.config import Config
from newstatarb.core.live.api import LiveAPI
from newstatarb.core.portfolio.api import PortfolioAPI
from newstatarb.core.signal.api import SignalAPI


class LiveShortOrchestrator:
    signal_api: SignalAPI
    portfolio_api: PortfolioAPI
    live_api: LiveAPI

    def __init__(self, config: Config):
        self.signal_api = SignalAPI(config.signal)
        self.portfolio_api = PortfolioAPI(config.portfolio)

    def run(self):
        pass
