from newstatarb.config.config import Config
from newstatarb.core.cointegration.api import CointegrationAPI
from newstatarb.core.data.api import DataAPI
from newstatarb.core.screening.api import ScreeningAPI


class LiveLongOrchestrator:
    data_api: DataAPI
    screening_api: ScreeningAPI
    cointegration_api: CointegrationAPI

    def __init__(self, config: Config):
        self.data_api = DataAPI(config.data)
        self.screening_api = ScreeningAPI(config.screening)
        self.cointegration_api = CointegrationAPI(config.cointegration)

    def run(self):
        pass
