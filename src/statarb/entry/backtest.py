from statarb.config.config import BacktestConfig
from statarb.orchestrator.backtest import BacktestOrchestrator


def main() -> None:
    config = BacktestConfig()
    bto = BacktestOrchestrator(config)
    bto.run()


if __name__ == "__main__":
    main()
