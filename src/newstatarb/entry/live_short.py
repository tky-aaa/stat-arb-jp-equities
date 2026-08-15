from newstatarb.config.config import Config
from newstatarb.orchestrator.live_short import LiveShortOrchestrator


def main() -> None:
    config = Config()
    bso = LiveShortOrchestrator(config)
    bso.run()


if __name__ == "__main__":
    main()
