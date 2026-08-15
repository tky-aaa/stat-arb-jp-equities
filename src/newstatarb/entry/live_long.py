from newstatarb.config.config import Config
from newstatarb.orchestrator.live_long import LiveLongOrchestrator


def main() -> None:
    config = Config()
    blo = LiveLongOrchestrator(config)
    blo.run()


if __name__ == "__main__":
    main()
