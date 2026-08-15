from newstatarb.config.config import LiveConfig
from newstatarb.orchestrator.live_long import LiveLongOrchestrator


def main() -> None:
    config = LiveConfig()
    orchestrator = LiveLongOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
