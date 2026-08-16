import argparse

from statarb.config.config import LiveConfig
from statarb.orchestrator.live_short import LiveShortOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)

    args = parser.parse_args()

    config = LiveConfig()
    orchestrator = LiveShortOrchestrator(config)
    orchestrator.run(
        test_start=args.test_start,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
