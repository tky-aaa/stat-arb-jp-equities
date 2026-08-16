from datetime import UTC, datetime
from pathlib import Path

from statarb.core.report.metrics import BacktestMetricsCalculator
from statarb.core.report.storage import ReportStorage


class ReportAPI:
    def __init__(
        self,
        results_path: str = "results",
        storage: ReportStorage | None = None,
        metrics_calculator: BacktestMetricsCalculator | None = None,
    ):
        self.results_path = Path(results_path)
        self.storage = storage or ReportStorage()
        self.metrics_calculator = metrics_calculator or BacktestMetricsCalculator()

    def service(
        self,
        outputs: dict[str, object],
    ) -> Path:
        if not outputs:
            raise ValueError("outputs must not be empty.")

        outputs = dict(outputs)

        if "backtest" in outputs and "portfolio" in outputs:
            decision = outputs["portfolio"]
            result = outputs["backtest"]

            outputs["metrics"] = self.metrics_calculator.calculate(
                result,
                decision.signals,
            )

        experiment_id = datetime.now(UTC).strftime(
            "%Y%m%d_%H%M%S_%f",
        )

        experiment_path = self.results_path / experiment_id

        for name, output in outputs.items():
            self.storage.save(
                output,
                experiment_path / f"{name}.pkl",
            )

        return experiment_path
