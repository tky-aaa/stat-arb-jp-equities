from datetime import UTC, datetime
from pathlib import Path

from statarb.core.report.export import ReportExporter
from statarb.core.report.metrics import BacktestMetricsCalculator
from statarb.core.report.storage import ReportStorage


class ReportAPI:
    def __init__(
        self,
        results_path: str = "results",
        storage: ReportStorage | None = None,
        metrics_calculator: BacktestMetricsCalculator | None = None,
        exporter: ReportExporter | None = None,
    ):
        self.results_path = Path(results_path)
        self.storage = storage or ReportStorage()
        self.metrics_calculator = metrics_calculator or BacktestMetricsCalculator()
        self.exporter = exporter or ReportExporter()

    def create_experiment_path(self) -> Path:
        experiment_id = datetime.now(UTC).strftime(
            "%Y%m%d_%H%M%S_%f",
        )

        experiment_path = self.results_path / experiment_id

        experiment_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return experiment_path

    def save_checkpoint(
        self,
        experiment_path: Path,
        name: str,
        output: object,
    ) -> None:
        self.storage.save(
            output,
            experiment_path / "checkpoint" / f"{name}.pkl",
        )

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

        experiment_path = self.create_experiment_path()

        for name, output in outputs.items():
            self.storage.save(
                output,
                experiment_path / f"{name}.pkl",
            )

        self.exporter.export(
            experiment_path,
            outputs,
        )

        return experiment_path
