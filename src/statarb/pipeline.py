from statarb.reports.export import (
    save_selected_spreads,
    save_summary,
)
from statarb.research.pipeline import run_research_pipeline


def main() -> None:
    results = run_research_pipeline()

    print("=" * 80)
    print("Research pipeline finished")
    print("=" * 80)

    print(f"candidate groups : {len(results['candidate_groups'])}")

    print(f"evaluation rows  : {len(results['evaluation'])}")

    print(f"ranking rows     : {len(results['ranking'])}")

    print(f"selected spreads : {len(results['selected'])}")

    print()

    print("=" * 80)
    print("Saving results")
    print("=" * 80)

    summary_fixed_path = save_summary(
        results["summary_fixed"],
        filename="summary_fixed.csv",
    )

    summary_gaussian_path = save_summary(
        results["summary_gaussian"],
        filename="summary_gaussian.csv",
    )

    summary_empirical_path = save_summary(
        results["summary_empirical"],
        filename="summary_empirical.csv",
    )

    summary_kalman_path = save_summary(
        results["summary_kalman"],
        filename="summary_kalman.csv",
    )

    spreads_path = save_selected_spreads(
        results["selected"],
    )

    print(f"summary fixed      : {summary_fixed_path}")
    print(f"summary gaussian   : {summary_gaussian_path}")
    print(f"summary empirical  : {summary_empirical_path}")
    print(f"summary kalman     : {summary_kalman_path}")
    print(f"selected spreads   : {spreads_path}")


if __name__ == "__main__":
    main()
