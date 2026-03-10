from __future__ import annotations

import json

from ramsey_two_asset import Calibration, ExperimentSpec, run_experiment


def main() -> None:
    calibration = Calibration()
    experiment = ExperimentSpec()
    results = run_experiment(calibration, experiment)
    summary = {
        "baseline_history_len": len(results["baseline"]["history"]) if results["baseline"] is not None else 0,
        "comparative_static_cases": sorted(results["comparative_statics"].keys()),
        "log_benchmark": (
            {k: v for k, v in results["log_benchmark"]["benchmark"].items() if "diff" in k}
            if results["log_benchmark"] is not None
            else None
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
