#!/usr/bin/env python3
"""Create compact, machine-readable summaries of paper reproduction outputs."""
import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path("em_opm/runs")


def read(path):
    return json.loads(Path(path).read_text())


def collision_summary():
    rows = [read(RUNS / f"collision_v2_seed{seed}/test_metrics.json")
            for seed in (11, 23, 31)]
    keys = ("AUROC", "AUPRC", "Brier", "ECE_10")
    out = {
        "seeds": [11, 23, 31],
        "runs": rows,
        "mean_std": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows], ddof=1)),
            }
            for key in keys
        },
    }
    target = RUNS / "collision_v2/multiseed_metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    return out


def threshold_summary():
    rows = []
    for path in sorted((RUNS / "tuning").glob("collision_threshold_*/metrics.json")):
        row = read(path)
        row["selection_score"] = (
            row["success_rate"] - row["collision_rate"]
            - row["boundary_rate"] - 0.25 * row["timeout_rate"]
        )
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No threshold-tuning metrics found")
    selected = max(rows, key=lambda row: row["selection_score"])
    out = {"criterion": "success-collision-boundary-0.25*timeout",
           "selected_threshold": selected["collision_threshold"], "runs": rows}
    (RUNS / "collision_threshold_selection.json").write_text(json.dumps(out, indent=2))
    return out


def paper_summary():
    paths = {
        "open_loop_il": RUNS / "il_v1/test_metrics.json",
        "boundary_prediction": RUNS / "risk_v1/test_metrics.json",
        "collision_prediction": RUNS / "collision_v2/multiseed_metrics.json",
        "closed_loop_il": RUNS / "closed_loop_il/metrics.json",
        "closed_loop_risk": RUNS / "closed_loop_risk/metrics.json",
        "closed_loop_route_upper": RUNS / "closed_loop_route_upper/metrics.json",
        "dynamic_none": RUNS / "dynamic_closed_loop_none/metrics.json",
        "dynamic_brake": RUNS / "dynamic_closed_loop_brake/metrics.json",
        "dynamic_geometric": RUNS / "dynamic_closed_loop_geometric/metrics.json",
        "dynamic_learned": RUNS / "dynamic_closed_loop_learned/metrics.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing paper results:\n" + "\n".join(missing))
    out = {name: read(path) for name, path in paths.items()}
    (RUNS / "paper_results.json").write_text(json.dumps(out, indent=2))
    return out


def check():
    required = [
        "em_opm/data/il_v1/manifest.json",
        "em_opm/data/risk_v1/manifest.json",
        "em_opm/data/dynamic_actors_v1/summary.json",
        "em_opm/data/collision_v2/manifest.json",
        "em_opm/runs/il_v1/best.pt",
        "em_opm/runs/risk_v1/best.pt",
        "em_opm/runs/collision_v2/best.pt",
    ]
    missing = [path for path in required if not Path(path).is_file()]
    out = {"required_artifacts": len(required), "missing": missing, "ok": not missing}
    if missing:
        raise FileNotFoundError("Missing artifacts:\n" + "\n".join(missing))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collision", "threshold", "paper", "check"))
    mode = parser.parse_args().mode
    result = {"collision": collision_summary, "threshold": threshold_summary,
              "paper": paper_summary, "check": check}[mode]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

