from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold-cost sensitivity analysis.")
    p.add_argument(
        "--groupkfold",
        type=Path,
        default=ROOT / "outputs" / "groupkfold" / "Optuna_calibrated_predictions.csv",
    )
    p.add_argument(
        "--timesplit",
        type=Path,
        default=ROOT / "outputs" / "timesplit" / "TimeSplit_test_predictions.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "threshold_cost",
    )
    p.add_argument("--ratios", nargs="+", type=float, default=[5, 10, 20])
    p.add_argument("--min-threshold", type=float, default=0.02)
    p.add_argument("--max-threshold", type=float, default=0.98)
    p.add_argument("--step", type=float, default=0.01)
    return p.parse_args()


def cost_analysis(y_true: np.ndarray, prob: np.ndarray, label: str, ratios: list[float], thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for ratio in ratios:
        best = None
        for threshold in thresholds:
            pred = (prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            total_cost = ratio * fn + fp
            if best is None or total_cost < best["total_cost"]:
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                best = {
                    "dataset": label,
                    "cost_ratio_FN:FP": f"{ratio:g}:1",
                    "best_threshold": threshold,
                    "Recall": recall,
                    "Precision": precision,
                    "F1": f1,
                    "Type_I_Error": fp / (fp + tn) if fp + tn else np.nan,
                    "Type_II_Error": fn / (fn + tp) if fn + tp else np.nan,
                    "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                    "total_cost": total_cost,
                }
        if best is not None:
            rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.arange(args.min_threshold, args.max_threshold + 1e-12, args.step)

    gk = pd.read_csv(args.groupkfold, encoding="utf-8-sig")
    gk_prob_col = "y_prob_calibrated" if "y_prob_calibrated" in gk.columns else "y_prob_raw"
    gk_out = cost_analysis(
        gk["y_true"].to_numpy(dtype=int),
        gk[gk_prob_col].to_numpy(),
        "GroupKFold",
        args.ratios,
        thresholds,
    )
    gk_out.to_csv(
        args.output_dir / "Threshold_Cost_Ratios_GroupKFold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_results = [gk_out]
    if args.timesplit.exists():
        ts = pd.read_csv(args.timesplit, encoding="utf-8-sig")
        ts_prob_col = "prob_cal" if "prob_cal" in ts.columns else "prob_raw"
        ts_out = cost_analysis(
            ts["y_true"].to_numpy(dtype=int),
            ts[ts_prob_col].to_numpy(),
            "TimeSplit Test",
            args.ratios,
            thresholds,
        )
        ts_out.to_csv(
            args.output_dir / "Threshold_Cost_Ratios_TimeSplit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_results.append(ts_out)

    pd.concat(all_results, ignore_index=True).to_csv(
        args.output_dir / "Threshold_Cost_Ratios_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.concat(all_results, ignore_index=True).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
