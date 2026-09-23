from __future__ import annotations

import argparse
from pathlib import Path

# Make direct execution from the repository root import the local src package.
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.common import best_f1_threshold, bootstrap_ci, delong_test

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap, DeLong and McNemar tests.")
    p.add_argument(
        "--base-models",
        type=Path,
        default=ROOT / "outputs" / "groupkfold" / "Base_vs_Full_OOF_predictions.csv",
    )
    p.add_argument(
        "--groupkfold-final",
        type=Path,
        default=ROOT / "outputs" / "groupkfold" / "Optuna_calibrated_predictions.csv",
    )
    p.add_argument(
        "--stacking",
        type=Path,
        default=ROOT / "outputs" / "stacking" / "Stacking_OOF.csv",
    )
    p.add_argument(
        "--timesplit",
        type=Path,
        default=ROOT / "outputs" / "timesplit" / "TimeSplit_test_predictions.csv",
    )
    p.add_argument(
        "--smote",
        type=Path,
        default=ROOT / "outputs" / "smote" / "SMOTE_OOF_predictions.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "statistics",
    )
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def mcnemar_p(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    # Match the manuscript workflow: use the F1-optimal threshold of model 1
    # for both sets of classification decisions in the paired comparison.
    a = (p1 >= threshold).astype(int)
    b = (p2 >= threshold).astype(int)
    discord_01 = int(((a == 0) & (b == 1)).sum())
    discord_10 = int(((a == 1) & (b == 0)).sum())
    n = discord_01 + discord_10
    p_value = 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(discord_01, discord_10), n, 0.5)))
    return {
        "discordant_model_1_only": discord_10,
        "discordant_model_2_only": discord_01,
        "p_value": p_value,
    }


def ci_table(y: np.ndarray, probabilities: dict[str, np.ndarray], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for label, prob in probabilities.items():
        for metric, metric_name in [
            ("auc", "AUC"),
            ("pr_auc", "PR_AUC"),
            ("brier", "Brier"),
        ]:
            lo, hi = bootstrap_ci(y, prob, metric=metric, n_boot=n_boot, seed=seed)
            rows.append({
                "Model": label,
                "Metric": metric_name,
                "CI_lower": lo,
                "CI_upper": hi,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TimeSplit CIs use the final test predictions.
    ts = pd.read_csv(args.timesplit, encoding="utf-8-sig")
    y_ts = ts["y_true"].to_numpy(dtype=int)
    ts_probs = {}
    if "prob_raw" in ts.columns:
        ts_probs["TimeSplit raw"] = ts["prob_raw"].to_numpy()
    if "prob_cal" in ts.columns:
        ts_probs["TimeSplit calibrated"] = ts["prob_cal"].to_numpy()
    ci_table(y_ts, ts_probs, args.bootstrap, args.seed).to_csv(
        args.output_dir / "Bootstrap_CI_TimeSplit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # GroupKFold CIs use the same Optuna-calibrated OOF source as the manuscript.
    gk = pd.read_csv(args.groupkfold_final, encoding="utf-8-sig")
    y_gk = gk["y_true"].to_numpy(dtype=int)
    gk_probs = {
        "GroupKFold raw": gk["y_prob_raw"].to_numpy(),
        "GroupKFold calibrated": gk["y_prob_calibrated"].to_numpy(),
    }
    ci_table(y_gk, gk_probs, args.bootstrap, args.seed).to_csv(
        args.output_dir / "Bootstrap_CI_GroupKFold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # DeLong comparisons among the standalone model OOF predictions.
    if args.base_models.exists():
        base = pd.read_csv(args.base_models, encoding="utf-8-sig")
        y = base["y_true"].to_numpy(dtype=int)
        pairs = [
            ("Full LightGBM vs Full XGBoost", "Full_LightGBM", "Full_XGBoost"),
            ("Full LightGBM vs Full Random Forest", "Full_LightGBM", "Full_RandomForest"),
            ("Full LightGBM vs Full Lasso-LR", "Full_LightGBM", "Full_Lasso-LR"),
        ]
        rows = []
        for name, a, b in pairs:
            if a in base.columns and b in base.columns:
                out = delong_test(y, base[a].to_numpy(), base[b].to_numpy())
                out["Comparison"] = name
                rows.append(out)
        if rows:
            pd.DataFrame(rows).to_csv(
                args.output_dir / "DeLong_Base_Models.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # Stacking comparisons are against the internal OOF base learners used by Stacking.
    if args.stacking.exists():
        stk = pd.read_csv(args.stacking, encoding="utf-8-sig")
        y = stk["y_true"].to_numpy(dtype=int)
        rows = []
        for name, a, b in [
            ("Stacking vs internal LightGBM", "stack", "lgb"),
            ("Stacking vs internal XGBoost", "stack", "xgb"),
            ("Stacking vs internal Random Forest", "stack", "rf"),
        ]:
            out = delong_test(y, stk[a].to_numpy(), stk[b].to_numpy())
            out["Comparison"] = name
            rows.append(out)
        pd.DataFrame(rows).to_csv(
            args.output_dir / "DeLong_Stacking_vs_Single.csv",
            index=False,
            encoding="utf-8-sig",
        )

        stack_threshold = best_f1_threshold(y, stk["stack"].to_numpy())
        lgb_threshold = best_f1_threshold(y, stk["lgb"].to_numpy())
        mcnemar = mcnemar_p(
            y,
            stk["stack"].to_numpy(),
            stk["lgb"].to_numpy(),
            stack_threshold,
        )
        pd.DataFrame([{
            "Comparison": "Stacking vs LightGBM",
            "Common_threshold_model1": stack_threshold,
            "Model2_own_F1_threshold_reference": lgb_threshold,
            **mcnemar,
        }]).to_csv(
            args.output_dir / "McNemar_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # SMOTE sensitivity is compared against the same GroupKFold class-weighted source.
    if args.smote.exists():
        sm = pd.read_csv(args.smote, encoding="utf-8-sig")
        y_sm = sm["y_true"].to_numpy(dtype=int)
        sm_prob = sm[
            "y_prob_calibrated" if "y_prob_calibrated" in sm.columns else "y_prob_raw"
        ].to_numpy()
        if len(y_sm) == len(y_gk) and np.array_equal(y_sm, y_gk):
            class_prob = gk["y_prob_calibrated"].to_numpy()
            out = delong_test(y_gk, class_prob, sm_prob)
            out["Comparison"] = "Class-weighted LightGBM vs SMOTE"
            pd.DataFrame([out]).to_csv(
                args.output_dir / "DeLong_SMOTE_vs_classweight.csv",
                index=False,
                encoding="utf-8-sig",
            )

    print("Statistical analyses completed.")


if __name__ == "__main__":
    main()
