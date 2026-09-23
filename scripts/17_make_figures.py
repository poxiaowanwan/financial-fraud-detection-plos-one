from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import shap

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate manuscript Figures 2-8.")
    p.add_argument("--groupkfold-dir", type=Path, default=ROOT / "outputs" / "groupkfold")
    p.add_argument("--timesplit-dir", type=Path, default=ROOT / "outputs" / "timesplit")
    p.add_argument("--explainability-dir", type=Path, default=ROOT / "outputs" / "explainability")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    return p.parse_args()


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 2: ROC curves, Base vs Full.
    pred = args.groupkfold_dir / "Base_vs_Full_OOF_predictions.csv"
    if pred.exists():
        df = pd.read_csv(pred)
        y = df["y_true"].to_numpy()
        model_map = {
            "Lasso-LR": ("Base_Lasso-LR", "Full_Lasso-LR"),
            "Random Forest": ("Base_RandomForest", "Full_RandomForest"),
            "XGBoost": ("Base_XGBoost", "Full_XGBoost"),
            "LightGBM": ("Base_LightGBM", "Full_LightGBM"),
        }
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        for label, (bc, fc) in model_map.items():
            if bc in df.columns:
                fpr, tpr, _ = roc_curve(y, df[bc])
                ax[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, df[bc]):.4f})")
            if fc in df.columns:
                fpr, tpr, _ = roc_curve(y, df[fc])
                ax[1].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, df[fc]):.4f})")
        for a, title in zip(
            ax,
            ["Base feature set (58 raw features)", "Full feature set (84 raw features)"],
        ):
            a.plot([0, 1], [0, 1], "k--", lw=1)
            a.set_xlabel("False Positive Rate")
            a.set_ylabel("True Positive Rate")
            a.set_title(title)
            a.legend(fontsize=8)
        save(fig, args.output_dir / "fig2_roc.png")

        # Figure 3: PR curves.
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        baseline = y.mean()
        for label, (bc, fc) in model_map.items():
            if bc in df.columns:
                p = df[bc].to_numpy()
                precision, recall, _ = precision_recall_curve(y, p)
                ax[0].plot(
                    recall,
                    precision,
                    label=f"{label} (PR-AUC={average_precision_score(y, p):.4f})",
                )
            if fc in df.columns:
                p = df[fc].to_numpy()
                precision, recall, _ = precision_recall_curve(y, p)
                ax[1].plot(
                    recall,
                    precision,
                    label=f"{label} (PR-AUC={average_precision_score(y, p):.4f})",
                )
        for a, title in zip(
            ax,
            ["Base feature set (58 raw features)", "Full feature set (84 raw features)"],
        ):
            a.axhline(baseline, ls="--", lw=1, label=f"Baseline ({baseline:.4f})")
            a.set_xlabel("Recall")
            a.set_ylabel("Precision")
            a.set_title(title)
            a.legend(fontsize=8)
        save(fig, args.output_dir / "fig3_pr.png")

    # Figure 4: TimeSplit calibration.
    ts = args.timesplit_dir / "TimeSplit_test_predictions.csv"
    if ts.exists():
        data = pd.read_csv(ts)
        y = data["y_true"].to_numpy()
        fig, ax = plt.subplots(figsize=(7, 6))
        for label, col in [("Raw", "prob_raw"), ("Calibrated", "prob_cal")]:
            if col not in data.columns:
                continue
            bins = np.linspace(0, 1, 11)
            pred_mean, obs = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (data[col] >= lo) & (data[col] < hi if hi < 1 else data[col] <= hi)
                if mask.any():
                    pred_mean.append(data.loc[mask, col].mean())
                    obs.append(y[mask].mean())
            ax.plot(pred_mean, obs, "o-", label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed fraud rate")
        ax.set_title("Calibration curve (Optuna LightGBM, TimeSplit test)")
        ax.legend()
        save(fig, args.output_dir / "fig4_calibration.png")

    # Figures 5-8: SHAP artifacts.
    ev = args.explainability_dir
    shap_path = ev / "SHAP_values.npy"
    x_path = ev / "X_test.npy"
    feat_path = ev / "SHAP_feature_values.csv"
    stable_path = ev / "SHAP_stability_5fold.csv"
    imp_path = ev / "SHAP_global_importance.csv"
    if shap_path.exists() and x_path.exists() and feat_path.exists():
        sv = np.load(shap_path)
        Xdf = pd.read_csv(feat_path)
        feature_names = Xdf.columns.tolist()

        if imp_path.exists():
            imp = pd.read_csv(imp_path).head(20).sort_values("MeanAbsSHAP")
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(imp["feature"], imp["MeanAbsSHAP"])
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Global SHAP importance")
            save(fig, args.output_dir / "fig5_shap_bar.png")

        plt.figure(figsize=(9, 8))
        shap.summary_plot(sv, Xdf, feature_names=feature_names, show=False, max_display=20)
        plt.title("SHAP beeswarm plot (Top 20)")
        plt.tight_layout()
        plt.savefig(args.output_dir / "fig6_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.savefig(args.output_dir / "fig6_shap_beeswarm.pdf", bbox_inches="tight")
        plt.close()

        if "TextualSimilarity" in feature_names:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                "TextualSimilarity",
                sv,
                Xdf,
                feature_names=feature_names,
                show=False,
                interaction_index=None,
            )
            plt.title("SHAP dependence: TextualSimilarity")
            plt.tight_layout()
            plt.savefig(args.output_dir / "fig7_shap_dependence.png", dpi=300, bbox_inches="tight")
            plt.savefig(args.output_dir / "fig7_shap_dependence.pdf", bbox_inches="tight")
            plt.close()

        if stable_path.exists():
            stability = pd.read_csv(stable_path)
            counts = stability.groupby("feature")["rank"].count().sort_values(ascending=False).head(15)
            stable_top = counts.index.tolist()[::-1]
            fig, ax = plt.subplots(figsize=(8, 7))
            subset = stability[stability["feature"].isin(stable_top)]
            means = subset.groupby("feature")["mean_abs_shap"].mean().sort_values()
            ax.barh(means.index, means.values)
            ax.set_xlabel("Mean |SHAP| across folds")
            ax.set_title("SHAP feature stability (Top 15)")
            save(fig, args.output_dir / "fig8_shap_stability.png")


if __name__ == "__main__":
    main()
