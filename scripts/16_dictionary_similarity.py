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
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from src.common import (
    EXCLUDE_COLS, LGBM_OPTUNA_PARAMS, MDA_FEATURES,
    best_f1_threshold, delong_test, normalize_stkcd,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_DICT = ["PositiveVocabularyNum", "NegativeVocabularyNum", "EmotionTone1", "EmotionTone2"]
DLUT_DICT = [
    "DLUT_PosNum", "DLUT_NegNum", "DLUT_PosRatio", "DLUT_NegRatio",
    "DLUT_PosIntensity", "DLUT_NegIntensity", "DLUT_EmotionScore",
    "DLUT_EmotionTone", "DLUT_NegAfterNeg", "DLUT_PosAfterNeg",
]
SIM_FEATS = ["TextualSimilarity", "Jaccard_prev", "EditSim_prev", "TFIDF_Cosine_prev"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dictionary comparison and similarity-feature ablation.")
    p.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "model_dataset_full.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dictionary_similarity")
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, prob),
        "PR_AUC": average_precision_score(y, prob),
        "Brier": brier_score_loss(y, prob),
        "F1": f1_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Type_I_Error": fp / (fp + tn) if fp + tn else np.nan,
        "Type_II_Error": fn / (fn + tp) if fn + tp else np.nan,
        "Threshold": threshold,
    }


def calibrated_groupkfold_oof(df: pd.DataFrame, feats: list[str], n_splits: int) -> np.ndarray:
    X = df[feats].to_numpy()
    y = df["Fraud"].astype(int).to_numpy()
    groups = normalize_stkcd(df["Stkcd"]).to_numpy()
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        model = lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    return iso.predict(oof)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, dtype={"Stkcd": str}, encoding="utf-8-sig", low_memory=False)
    df["Stkcd"] = normalize_stkcd(df["Stkcd"])
    y = df["Fraud"].astype(int).to_numpy()
    all_feats = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if len(all_feats) != 83:
        raise ValueError(f"Expected 83 numeric features, found {len(all_feats)}.")
    non_mda = [c for c in all_feats if c not in MDA_FEATURES]
    feature_sets = {
        "NonMDA_only": non_mda,
        "+BaseDict": non_mda + [c for c in BASE_DICT if c in all_feats],
        "+DLUTDict": non_mda + [c for c in DLUT_DICT if c in all_feats],
        "+BothDicts": non_mda + [c for c in BASE_DICT + DLUT_DICT if c in all_feats],
        "+Full_MDA(83)": all_feats,
    }
    probs = {}
    rows = []
    for name, feats in feature_sets.items():
        prob = calibrated_groupkfold_oof(df, feats, args.n_splits)
        threshold = best_f1_threshold(y, prob)
        row = metrics(y, prob, threshold)
        row["FeatureSet"] = name
        row["N_Features"] = len(feats)
        rows.append(row)
        probs[name] = prob
        print(f"{name}: n={len(feats)}, AUC={row['AUC']:.4f}, PR-AUC={row['PR_AUC']:.4f}, Brier={row['Brier']:.4f}")
    pd.DataFrame(rows).to_csv(args.output_dir / "Dictionary_comparison.csv", index=False, encoding="utf-8-sig")

    d_rows = []
    for label, a, b in [("BaseDict vs DLUTDict", "+BaseDict", "+DLUTDict"), ("BothDicts vs NonMDA_only", "+BothDicts", "NonMDA_only")]:
        out = delong_test(y, probs[a], probs[b])
        out["Comparison"] = label
        d_rows.append(out)
    pd.DataFrame(d_rows).to_csv(args.output_dir / "Dictionary_DeLong.csv", index=False, encoding="utf-8-sig")

    # Leave-one-out ablation for the four similarity measures.
    full_prob = probs["+Full_MDA(83)"]
    base_row = metrics(y, full_prob, best_f1_threshold(y, full_prob))
    base_row.update({"Config": "Full(83)", "N_Features": len(all_feats)})
    abl_rows = [base_row]
    for drop in SIM_FEATS:
        feats = [c for c in all_feats if c != drop]
        prob = calibrated_groupkfold_oof(df, feats, args.n_splits)
        row = metrics(y, prob, best_f1_threshold(y, prob))
        row["Config"] = f"-{drop}"
        row["N_Features"] = len(feats)
        out = delong_test(y, full_prob, prob)
        row["AUC_Diff_vs_Full"] = out["AUC_1"] - out["AUC_2"]
        row["DeLong_p_vs_Full"] = out["p_value"]
        abl_rows.append(row)
    pd.DataFrame(abl_rows).to_csv(args.output_dir / "Similarity_ablation.csv", index=False, encoding="utf-8-sig")
    print("Dictionary and similarity analyses completed.")


if __name__ == "__main__":
    main()
