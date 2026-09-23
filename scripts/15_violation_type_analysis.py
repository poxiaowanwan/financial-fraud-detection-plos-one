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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve

from src.common import EXCLUDE_COLS, LGBM_OPTUNA_PARAMS, normalize_stkcd

ROOT = Path(__file__).resolve().parents[1]

TARGET_TYPES = ["P2501", "P2502", "P2503", "P2506"]
TYPE_NAMES = {
    "P2501": "Fabricated profits",
    "P2502": "Fictitious asset reporting",
    "P2503": "False or misleading disclosures",
    "P2506": "Inaccurate disclosures (other)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replicate violation-type and fraud-history stratification."
    )
    p.add_argument(
        "--input", type=Path,
        default=ROOT / "data" / "processed" / "model_dataset_full.csv",
    )
    p.add_argument(
        "--violation", type=Path,
        default=ROOT / "data" / "processed" / "target_four_fraud_types.csv",
    )
    p.add_argument(
        "--labels", type=Path,
        default=ROOT / "data" / "processed" / "fraud_labels_firm_year.csv",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "violation_type",
    )
    return p.parse_args()


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, threshold = precision_recall_curve(y_true, prob)
    if len(threshold) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(threshold[np.argmax(f1)])


def build_groupkfold_predictions(
    data: pd.DataFrame,
    features: list[str],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Outer 5-fold GroupKFold with fold-specific inner 3-fold calibration."""
    X_all = data[features].to_numpy()
    y = data["Fraud"].astype(int).to_numpy()
    groups = normalize_stkcd(data["Stkcd"]).to_numpy()

    outer = GroupKFold(n_splits=n_splits)
    raw_oof = np.zeros(len(data))
    cal_oof = np.zeros(len(data))

    for fold, (tr, te) in enumerate(outer.split(X_all, y, groups), 1):
        outer_imputer = SimpleImputer(strategy="median")
        X_tr = outer_imputer.fit_transform(X_all[tr])
        X_te = outer_imputer.transform(X_all[te])

        model = lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS)
        model.fit(X_tr, y[tr])
        p = model.predict_proba(X_te)[:, 1]
        raw_oof[te] = p

        inner = GroupKFold(n_splits=3)
        inner_oof = np.zeros(len(tr))
        for i_tr, i_val in inner.split(X_tr, y[tr], groups[tr]):
            imputer_i = SimpleImputer(strategy="median")
            Xi = imputer_i.fit_transform(X_tr[i_tr])
            Xv = imputer_i.transform(X_tr[i_val])
            inner_model = lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS)
            inner_model.fit(Xi, y[tr][i_tr])
            inner_oof[i_val] = inner_model.predict_proba(Xv)[:, 1]

        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            inner_oof, y[tr]
        )
        cal_oof[te] = calibrator.predict(p)
        print(f"Fold {fold}/{n_splits} completed.")

    return raw_oof, cal_oof


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        args.input, dtype={"Stkcd": str}, encoding="utf-8-sig", low_memory=False
    )
    data["Stkcd"] = normalize_stkcd(data["Stkcd"])
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype(int)

    features = [
        c for c in data.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(data[c])
    ]
    if len(features) != 83:
        raise ValueError(
            f"Expected 83 numeric features for the reported workflow, found {len(features)}."
        )

    y = data["Fraud"].astype(int).to_numpy()
    _, oof_cal = build_groupkfold_predictions(data, features)
    threshold = best_f1_threshold(y, oof_cal)

    viol = pd.read_csv(args.violation, encoding="utf-8-sig", low_memory=False)
    viol["Stkcd"] = normalize_stkcd(viol["Stkcd"])
    viol["year"] = pd.to_numeric(viol["year"], errors="coerce")
    viol = viol.dropna(subset=["Stkcd", "year"]).copy()
    viol["year"] = viol["year"].astype(int)
    if "ViolationTypeID" not in viol.columns:
        raise ValueError("Violation data must contain ViolationTypeID.")
    type_map = viol.groupby(["Stkcd", "year"])["ViolationTypeID"].apply(set).to_dict()

    keys = list(zip(data["Stkcd"], data["year"]))
    data["vtypes"] = [type_map.get(key, set()) for key in keys]
    pred_flag = oof_cal >= threshold

    rows = []
    for vtype in TARGET_TYPES:
        has_type = data["vtypes"].apply(
            lambda s: isinstance(s, set) and vtype in s
        )
        fraud_mask = has_type & (data["Fraud"] == 1)
        nonfraud_mask = data["Fraud"] == 0
        if fraud_mask.sum() == 0:
            continue
        fraud_idx = fraud_mask.to_numpy()
        nonfraud_idx = nonfraud_mask.to_numpy()
        rows.append({
            "ViolationType": vtype,
            "TypeName": TYPE_NAMES[vtype],
            "N_FraudFirmYears": int(fraud_mask.sum()),
            "Recall_on_this_type": float(pred_flag[fraud_idx].mean()),
            "Mean_Prob_fraud": float(oof_cal[fraud_idx].mean()),
            "Mean_Prob_nonfraud": float(oof_cal[nonfraud_idx].mean()),
            "Threshold": threshold,
        })

    pd.DataFrame(rows).to_csv(
        args.output_dir / "ViolationType_stratification.csv",
        index=False, encoding="utf-8-sig"
    )

    # Single-type vs multi-type fraud.
    data["n_vtypes"] = data["vtypes"].apply(
        lambda s: len(s) if isinstance(s, set) else 0
    )
    history_rows = []
    for n in [1, 2, 3]:
        mask = (data["Fraud"] == 1) & (data["n_vtypes"] == n)
        if mask.sum():
            idx = mask.to_numpy()
            history_rows.append({
                "Category": f"{n}_types",
                "N": int(mask.sum()),
                "Recall": float(pred_flag[idx].mean()),
                "Mean_Prob": float(oof_cal[idx].mean()),
                "Threshold": threshold,
            })

    labels = pd.read_csv(args.labels, encoding="utf-8-sig", low_memory=False)
    labels["Stkcd"] = normalize_stkcd(labels["Stkcd"])
    labels["year"] = pd.to_numeric(labels["year"], errors="coerce")
    labels = labels.dropna(subset=["Stkcd", "year"]).copy()
    labels["year"] = labels["year"].astype(int)
    needed = {"first_fraud", "repeat_fraud"}
    if needed.issubset(labels.columns):
        label_map = labels.set_index(["Stkcd", "year"])[
            ["first_fraud", "repeat_fraud"]
        ].to_dict("index")
        data["first_fraud"] = [
            label_map.get(key, {}).get("first_fraud", 0) for key in keys
        ]
        data["repeat_fraud"] = [
            label_map.get(key, {}).get("repeat_fraud", 0) for key in keys
        ]
        for label, col in [("First-time", "first_fraud"), ("Repeat", "repeat_fraud")]:
            mask = (data["Fraud"] == 1) & (data[col] == 1)
            if mask.sum():
                idx = mask.to_numpy()
                history_rows.append({
                    "Category": label,
                    "N": int(mask.sum()),
                    "Recall": float(pred_flag[idx].mean()),
                    "Mean_Prob": float(oof_cal[idx].mean()),
                    "Threshold": threshold,
                })

    pd.DataFrame(history_rows).to_csv(
        args.output_dir / "Fraud_history_stratification.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"Global OOF F1-optimal threshold: {threshold:.4f}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(history_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
