from __future__ import annotations
import argparse
from pathlib import Path

# Make direct execution from the repository root import the local src package.
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, recall_score, precision_score, precision_recall_curve,
    confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.common import (
    MDA_FEATURES, EXCLUDE_COLS, LGBM_DEFAULT_PARAMS, LGBM_OPTUNA_PARAMS,
    XGB_PARAMS, RF_PARAMS, LASSO_PARAMS, best_f1_threshold,
    compute_metrics, normalize_stkcd,
)

ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(description="GroupKFold model comparison: Base vs Full.")
    p.add_argument("--input",type=Path,default=ROOT/"data/processed/model_dataset_full.csv")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/groupkfold")
    p.add_argument("--n-splits",type=int,default=5)
    return p.parse_args()

def make_model(name):
    if name=="Lasso-LR":
        return Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("model",LogisticRegression(**LASSO_PARAMS)),
        ])
    if name=="Random Forest":
        return Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",RandomForestClassifier(**RF_PARAMS)),
        ])
    if name=="LightGBM":
        return Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",lgb.LGBMClassifier(**LGBM_DEFAULT_PARAMS)),
        ])
    if name=="XGBoost":
        return Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",xgb.XGBClassifier(**XGB_PARAMS)),
        ])
    raise ValueError(name)

def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,encoding="utf-8-sig",low_memory=False)
    df["Stkcd"]=normalize_stkcd(df["Stkcd"])
    y=df["Fraud"].astype(int).to_numpy(); groups=df["Stkcd"].to_numpy()
    all_feats=[c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    mda=[c for c in MDA_FEATURES if c in all_feats]
    base=[c for c in all_feats if c not in mda]
    feature_sets={"Base":base,"Full":all_feats}
    gkf=GroupKFold(n_splits=args.n_splits)

    results=[]; pred_store={"y_true":y}; t0=time.time()
    for fs_name, feats in feature_sets.items():
        X=df[feats]
        for model_name in ["Lasso-LR","Random Forest","XGBoost","LightGBM"]:
            oof=np.zeros(len(df))
            fold_rows=[]
            for fold,(tr,te) in enumerate(gkf.split(X,y,groups),1):
                # Tune the classification threshold on inner GroupKFold OOF
                # predictions only; the outer test fold is never used for tuning.
                inner=GroupKFold(n_splits=3)
                inner_prob=np.zeros(len(tr))
                for itr,iva in inner.split(X.iloc[tr], y[tr], groups[tr]):
                    inner_model=make_model(model_name)
                    inner_model.fit(X.iloc[tr].iloc[itr], y[tr][itr])
                    inner_prob[iva]=inner_model.predict_proba(X.iloc[tr].iloc[iva])[:,1]
                thr=best_f1_threshold(y[tr],inner_prob)

                model=make_model(model_name)
                model.fit(X.iloc[tr],y[tr])
                prob=model.predict_proba(X.iloc[te])[:,1]
                oof[te]=prob
                met=compute_metrics(y[te],prob,thr)
                met["fold"]=fold
                fold_rows.append(met)
            fold_df=pd.DataFrame(fold_rows)
            row={"FeatureSet":fs_name,"Model":model_name,"N_Features":len(feats)}
            for metric in ["AUC","PR_AUC","Brier","F1","Recall","Precision","Specificity"]:
                row[metric]=fold_df[metric].mean()
                row[f"{metric}_std"]=fold_df[metric].std()
            results.append(row)
            model_key = {
                "Lasso-LR": "Lasso-LR",
                "Random Forest": "RandomForest",
                "XGBoost": "XGBoost",
                "LightGBM": "LightGBM",
            }[model_name]
            pred_store[f"{fs_name}_{model_key}"]=oof


    # Primary GroupKFold LightGBM with the Optuna-selected parameter set.
    # Calibration is fitted from the complete OOF predictions, matching the
    # analysis workflow used for the reported GroupKFold main result.
    full_feats = all_feats
    X = df[full_feats]
    raw_oof = np.zeros(len(df))
    for tr, te in gkf.split(X, y, groups):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS)),
        ])
        model.fit(X.iloc[tr], y[tr])
        raw_oof[te] = model.predict_proba(X.iloc[te])[:,1]

    iso = __import__("sklearn.isotonic", fromlist=["IsotonicRegression"]).IsotonicRegression(
        out_of_bounds="clip"
    ).fit(raw_oof, y)
    cal_oof = iso.predict(raw_oof)
    pd.DataFrame({
        "y_true": y,
        "y_prob_raw": raw_oof,
        "y_prob_calibrated": cal_oof,
    }).to_csv(args.output_dir/"Optuna_calibrated_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "AUC_raw": roc_auc_score(y, raw_oof),
        "AUC_calibrated": roc_auc_score(y, cal_oof),
        "PR_AUC_calibrated": average_precision_score(y, cal_oof),
        "Brier_raw": brier_score_loss(y, raw_oof),
        "Brier_calibrated": brier_score_loss(y, cal_oof),
        "Threshold": best_f1_threshold(y, cal_oof),
    }]).to_csv(args.output_dir/"Optuna_GroupKFold_metrics.csv", index=False, encoding="utf-8-sig")

    result_df=pd.DataFrame(results)
    result_df.to_csv(args.output_dir/"Base_vs_Full_results.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(pred_store).to_csv(args.output_dir/"Base_vs_Full_OOF_predictions.csv",index=False,encoding="utf-8-sig")
    print(result_df.round(4).to_string(index=False))
    print(f"Elapsed: {(time.time()-t0)/60:.2f} min")

if __name__=="__main__":
    main()
