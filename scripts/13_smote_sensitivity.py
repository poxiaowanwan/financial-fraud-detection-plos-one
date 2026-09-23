from __future__ import annotations
import argparse
from pathlib import Path

# Make direct execution from the repository root import the local src package.
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))
import numpy as np, pandas as pd
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.common import numeric_feature_columns, LGBM_OPTUNA_PARAMS, normalize_stkcd, delong_test

ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(description="SMOTE vs class-weighted LightGBM sensitivity analysis.")
    p.add_argument("--input",type=Path,default=ROOT/"data/processed/model_dataset_full.csv")
    p.add_argument("--class-weight-predictions",type=Path,default=ROOT/"outputs/groupkfold/Optuna_calibrated_predictions.csv")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/smote")
    return p.parse_args()

def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,dtype={"Stkcd":str},encoding="utf-8-sig",low_memory=False)
    df["Stkcd"]=normalize_stkcd(df["Stkcd"])
    feats=numeric_feature_columns(df); X=df[feats]; y=df.Fraud.astype(int).to_numpy(); groups=df.Stkcd.to_numpy()
    gkf=GroupKFold(n_splits=5); oof=np.zeros(len(df)); folds=np.zeros(len(df),dtype=int)
    for fold,(tr,te) in enumerate(gkf.split(X,y,groups),1):
        imp=SimpleImputer(strategy="median")
        Xtr=imp.fit_transform(X.iloc[tr]); Xte=imp.transform(X.iloc[te])
        sm=SMOTE(random_state=42)
        Xres,yres=sm.fit_resample(Xtr,y[tr])
        m=lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS).fit(Xres,yres)
        oof[te]=m.predict_proba(Xte)[:,1]; folds[te]=fold
    iso=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
    cal=iso.predict(oof)
    out=pd.DataFrame({"y_true":y,"y_prob_raw":oof,"y_prob_calibrated":cal,"fold":folds})
    out.to_csv(args.output_dir/"SMOTE_OOF_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{
        "AUC_raw":roc_auc_score(y,oof),
        "AUC_calibrated":roc_auc_score(y,cal),
        "PR_AUC_calibrated":average_precision_score(y,cal)
    }]).to_csv(args.output_dir/"SMOTE_summary.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__":
    main()
