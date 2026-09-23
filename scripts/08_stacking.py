from __future__ import annotations
import argparse
from pathlib import Path

# Make direct execution from the repository root import the local src package.
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

from src.common import LGBM_OPTUNA_PARAMS, XGB_PARAMS, RF_PARAMS, numeric_feature_columns, normalize_stkcd, compute_metrics, best_f1_threshold

ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(description="GroupKFold stacking experiment.")
    p.add_argument("--input",type=Path,default=ROOT/"data/processed/model_dataset_full.csv")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/stacking")
    return p.parse_args()

def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,dtype={"Stkcd":str},encoding="utf-8-sig",low_memory=False)
    df["Stkcd"]=normalize_stkcd(df["Stkcd"])
    feats=numeric_feature_columns(df)
    X=df[feats].copy(); y=df["Fraud"].astype(int).to_numpy(); groups=df["Stkcd"].to_numpy()

    lgb_p=np.zeros(len(df)); xgb_p=np.zeros(len(df)); rf_p=np.zeros(len(df)); stack_p=np.zeros(len(df)); folds=np.zeros(len(df),dtype=int)
    gkf=GroupKFold(n_splits=5)
    for fold,(tr,te) in enumerate(gkf.split(X,y,groups),1):
        imp=SimpleImputer(strategy="median")
        Xtr=imp.fit_transform(X.iloc[tr]); Xte=imp.transform(X.iloc[te])
        m_lgb=lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS).fit(Xtr,y[tr])
        m_xgb=xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr,y[tr])
        m_rf=RandomForestClassifier(**RF_PARAMS).fit(Xtr,y[tr])
        p1=m_lgb.predict_proba(Xte)[:,1]; p2=m_xgb.predict_proba(Xte)[:,1]; p3=m_rf.predict_proba(Xte)[:,1]
        lgb_p[te],xgb_p[te],rf_p[te]=p1,p2,p3

        inner=GroupKFold(n_splits=3)
        meta=np.zeros((len(tr),3))
        for itr,iva in inner.split(Xtr,y[tr],groups[tr]):
            imp_i=SimpleImputer(strategy="median")
            Xi=imp_i.fit_transform(Xtr[itr]); Xv=imp_i.transform(Xtr[iva])
            a=lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS).fit(Xi,y[tr][itr])
            b=xgb.XGBClassifier(**XGB_PARAMS).fit(Xi,y[tr][itr])
            c=RandomForestClassifier(**RF_PARAMS).fit(Xi,y[tr][itr])
            meta[iva,0]=a.predict_proba(Xv)[:,1]
            meta[iva,1]=b.predict_proba(Xv)[:,1]
            meta[iva,2]=c.predict_proba(Xv)[:,1]
        meta_test=np.column_stack([p1,p2,p3])
        meta_model=LogisticRegression(C=1.0,max_iter=1000,class_weight="balanced")
        meta_model.fit(meta,y[tr])
        stack_p[te]=meta_model.predict_proba(meta_test)[:,1]
        folds[te]=fold

    rows=[]
    for name,p in [("LightGBM",lgb_p),("XGBoost",xgb_p),("Random Forest",rf_p),("Stacking",stack_p)]:
        thr=best_f1_threshold(y,p)
        rows.append({"Model":name,"AUC":roc_auc_score(y,p),**compute_metrics(y,p,thr)})
    pd.DataFrame(rows).to_csv(args.output_dir/"Stacking_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"y_true":y,"lgb":lgb_p,"xgb":xgb_p,"rf":rf_p,"stack":stack_p,"fold":folds}).to_csv(args.output_dir/"Stacking_OOF.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__":
    main()
