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
import lightgbm as lgb
import optuna
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold

from src.common import EXCLUDE_COLS, numeric_feature_columns, compute_metrics

ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(description="TimeSplit LightGBM + Optuna + isotonic calibration.")
    p.add_argument("--input",type=Path,default=ROOT/"data/processed/model_dataset_full.csv")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/timesplit")
    p.add_argument("--trials",type=int,default=30)
    return p.parse_args()

def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.input,dtype={"Stkcd":str},encoding="utf-8-sig",low_memory=False)
    feats=numeric_feature_columns(data)
    train=data[data["year"].between(2015,2021)].copy()
    val=data[data["year"]==2022].copy()
    test=data[data["year"].between(2023,2024)].copy()

    X_tr=train[feats].copy(); y_tr=train["Fraud"].astype(int).to_numpy(); g_tr=train["Stkcd"].astype(str).to_numpy()
    X_va=val[feats].copy(); y_va=val["Fraud"].astype(int).to_numpy()
    X_te=test[feats].copy(); y_te=test["Fraud"].astype(int).to_numpy()

    imputer=SimpleImputer(strategy="median")
    X_tr=imputer.fit_transform(X_tr); X_va=imputer.transform(X_va); X_te=imputer.transform(X_te)

    def objective(trial):
        params=dict(
            n_estimators=trial.suggest_int("n_estimators",200,600,step=100),
            learning_rate=trial.suggest_float("learning_rate",0.01,0.05,log=True),
            num_leaves=trial.suggest_int("num_leaves",15,50),
            min_child_samples=trial.suggest_int("min_child_samples",10,40),
            subsample=trial.suggest_float("subsample",0.6,1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
            reg_alpha=trial.suggest_float("reg_alpha",0.0,0.5),
            reg_lambda=trial.suggest_float("reg_lambda",0.0,0.5),
            class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1,
        )
        inner=GroupKFold(n_splits=3)
        scores=[]
        for tr_idx,va_idx in inner.split(X_tr,y_tr,g_tr):
            model=lgb.LGBMClassifier(**params)
            model.fit(X_tr[tr_idx],y_tr[tr_idx])
            p=model.predict_proba(X_tr[va_idx])[:,1]
            scores.append(average_precision_score(y_tr[va_idx],p))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    t0=time.time(); study.optimize(objective,n_trials=args.trials,show_progress_bar=False)
    best={**study.best_params,"class_weight":"balanced","random_state":42,"n_jobs":-1,"verbose":-1}
    model=lgb.LGBMClassifier(**best).fit(X_tr,y_tr)

    p_va_raw=model.predict_proba(X_va)[:,1]
    p_te_raw=model.predict_proba(X_te)[:,1]

    iso=IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_va_raw,y_va)
    p_va_cal=iso.predict(p_va_raw)
    p_te_cal=iso.predict(p_te_raw)

    prec,rec,thr=precision_recall_curve(y_va,p_va_cal)
    f1=2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    threshold=float(thr[np.argmax(f1)]) if len(thr) else 0.5

    rows=[]
    rows.append({"Set":"Train","N":len(y_tr),**compute_metrics(y_tr,model.predict_proba(X_tr)[:,1],0.5)})
    rows.append({"Set":"Val (raw)","N":len(y_va),**compute_metrics(y_va,p_va_raw,0.5)})
    rows.append({"Set":"Val (calibrated)","N":len(y_va),**compute_metrics(y_va,p_va_cal,threshold)})
    rows.append({"Set":"Test (raw)","N":len(y_te),**compute_metrics(y_te,p_te_raw,0.5)})
    rows.append({"Set":"Test (calibrated)","N":len(y_te),**compute_metrics(y_te,p_te_cal,threshold)})

    pd.DataFrame(rows).to_csv(args.output_dir/"TimeSplit_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"y_true":y_va,"prob_raw":p_va_raw,"prob_cal":p_va_cal}).to_csv(args.output_dir/"TimeSplit_val_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"y_true":y_te,"prob_raw":p_te_raw,"prob_cal":p_te_cal}).to_csv(args.output_dir/"TimeSplit_test_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([best]).to_csv(args.output_dir/"TimeSplit_best_params.csv",index=False,encoding="utf-8-sig")
    print(f"Best inner PR-AUC: {study.best_value:.4f}")
    print("Best parameters:", best)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"Optuna elapsed: {(time.time()-t0)/60:.2f} min")

if __name__=="__main__":
    main()
