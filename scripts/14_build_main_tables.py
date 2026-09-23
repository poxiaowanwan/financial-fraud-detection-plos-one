from __future__ import annotations
import argparse
from pathlib import Path

# Make direct execution from the repository root import the local src package.
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score

from src.common import best_f1_threshold

ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(description="Assemble the main Table 3 from saved predictions.")
    p.add_argument("--groupkfold",type=Path,default=ROOT/"outputs/groupkfold/Base_vs_Full_OOF_predictions.csv")
    p.add_argument("--timesplit",type=Path,default=ROOT/"outputs/timesplit/TimeSplit_test_predictions.csv")
    p.add_argument("--stacking",type=Path,default=ROOT/"outputs/stacking/Stacking_OOF.csv")
    p.add_argument("--optuna-gk",type=Path,default=ROOT/"outputs/groupkfold/Optuna_calibrated_predictions.csv")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/tables/Table3_MainResults.csv")
    return p.parse_args()

def row(y,p,name,evaluation):
    thr=best_f1_threshold(y,p); pred=(p>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "Model":name,"Evaluation":evaluation,"AUC":roc_auc_score(y,p),
        "PR_AUC":average_precision_score(y,p),"Brier":brier_score_loss(y,p),
        "Recall":tp/(tp+fn) if tp+fn else 0,
        "Precision":tp/(tp+fp) if tp+fp else 0,
        "Type_I_Error":fp/(fp+tn) if fp+tn else 0,
        "Type_II_Error":fn/(fn+tp) if fn+tp else 0,
        "Threshold":thr,"N":len(y),"Fraud_rate":y.mean(),
    }

def main():
    args=parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    gk=pd.read_csv(args.groupkfold); y=gk.y_true.to_numpy()
    for col,label in [
        ("Full_Lasso-LR","Lasso-LR"),("Full_RandomForest","Random Forest"),
        ("Full_XGBoost","XGBoost"),("Full_LightGBM","LightGBM (default)")
    ]:
        if col in gk: rows.append(row(y,gk[col].to_numpy(),label,"GroupKFold"))
    ts_path=args.timesplit
    opt=args.optuna_gk
    if opt.exists():
        od=pd.read_csv(opt); yg=od.y_true.to_numpy()
        p_raw=od["y_prob_raw"].to_numpy(); p_cal=od["y_prob_calibrated"].to_numpy()
        rcal=row(yg,p_cal,"LightGBM + Optuna + Cal","GroupKFold")
        rcal["AUC_raw"]=roc_auc_score(yg,p_raw); rcal["Brier_raw"]=brier_score_loss(yg,p_raw)
        rows.append(rcal)
    if ts_path.exists():
        ts=pd.read_csv(ts_path); yts=ts.y_true.to_numpy()
        if "prob_cal" in ts:
            rts=row(yts,ts.prob_cal.to_numpy(),"LightGBM + Optuna + Cal","TimeSplit")
            if "prob_raw" in ts:
                rts["AUC_raw"]=roc_auc_score(yts,ts.prob_raw.to_numpy())
                rts["Brier_raw"]=brier_score_loss(yts,ts.prob_raw.to_numpy())
            rows.append(rts)
    if args.stacking.exists():
        stk=pd.read_csv(args.stacking)
        ys=stk.y_true.to_numpy()
        rows.append(row(ys,stk.stack.to_numpy(),"Stacking","GroupKFold"))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False,encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

if __name__=="__main__":
    main()
