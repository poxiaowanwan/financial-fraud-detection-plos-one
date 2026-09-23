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
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import shap
from src.common import numeric_feature_columns, LGBM_OPTUNA_PARAMS, EXCLUDE_COLS

ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(description="SHAP + permutation importance analysis.")
    p.add_argument("--input",type=Path,default=ROOT/"data/processed/model_dataset_full.csv")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/explainability")
    return p.parse_args()

def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,encoding="utf-8-sig",low_memory=False)
    train=df[df["year"]<=2021].copy(); test=df[df["year"]>=2023].copy()
    feature_cols=[c for c in df.columns if c not in EXCLUDE_COLS]
    obj=[c for c in feature_cols if df[c].dtype=="object"]
    if obj:
        merged=pd.get_dummies(df,columns=obj,prefix=obj,dummy_na=False)
        feature_cols=[c for c in merged.columns if c not in EXCLUDE_COLS]
    else:
        merged=df.copy()
    feature_cols=[c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]
    train=merged[merged["year"]<=2021].copy(); test=merged[merged["year"]>=2023].copy()
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(train[feature_cols]); Xte=imp.transform(test[feature_cols])
    ytr=train["Fraud"].astype(int).to_numpy(); yte=test["Fraud"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS).fit(Xtr,ytr)
    p=model.predict_proba(Xte)[:,1]
    print("Test AUC:",roc_auc_score(yte,p))

    perm=permutation_importance(model,Xte,yte,scoring="roc_auc",n_repeats=10,random_state=42,n_jobs=-1)
    perm_df=pd.DataFrame({
        "feature":feature_cols,
        "perm_importance_mean":perm.importances_mean,
        "perm_importance_std":perm.importances_std,
    }).sort_values("perm_importance_mean",ascending=False)
    perm_df["perm_rank"]=np.arange(1,len(perm_df)+1)
    perm_df.to_csv(args.output_dir/"Permutation_Importance.csv",index=False,encoding="utf-8-sig")

    explainer=shap.TreeExplainer(model)
    shap_values=explainer.shap_values(Xte)
    if isinstance(shap_values,list): shap_values=shap_values[1]
    mean_abs=np.abs(shap_values).mean(axis=0)
    shap_all=pd.DataFrame({"feature":feature_cols,"MeanAbsSHAP":mean_abs})
    shap_all=shap_all.sort_values("MeanAbsSHAP",ascending=False).reset_index(drop=True)
    shap_all["rank"]=np.arange(1,len(shap_all)+1)

    # Reported ranking excludes six audit-opinion one-hot dummy columns.
    numeric_original=[c for c in feature_cols if not c.startswith("TypeAuditOpin_")]
    report=shap_all[shap_all["feature"].isin(numeric_original)].copy()
    report["reported_rank"]=np.arange(1,len(report)+1)
    report.to_csv(args.output_dir/"SHAP_global_importance.csv",index=False,encoding="utf-8-sig")

    merged_rank=perm_df.merge(shap_all[["feature","MeanAbsSHAP","rank"]],on="feature",how="inner")
    rho,p=spearmanr(merged_rank["perm_importance_mean"],merged_rank["MeanAbsSHAP"])
    merged_rank["spearman_rho"]=rho; merged_rank["spearman_p"]=p
    merged_rank.to_csv(args.output_dir/"Permutation_vs_SHAP.csv",index=False,encoding="utf-8-sig")

    np.save(args.output_dir/"SHAP_values.npy",shap_values)
    np.save(args.output_dir/"X_test.npy",Xte)
    pd.DataFrame(Xte, columns=feature_cols).to_csv(
        args.output_dir/"SHAP_feature_values.csv", index=False, encoding="utf-8-sig"
    )

    # Five-fold SHAP stability analysis on the full sample with firm-isolated folds.
    from sklearn.model_selection import GroupKFold
    groups_all = merged["Stkcd"].to_numpy()
    y_all = merged["Fraud"].astype(int).to_numpy()
    X_all_df = merged[feature_cols].copy()
    stable = []
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all_df, y_all, groups_all), 1):
        imp_fold = SimpleImputer(strategy="median")
        Xtr_fold = imp_fold.fit_transform(X_all_df.iloc[tr_idx])
        Xte_fold = imp_fold.transform(X_all_df.iloc[te_idx])
        model_fold = lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS).fit(Xtr_fold, y_all[tr_idx])
        sv = shap.TreeExplainer(model_fold).shap_values(Xte_fold)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)[:15]
        for rank, idxf in enumerate(order, 1):
            stable.append({
                "fold": fold,
                "rank": rank,
                "feature": feature_cols[idxf],
                "mean_abs_shap": mean_abs[idxf],
            })
    pd.DataFrame(stable).to_csv(
        args.output_dir/"SHAP_stability_5fold.csv", index=False, encoding="utf-8-sig"
    )

if __name__=="__main__":
    main()
