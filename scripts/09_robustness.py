from __future__ import annotations

import argparse
from pathlib import Path

# Make direct execution from the repository root import the local src package.
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.common import (
    EXCLUDE_COLS,
    LGBM_OPTUNA_PARAMS,
    MDA_FEATURES,
    normalize_stkcd,
    numeric_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robustness and sensitivity analyses for the final LightGBM model."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "model_dataset_full.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "robustness",
    )
    p.add_argument(
        "--industry",
        type=Path,
        default=ROOT / "data" / "raw" / "industry" / "industry_data.xlsx",
    )
    p.add_argument(
        "--violation",
        type=Path,
        default=ROOT / "data" / "processed" / "target_four_fraud_types.csv",
    )
    return p.parse_args()


def threshold_from_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, threshold = precision_recall_curve(y_true, prob)
    if len(threshold) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(threshold[np.argmax(f1)])


def _fit_timesplit_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "Fraud",
    winsorize_cols: list[str] | None = None,
) -> dict[str, float | int | str]:
    train = train.copy()
    val = val.copy()
    test = test.copy()

    if winsorize_cols:
        bounds = train[winsorize_cols].quantile([0.01, 0.99])
        lower = bounds.loc[0.01]
        upper = bounds.loc[0.99]
        for part in (train, val, test):
            part[winsorize_cols] = part[winsorize_cols].clip(
                lower=lower, upper=upper, axis="columns"
            )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[feature_cols])
    X_val = imputer.transform(val[feature_cols])
    X_test = imputer.transform(test[feature_cols])

    y_train = train[label_col].astype(int).to_numpy()
    y_val = val[label_col].astype(int).to_numpy()
    y_test = test[label_col].astype(int).to_numpy()

    model = lgb.LGBMClassifier(**LGBM_OPTUNA_PARAMS)
    model.fit(X_train, y_train)

    prob_val_raw = model.predict_proba(X_val)[:, 1]
    prob_test_raw = model.predict_proba(X_test)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_val_raw, y_val)
    prob_val_cal = iso.predict(prob_val_raw)
    prob_test_cal = iso.predict(prob_test_raw)
    threshold = threshold_from_f1(y_val, prob_val_cal)

    pred = (prob_test_cal >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    return {
        "AUC_raw": roc_auc_score(y_test, prob_test_raw),
        "AUC_calibrated": roc_auc_score(y_test, prob_test_cal),
        "PR_AUC_calibrated": average_precision_score(y_test, prob_test_cal),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Type_I_Error": fp / (fp + tn) if fp + tn else np.nan,
        "Type_II_Error": fn / (fn + tp) if fn + tp else np.nan,
        "Threshold": threshold,
        "N": len(y_test),
        "Fraud_rate": float(y_test.mean()),
    }


def run_timesplit(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "Fraud",
    winsorize_cols: list[str] | None = None,
) -> dict[str, float | int | str]:
    train = df[df["year"].between(2015, 2021)].copy()
    val = df[df["year"] == 2022].copy()
    test = df[df["year"].between(2023, 2024)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("TimeSplit requires non-empty train, validation, and test sets.")
    return _fit_timesplit_model(
        train,
        val,
        test,
        feature_cols=feature_cols,
        label_col=label_col,
        winsorize_cols=winsorize_cols,
    )


def add_industry(df: pd.DataFrame, industry_path: Path) -> pd.DataFrame:
    if not industry_path.exists():
        raise FileNotFoundError(f"Industry file not found: {industry_path}")

    ind = pd.read_excel(industry_path, sheet_name=0, dtype=str)
    code_col = next(
        (c for c in ["Stkcd", "Symbol", "证券代码", "股票代码"] if c in ind.columns),
        None,
    )
    name_col = next(
        (c for c in ["IndustryName", "IndustryName1", "行业名称", "行业"] if c in ind.columns),
        None,
    )
    year_col = next(
        (c for c in ["year", "Year", "年度", "年份"] if c in ind.columns),
        None,
    )
    if code_col is None or name_col is None:
        raise ValueError("Industry file must contain stock-code and industry-name columns.")

    ind = ind.rename(columns={code_col: "Stkcd", name_col: "IndustryName"})
    ind["Stkcd"] = normalize_stkcd(ind["Stkcd"])

    if year_col:
        ind["year"] = pd.to_numeric(ind[year_col], errors="coerce")
        ind = ind[["Stkcd", "year", "IndustryName"]].dropna(subset=["Stkcd", "year"])
        ind = ind.drop_duplicates(["Stkcd", "year"])
        out = df.merge(ind, on=["Stkcd", "year"], how="left")
    else:
        ind = ind[["Stkcd", "IndustryName"]].drop_duplicates("Stkcd")
        out = df.merge(ind, on="Stkcd", how="left")
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        args.input,
        dtype={"Stkcd": str},
        encoding="utf-8-sig",
        low_memory=False,
    )
    df["Stkcd"] = normalize_stkcd(df["Stkcd"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    all_numeric = numeric_feature_columns(df)
    base_numeric = [c for c in all_numeric if c not in MDA_FEATURES]

    rows: list[dict] = []
    rows.append({
        "analysis": "Full sample",
        **run_timesplit(df, all_numeric),
    })

    # The manuscript robustness exercise clips the 57 continuous/base numeric
    # predictors using boundaries computed from the training years only, while
    # retaining the full 83-numeric-feature model.
    winsor_cols = [
        c for c in base_numeric
        if df[c].nunique(dropna=True) > 5
    ]
    if winsor_cols:
        train = df[df["year"].between(2015, 2021)]
        bounds = train[winsor_cols].quantile([0.01, 0.99])
        pd.DataFrame({
            "feature": winsor_cols,
            "lower_1pct": [bounds.loc[0.01, c] for c in winsor_cols],
            "upper_99pct": [bounds.loc[0.99, c] for c in winsor_cols],
        }).to_csv(
            args.output_dir / "Winsorize_Bounds.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rows.append({
            "analysis": "Winsorized Base features",
            **run_timesplit(df, all_numeric, winsorize_cols=winsor_cols),
        })

    # Narrow fraud definition: P2501 + P2502.
    if args.violation.exists():
        viol = pd.read_csv(args.violation, encoding="utf-8-sig")
        if {"Stkcd", "year", "ViolationTypeID"}.issubset(viol.columns):
            narrow = viol[
                viol["ViolationTypeID"].isin(["P2501", "P2502"])
            ][["Stkcd", "year"]].drop_duplicates()
            narrow["Stkcd"] = normalize_stkcd(narrow["Stkcd"])
            narrow["year"] = pd.to_numeric(narrow["year"], errors="coerce")
            narrow["Fraud_narrow"] = 1
            tmp = df.merge(narrow, on=["Stkcd", "year"], how="left")
            tmp["Fraud_narrow"] = tmp["Fraud_narrow"].fillna(0).astype(int)

            check = pd.DataFrame({
                "Category": [
                    "Wide fraud",
                    "Narrow fraud",
                    "Both wide and narrow",
                    "Wide only",
                    "Narrow only",
                    "Neither",
                ],
                "Observations": [
                    int(tmp["Fraud"].sum()),
                    int(tmp["Fraud_narrow"].sum()),
                    int(((tmp["Fraud"] == 1) & (tmp["Fraud_narrow"] == 1)).sum()),
                    int(((tmp["Fraud"] == 1) & (tmp["Fraud_narrow"] == 0)).sum()),
                    int(((tmp["Fraud"] == 0) & (tmp["Fraud_narrow"] == 1)).sum()),
                    int(((tmp["Fraud"] == 0) & (tmp["Fraud_narrow"] == 0)).sum()),
                ],
            })
            check.to_csv(
                args.output_dir / "Narrow_Fraud_Sample_Check.csv",
                index=False,
                encoding="utf-8-sig",
            )

            rows.append({
                "analysis": "Narrow P2501+P2502",
                **run_timesplit(tmp, all_numeric, label_col="Fraud_narrow"),
            })

    # ST/*ST sensitivity uses the ShortName already retained in the modeling
    # dataset when it is available.
    if "ShortName" in df.columns:
        non_st = ~df["ShortName"].astype(str).str.contains(
            r"\*?ST", case=False, regex=True, na=False
        )
        tmp = df.loc[non_st].copy()
        if not tmp.empty:
            rows.append({
                "analysis": "Exclude ST/*ST",
                **run_timesplit(tmp, all_numeric),
            })

    # Industry matching and industry-specific analyses.
    ind_df = add_industry(df, args.industry)
    ind_df["IndustryName"] = ind_df["IndustryName"].fillna("Unknown").astype(str)
    matched = ind_df[ind_df["IndustryName"] != "Unknown"].copy()
    if not matched.empty:
        rows.append({
            "analysis": "Industry-matched only",
            **run_timesplit(matched, all_numeric),
        })

        ind_df["is_manufacturing"] = ind_df["IndustryName"].str.contains(
            "制造业|Manufactur", case=False, regex=True, na=False
        )
        for label, subset in [
            ("Manufacturing", ind_df[ind_df["is_manufacturing"]]),
            (
                "Non-Manufacturing",
                ind_df[(~ind_df["is_manufacturing"]) & (ind_df["IndustryName"] != "Unknown")],
            ),
        ]:
            if subset["Fraud"].sum() > 0 and subset["year"].between(2023, 2024).any():
                rows.append({
                    "analysis": label,
                    **run_timesplit(subset, all_numeric),
                })

        # Industry FE: one-hot encode industry labels using the full dataframe.
        fe = ind_df.copy()
        dummies = pd.get_dummies(
            fe["IndustryName"], prefix="Industry", dummy_na=False, dtype=int
        )
        fe = pd.concat([fe.drop(columns=["IndustryName"]), dummies], axis=1)
        fe_features = numeric_feature_columns(fe)
        rows.append({
            "analysis": "Full + Industry FE",
            **run_timesplit(fe, fe_features),
        })

    pd.DataFrame(rows).to_csv(
        args.output_dir / "robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # VIF / correlation are computed on the 57 continuous Base numeric features,
    # matching the manuscript design. Variables with <=5 unique non-missing
    # values are not included in the VIF calculation.
    vif_cols = [
        c for c in base_numeric
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 5
    ]
    complete = df[vif_cols].copy()
    complete = complete.loc[:, complete.notna().any()]
    complete = complete.fillna(complete.median(numeric_only=True))

    vif_rows = []
    values = complete.to_numpy(dtype=float)
    for i, col in enumerate(complete.columns):
        try:
            vif = variance_inflation_factor(values, i)
        except Exception:
            vif = np.nan
        vif_rows.append({"feature": col, "VIF": vif})
    pd.DataFrame(vif_rows).sort_values(
        "VIF", ascending=False
    ).to_csv(
        args.output_dir / "VIF_Financial_NonFinancial.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corr = complete.corr()
    corr.to_csv(
        args.output_dir / "Correlation_Matrix.csv",
        encoding="utf-8-sig",
    )

    pairs: list[dict] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and abs(value) >= 0.8:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": value,
                })
    high = pd.DataFrame(pairs)
    if not high.empty:
        high = high.sort_values(
            "correlation", key=lambda s: s.abs(), ascending=False
        )
    high.to_csv(
        args.output_dir / "High_Correlation_Pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
