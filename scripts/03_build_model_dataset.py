from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    p = argparse.ArgumentParser(description="Build one-year-lagged modeling dataset.")
    p.add_argument("--input", type=Path, default=ROOT/"data/processed/financial_nonfinancial_fraud_panel.csv")
    p.add_argument("--output", type=Path, default=ROOT/"data/processed/model_dataset_base.csv")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2024)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input, dtype={"Stkcd": str}, low_memory=False)
    df["Stkcd"] = (
        df["Stkcd"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    id_cols = {"Stkcd","year","Fraud","fraud_types","fraud_count","first_fraud","repeat_fraud"}
    feature_cols = [c for c in df.columns if c not in id_cols]

    df = df.sort_values(["Stkcd","year"]).copy()
    lagged = df[["Stkcd","year"] + feature_cols].copy()
    lagged[feature_cols] = lagged.groupby("Stkcd")[feature_cols].shift(1)

    model = df[["Stkcd","year","Fraud","fraud_types","fraud_count","first_fraud","repeat_fraud"]].copy()
    lagged = lagged.rename(columns={c:f"lag1_{c}" for c in feature_cols})
    model = model.merge(lagged, on=["Stkcd","year"], how="left")
    model = model[model["year"].between(args.start_year,args.end_year)].copy()

    lag_cols = [c for c in model.columns if c.startswith("lag1_")]
    model = model.dropna(subset=lag_cols, how="all")
    model = model.rename(columns={f"lag1_{c}":c for c in feature_cols})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(args.output, index=False, encoding="utf-8-sig")

    isolated = model.copy()
    train_firms = set(isolated.loc[isolated["year"].between(2015,2021),"Stkcd"])
    test_mask = isolated["year"].between(2023,2024)
    overlap = train_firms & set(isolated.loc[test_mask,"Stkcd"])
    isolated = isolated[~(test_mask & isolated["Stkcd"].isin(overlap))].copy()
    isolated.to_csv(args.output.with_name("model_dataset_company_isolated.csv"), index=False, encoding="utf-8-sig")

    diag = {
        "shape": str(model.shape),
        "firms": int(model["Stkcd"].nunique()),
        "fraud_observations": int(model["Fraud"].sum()),
        "fraud_rate": float(model["Fraud"].mean()),
        "train_2015_2021": int(model["year"].between(2015,2021).sum()),
        "validation_2022": int((model["year"]==2022).sum()),
        "test_2023_2024": int(model["year"].between(2023,2024).sum()),
        "train_test_overlapping_firms_before_isolation": int(len(overlap)),
        "test_2023_2024_after_company_isolation": int(isolated["year"].between(2023,2024).sum()),
    }
    with open(args.output.parent/"model_dataset_diagnostics.txt","w",encoding="utf-8") as f:
        for k,v in diag.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()
