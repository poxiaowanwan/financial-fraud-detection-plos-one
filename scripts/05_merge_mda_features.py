from __future__ import annotations
import argparse
from pathlib import Path

# Make direct execution from the repository root import the local src package.
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))
import pandas as pd
from src.common import MDA_FEATURES

ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(description="Merge an existing MD&A feature file into the model dataset.")
    p.add_argument("--model",type=Path,default=ROOT/"data/processed/model_dataset_base.csv")
    p.add_argument("--mda",type=Path,default=ROOT/"data/processed/mda_features_stage2_notext.pkl")
    p.add_argument("--output",type=Path,default=ROOT/"data/processed/model_dataset_full.csv")
    return p.parse_args()

def main():
    args=parse_args()
    model=pd.read_csv(args.model,dtype={"Stkcd":str},low_memory=False)
    mda=pd.read_pickle(args.mda)
    if "Symbol" in mda.columns and "Stkcd" not in mda.columns:
        mda=mda.rename(columns={"Symbol":"Stkcd"})
    model["Stkcd"]=model["Stkcd"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    mda["Stkcd"]=mda["Stkcd"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    features=[c for c in MDA_FEATURES if c in mda.columns]
    lag=mda[["Stkcd","year"]+features].copy()
    lag["year"]=pd.to_numeric(lag["year"],errors="coerce")+1
    out=model.merge(lag,on=["Stkcd","year"],how="left")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False,encoding="utf-8-sig")
    print("Rows:",len(out))
    print("MD&A feature columns:",len(features))
    print("Coverage:",out[features].notna().any(axis=1).mean() if features else 0.0)

if __name__=="__main__":
    main()
