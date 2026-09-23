from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET_TYPES = {"P2501", "P2502", "P2503", "P2506"}
TYPE_NAME = {
    "P2501": "Fabricated Profits",
    "P2502": "Fictitious Asset Reporting",
    "P2503": "False or Misleading Disclosures",
    "P2506": "Inaccurate Disclosure (Other)",
}

def parse_args():
    p = argparse.ArgumentParser(description="Construct firm-year fraud labels.")
    p.add_argument("--violation", type=Path, default=ROOT/"data/raw/violation")
    p.add_argument("--panel", type=Path, default=ROOT/"data/processed/financial_nonfinancial_panel.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT/"data/processed")
    return p.parse_args()

def normalize_stkcd(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace(".0", "")
    m = re.search(r"(\d{1,6})", s)
    return m.group(1).zfill(6) if m else None

def find_violation_file(path):
    if path.is_file():
        return path
    files = sorted(
        list(path.glob("*.xlsx")) +
        list(path.glob("*.xls")) +
        list(path.glob("*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"No violation file under {path}")
    for f in files:
        if "违规" in f.name or "violation" in f.name.lower():
            return f
    return files[0]

def read_violation_file(path):
    f = find_violation_file(path)
    if f.suffix.lower() in {".xlsx",".xls"}:
        raw = pd.read_excel(f, header=None, nrows=20)
        header_idx = 0
        for i in range(len(raw)):
            row = raw.iloc[i].astype(str).tolist()
            if any("ViolationID" in x for x in row) and any("Symbol" in x for x in row):
                header_idx = i
                break
        df = pd.read_excel(f, header=header_idx, dtype=str)
    else:
        df = pd.read_csv(f, dtype=str)

    rename = {}
    for c in df.columns:
        s = str(c)
        if "ViolationID" in s or "违规事件ID" in s: rename[c] = "ViolationID"
        elif "Symbol" in s or "证券代码" in s: rename[c] = "Symbol"
        elif "DisposalDate" in s or "处理文件日期" in s: rename[c] = "DisposalDate"
        elif "DeclareDate" in s or "公告日期" in s: rename[c] = "DeclareDate"
        elif "ViolationTypeID" in s or "违规类型编码" in s: rename[c] = "ViolationTypeID"
        elif "ViolationYear" in s or "违规年度" in s: rename[c] = "ViolationYear"
        elif "IsViolated" in s or "上市公司是否违规" in s: rename[c] = "IsViolated"
    df = df.rename(columns=rename)
    required = {"Symbol","ViolationTypeID","ViolationYear","IsViolated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing violation columns: {sorted(missing)}")
    df["Stkcd"] = df["Symbol"].apply(normalize_stkcd)
    df = df[df["Stkcd"].notna()].copy()
    df["IsViolated"] = df["IsViolated"].astype(str).str.strip().str.upper()
    return df, f

def split_semicolon(s):
    if pd.isna(s):
        return []
    s = str(s).replace("；",";").replace("\n",";").replace("\r",";")
    return [p.strip() for p in s.split(";") if p.strip()]

def split_types(s):
    if pd.isna(s):
        return []
    s = str(s)
    for ch in ["、",",","，","；",";"," "]:
        s = s.replace(ch,"|")
    return [p.strip() for p in s.split("|") if p.strip()]

def extract_year(value):
    if pd.isna(value):
        return None
    m = re.search(r"(19|20)\d{2}", str(value))
    return int(m.group(0)) if m else None

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viol, source = read_violation_file(args.violation)
    viol = viol[viol["IsViolated"] == "Y"].copy()

    rows = []
    for _, r in viol.iterrows():
        type_groups = split_semicolon(r.get("ViolationTypeID"))
        year_groups = split_semicolon(r.get("ViolationYear"))
        if not type_groups:
            continue

        if len(year_groups) == len(type_groups):
            pairs = list(zip(type_groups, year_groups))
        elif len(year_groups) == 1:
            pairs = [(tg, year_groups[0]) for tg in type_groups]
        elif len(type_groups) == 1:
            pairs = [(type_groups[0], yg) for yg in year_groups]
        else:
            pairs = [
                (tg, year_groups[i] if i < len(year_groups)
                 else (year_groups[-1] if year_groups else None))
                for i, tg in enumerate(type_groups)
            ]

        for type_group, year_value in pairs:
            year = extract_year(year_value)
            year_source = "ViolationYear"
            if year is None:
                year = extract_year(r.get("DeclareDate"))
                year_source = "DeclareDate_fallback"
            if year is None:
                year = extract_year(r.get("DisposalDate"))
                year_source = "DisposalDate_fallback"

            for violation_type in split_types(type_group):
                rows.append({
                    "ViolationID": r.get("ViolationID"),
                    "Stkcd": r["Stkcd"],
                    "year": year,
                    "ViolationTypeID": violation_type,
                    "ViolationTypeName": TYPE_NAME.get(violation_type, violation_type),
                    "IsViolated": r.get("IsViolated"),
                    "DisposalDate": r.get("DisposalDate"),
                    "DeclareDate": r.get("DeclareDate"),
                    "year_source": year_source,
                })

    expanded = pd.DataFrame(rows)
    target = expanded[
        expanded["ViolationTypeID"].isin(TARGET_TYPES)
        & expanded["year"].notna()
        & expanded["year"].between(2015, 2024)
    ].copy()

    fraud_years = (
        target.groupby(["Stkcd","year"])
        .agg(
            fraud_types=("ViolationTypeID", lambda x: ";".join(sorted(set(x)))),
            fraud_count=("ViolationTypeID","size"),
        )
        .reset_index()
        .sort_values(["Stkcd","year"])
    )
    fraud_years["Fraud"] = 1
    fraud_years["first_fraud"] = (~fraud_years.duplicated("Stkcd")).astype(int)
    fraud_years["repeat_fraud"] = fraud_years.duplicated("Stkcd").astype(int)

    panel = pd.read_csv(args.panel, encoding="utf-8-sig", low_memory=False)
    panel["Stkcd"] = panel["Stkcd"].apply(normalize_stkcd)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel.merge(
        fraud_years[["Stkcd","year","Fraud","fraud_types","fraud_count","first_fraud","repeat_fraud"]],
        on=["Stkcd","year"], how="left"
    )
    for col in ["Fraud","first_fraud","repeat_fraud"]:
        panel[col] = panel[col].fillna(0).astype(int)
    panel["fraud_types"] = panel["fraud_types"].fillna("")

    expanded.to_csv(args.output_dir/"violation_expanded_firm_year_type.csv", index=False, encoding="utf-8-sig")
    target.to_csv(args.output_dir/"target_four_fraud_types.csv", index=False, encoding="utf-8-sig")
    fraud_years.to_csv(args.output_dir/"fraud_labels_firm_year.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(args.output_dir/"financial_nonfinancial_fraud_panel.csv", index=False, encoding="utf-8-sig")

    with open(args.output_dir/"fraud_label_diagnostics.txt","w",encoding="utf-8") as f:
        f.write(f"Violation source: {source}\n")
        f.write(f"Raw rows: {len(viol)}\n")
        f.write(f"Expanded rows: {len(expanded)}\n")
        f.write(f"Target four-category rows: {len(target)}\n")
        f.write(f"Fraud firm-year observations: {len(fraud_years)}\n")
        f.write(f"Fraud firms: {fraud_years['Stkcd'].nunique()}\n")
        f.write("\nViolation types:\n")
        f.write(target["ViolationTypeID"].value_counts().to_string())
        f.write("\n\nMerged panel:\n")
        f.write(str(panel.shape))
        f.write("\n\nFraud distribution:\n")
        f.write(panel["Fraud"].value_counts().to_string())

if __name__ == "__main__":
    main()
