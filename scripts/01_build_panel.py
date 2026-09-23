from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CODE_CANDS = ["Stkcd", "股票代码", "证券代码", "Symbol"]
DATE_CANDS = [
    "Accper", "统计截止日期", "会计期间", "截止日期",
    "报告期", "年度", "年份", "Date", "EndDate"
]
AUX_COLS = [
    "ShortName", "Typrep", "Source", "Accper",
    "股票简称", "报表类型编码", "公告来源"
]
COL_MAP = {
    "股票代码": "Stkcd",
    "证券代码": "Stkcd",
    "股票简称": "ShortName",
    "统计截止日期": "Accper",
    "报表类型编码": "Typrep",
    "公告来源": "Source",
    "会计期间": "Accper",
    "截止日期": "Accper",
    "报告期": "Accper",
    "年度": "year",
    "年份": "year",
}

def parse_args():
    p = argparse.ArgumentParser(description="Build financial + non-financial panel.")
    p.add_argument("--financial-dir", type=Path, default=ROOT/"data/raw/financial")
    p.add_argument("--nonfinancial-dir", type=Path, default=ROOT/"data/raw/nonfinancial")
    p.add_argument("--output-dir", type=Path, default=ROOT/"data/processed")
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--typrep", default="A")
    return p.parse_args()

def normalize_colname(c):
    if pd.isna(c):
        return ""
    c = str(c).strip().replace("\n", "").replace("\r", "")
    c = re.sub(r"\s+", " ", c)
    return COL_MAP.get(c, c)

def first_existing(df, cands):
    return next((c for c in cands if c in df.columns), None)

def read_csv_auto(path):
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, header=None, dtype=str, encoding=enc)
        except Exception:
            pass
    raise ValueError(f"CSV read failed: {path}")

def read_raw_tables(path):
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            xls = pd.ExcelFile(path)
            tables = []
            for sh in xls.sheet_names:
                try:
                    raw = pd.read_excel(path, sheet_name=sh, header=None, dtype=str)
                    tables.append((sh, raw))
                except Exception as exc:
                    print(f"[SKIP] {path.name}/{sh}: {exc}")
            return tables
        except Exception as exc:
            print(f"[SKIP] {path}: {exc}")
            return []
    if suffix == ".csv":
        return [("csv", read_csv_auto(path))]
    return []

def find_header_row(raw):
    for i in range(min(10, len(raw))):
        vals = [normalize_colname(x) for x in raw.iloc[i].tolist()]
        if any(v in CODE_CANDS for v in vals):
            return i
    return None

def clean_one_table(raw, source_file, sheet_name, start_year, end_year, typrep):
    header_row = find_header_row(raw)
    if header_row is None:
        return None, "No stock-code header row detected."

    header = [normalize_colname(x) for x in raw.iloc[header_row].tolist()]
    keep = [i for i, c in enumerate(header) if c and not c.startswith("Unnamed")]
    df = raw.iloc[header_row + 1:, keep].copy()
    header = [header[i] for i in keep]

    seen = {}
    unique_header = []
    for c in header:
        count = seen.get(c, 0)
        unique_header.append(c if count == 0 else f"{c}__dup{count}")
        seen[c] = count + 1
    df.columns = unique_header

    code_col = first_existing(df, CODE_CANDS)
    date_col = first_existing(df, DATE_CANDS)
    if code_col is None or date_col is None:
        return None, "Required stock-code/date column not found."

    df["Stkcd"] = (
        df[code_col].astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(6)
    )
    df["Accper"] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df["Accper"].dt.year.astype("Int64")

    df = df[df["Stkcd"].str.match(r"^\d{6}$", na=False)]
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(start_year, end_year)]

    if typrep and "Typrep" in df.columns:
        df = df[
            df["Typrep"].astype(str).str.strip().str.upper()
            == typrep.upper()
        ]

    drop_cols = [
        c for c in AUX_COLS + [code_col, date_col]
        if c in df.columns and c not in {"Stkcd", "year"}
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates(["Stkcd", "year"], keep="last")
    df["_source_file"] = source_file
    df["_source_sheet"] = sheet_name
    return df, None

def read_folder(folder, label, start_year, end_year, typrep):
    all_dfs = []
    logs = []
    files = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv"}
    )
    print(f"\nReading {label}: {len(files)} file(s)")
    for path in files:
        for sheet, raw in read_raw_tables(path):
            if raw.empty:
                continue
            df, error = clean_one_table(
                raw, path.name, sheet, start_year, end_year, typrep
            )
            if error:
                logs.append({
                    "folder": label,
                    "file": path.name,
                    "sheet": sheet,
                    "status": "skipped",
                    "message": error,
                })
                continue
            all_dfs.append(df)
            logs.append({
                "folder": label, "file": path.name, "sheet": sheet,
                "status": "success", "rows": len(df),
                "firms": df["Stkcd"].nunique(),
                "min_year": int(df["year"].min()) if len(df) else None,
                "max_year": int(df["year"].max()) if len(df) else None,
                "columns": df.shape[1],
            })

    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame(logs)

    base = all_dfs[0].drop(columns=["_source_file", "_source_sheet"], errors="ignore")
    for idx, right in enumerate(all_dfs[1:], 1):
        right = right.drop(columns=["_source_file", "_source_sheet"], errors="ignore")
        overlap = [c for c in right.columns if c in base.columns and c not in {"Stkcd","year"}]
        right = right.rename(columns={c: f"{c}__{idx}" for c in overlap})
        base = pd.merge(base, right, on=["Stkcd","year"], how="outer")
    return base, pd.DataFrame(logs)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fin, fin_logs = read_folder(
        args.financial_dir, "financial",
        args.start_year, args.end_year, args.typrep
    )
    nf, nf_logs = read_folder(
        args.nonfinancial_dir, "non-financial",
        args.start_year, args.end_year, args.typrep
    )
    if fin.empty:
        raise SystemExit("No valid financial data were found.")

    fin.to_csv(args.output_dir/"financial_cleaned.csv", index=False, encoding="utf-8-sig")
    if not nf.empty:
        nf.to_csv(args.output_dir/"nonfinancial_cleaned.csv", index=False, encoding="utf-8-sig")

    if not nf.empty:
        overlap = [c for c in nf.columns if c in fin.columns and c not in {"Stkcd","year"}]
        nf2 = nf.rename(columns={c: f"{c}_nf" for c in overlap})
        merged = pd.merge(fin, nf2, on=["Stkcd","year"], how="left", suffixes=("","_nf"))
        merged_inner = pd.merge(fin, nf2, on=["Stkcd","year"], how="inner", suffixes=("","_nf"))
    else:
        merged = fin.copy()
        merged_inner = fin.copy()

    merged.to_csv(args.output_dir/"financial_nonfinancial_panel.csv", index=False, encoding="utf-8-sig")
    merged_inner.to_csv(args.output_dir/"financial_nonfinancial_panel_inner.csv", index=False, encoding="utf-8-sig")

    fin_keys = set(zip(fin["Stkcd"], fin["year"]))
    nf_keys = set(zip(nf["Stkcd"], nf["year"])) if not nf.empty else set()
    diag = pd.DataFrame([
        {"item":"financial_observations","value":len(fin)},
        {"item":"nonfinancial_observations","value":len(nf)},
        {"item":"financial_unique_firm_year","value":len(fin_keys)},
        {"item":"nonfinancial_unique_firm_year","value":len(nf_keys)},
        {"item":"matched_keys","value":len(fin_keys & nf_keys)},
        {"item":"financial_only_keys","value":len(fin_keys - nf_keys)},
        {"item":"nonfinancial_only_keys","value":len(nf_keys - fin_keys)},
        {"item":"left_panel_rows","value":len(merged)},
        {"item":"inner_panel_rows","value":len(merged_inner)},
    ])
    diag.to_csv(args.output_dir/"panel_match_diagnostics.csv", index=False, encoding="utf-8-sig")
    fin_logs.to_csv(args.output_dir/"financial_read_log.csv", index=False, encoding="utf-8-sig")
    nf_logs.to_csv(args.output_dir/"nonfinancial_read_log.csv", index=False, encoding="utf-8-sig")
    print(diag.to_string(index=False))

if __name__ == "__main__":
    main()
