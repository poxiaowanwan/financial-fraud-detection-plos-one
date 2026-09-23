from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    parser = argparse.ArgumentParser(description="Check the local data layout.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    return parser.parse_args()

def main():
    args = parse_args()
    root = args.data_root
    checks = {
        "financial_dir": root / "financial",
        "nonfinancial_dir": root / "nonfinancial",
        "mda_dir": root / "mda",
        "dlut_dictionary": root / "dlut" / "dlut_sentiment.xlsx",
        "violation_file_or_dir": root / "violation",
        "industry_file_or_dir": root / "industry",
    }
    print("Input layout check")
    print("=" * 70)
    ok = True
    for name, path in checks.items():
        exists = path.exists()
        if name.endswith("_dir"):
            files = list(path.glob("*")) if exists else []
            status = f"{len(files)} file(s)" if exists else "MISSING"
        else:
            status = "OK" if exists else "MISSING"
        print(f"{name:24s} {status:>12s}  {path}")
        ok = ok and exists
    print("=" * 70)
    print("Directories may contain multiple raw files.")
    if not ok:
        raise SystemExit("Some required inputs are missing. See data/raw/README.md.")

if __name__ == "__main__":
    main()
