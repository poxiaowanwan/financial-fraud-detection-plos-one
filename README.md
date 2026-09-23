# Financial Fraud Detection: Reproducible Python Code

This repository contains the cleaned, path-independent Python implementation of the experiments reported in:

**Financial fraud detection through multi-dimensional feature fusion: a machine learning approach incorporating financial, non-financial, and MD&A tone indicators**

The original work was developed in several exploratory Jupyter notebooks. The public-release version removes notebook duplication, debugging cells, and machine-specific paths while preserving the substantive modeling workflow used for the manuscript.

## Reproduction workflow

All commands are run from the repository root. Default input/output locations are project-relative; no script requires a machine-specific absolute path.

```bash
python scripts/00_validate_inputs.py
python scripts/01_build_panel.py
python scripts/02_build_fraud_labels.py
python scripts/03_build_model_dataset.py
python scripts/04_process_mda.py

python scripts/06_groupkfold_models.py
python scripts/07_timesplit_lightgbm.py
python scripts/08_stacking.py
python scripts/12_threshold_cost.py
python scripts/13_smote_sensitivity.py
python scripts/09_robustness.py
python scripts/10_explainability.py
python scripts/11_statistics.py
python scripts/14_build_main_tables.py
python scripts/15_violation_type_analysis.py
python scripts/16_dictionary_similarity.py
python scripts/17_make_figures.py
```

`05_merge_mda_features.py` is a convenience script for users who already have an independently generated MD&A feature file and do not need to rerun the full MD&A pipeline.

## Experimental design captured by the scripts

- Sample: Chinese A-share listed firms, 2015–2024.
- Prediction: year `t` fraud from firm information in year `t-1`.
- Fraud labels: P2501, P2502, P2503, and P2506.
- Raw feature sets: Base = 58 financial/non-financial features; Full = Base + 26 MD&A features = 84 raw features.
- One-hot encoding expands the model matrix to 89 columns because audit-opinion type is categorical.
- Reported SHAP ranking retains the 83 numeric features and excludes the six audit-opinion dummy columns.
- Primary out-of-time evaluation: 2015–2021 training, 2022 validation, 2023–2024 testing.
- Complementary evaluation: five-fold GroupKFold with firm-level isolation.
- Primary model: LightGBM with Optuna hyperparameter search and isotonic calibration.
- Class imbalance: `class_weight='balanced'`; SMOTE is evaluated as a sensitivity analysis.
- Statistical inference: DeLong tests, 1,000-resample bootstrap confidence intervals, and McNemar tests.
- Explainability: SHAP, permutation importance, and five-fold SHAP stability.
- Robustness: narrow fraud definition, ST/*ST exclusion, training-sample winsorization, industry matching/stratification, industry fixed effects, and multicollinearity diagnostics.

## Data and licensing

The experiments use CSMAR and related licensed data. These raw files are **not** bundled with this repository and should not be uploaded publicly unless redistribution is explicitly permitted.

Place permitted local copies under `data/raw/` following `data/raw/README.md`. The code discovers financial and non-financial exports recursively, while the MD&A, violation, industry, and DLUT inputs have documented canonical names.

## Environment

Recommended Python: 3.10 or 3.11.

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Every script supports `--help`, so input and output locations can be changed without editing source code.

Example:

```bash
python scripts/07_timesplit_lightgbm.py --input data/processed/model_dataset_full.csv --output-dir outputs/timesplit
```

## Repository layout

```text
financial-fraud-reproducibility/
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── common.py
├── scripts/
├── tests/
├── docs/
└── data/
    └── raw/
```

The original exploratory notebooks are intentionally excluded from the public release. They remain useful as private provenance records but are not required for reproduction.
