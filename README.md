# Financial Fraud Detection — Reproducibility Package

This repository contains the Python code and the minimal dataset needed to reproduce the empirical analyses reported in the PLOS ONE manuscript:

**Financial fraud detection through multi-dimensional feature fusion: a machine learning approach incorporating financial, non-financial, and MD&A tone indicators**

## 1. What is included

- `data/PLOS_ONE_minimal_dataset.csv`: the author-prepared minimal dataset used for reproduction.
- `src/`: modular Python scripts for preprocessing, model fitting, statistical testing, robustness checks, explainability, and figures.
- `run_all.py`: one-command entry point.
- `outputs/`: generated automatically; it is intentionally not populated in the repository.

The underlying restricted CSMAR source files are **not** redistributed. The minimal dataset contains only the variables required for the analyses reported in the manuscript.

## 2. Dataset structure

The minimal dataset should contain:

### Identifiers and outcome variables

- `FirmID`: stable de-identified firm identifier. The same firm must retain the same `FirmID` across years.
- `year`: fiscal year.
- `Fraud`: primary fraud label.
- `Fraud_narrow`: narrow robustness label based on P2501 + P2502.
- `fraud_types`, `fraud_count`, `first_fraud`, `repeat_fraud`: supplementary fraud/offender information.

### Robustness variables

- `IndustryName`
- `IndustryCategory`
- `is_ST`
- `is_manufacturing`
- `set` (retained only for compatibility; the main analysis uses calendar year rather than this column).

### Predictors

The main model uses **84 raw predictors**:

- 40 financial variables;
- 18 non-financial variables;
- 26 MD&A textual variables.

`TypeAuditOpin` is the categorical audit-opinion variable. One-hot encoding produces six audit-opinion dummy variables, resulting in **89 model-input columns**.

For the reported SHAP ranking, those six audit-opinion dummy variables are excluded, leaving **83 numeric/original numerical-binary features**. TextualSimilarity is therefore reported as rank **10/83**.

## 3. Main evaluation design

The manuscript uses two complementary designs:

1. **Five-fold GroupKFold** grouped by `FirmID`, preventing the same firm from appearing in both training and evaluation folds.
2. **Out-of-time evaluation:** 2015–2021 training, 2022 validation, and 2023–2024 test.

The primary model is LightGBM with 30-trial Optuna optimization and isotonic probability calibration. Hyperparameter optimization uses a three-fold GroupKFold within the 2015–2021 training period and optimizes PR-AUC. Calibration is fitted on the 2022 validation period and applied to the 2023–2024 test period.

## 4. Installation

Python 3.10–3.13 is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the complete analysis

Place the full minimal dataset at:

```text
data/PLOS_ONE_minimal_dataset.csv
```

Then run:

```bash
python run_all.py
```

The scripts run in dependency order:

1. `run_optuna.py` — Optuna tuning, isotonic calibration, primary TimeSplit predictions.
2. `run_main.py` — main model comparisons and Table 3 outputs.
3. `run_smote.py` — class weighting vs SMOTE using the same GroupKFold observations.
4. `run_stacking.py` — Stacking, DeLong, and McNemar analyses.
5. `run_threshold_cost.py` — FN:FP threshold-cost analysis.
6. `run_shap.py` — SHAP and permutation importance.
7. `run_robustness.py` — ST, industry, narrow-label, winsorization, and industry-FE checks.
8. `make_figures.py` — figures supported by the generated output files.

All generated files are written to `outputs/`.

## 6. Important reproducibility notes

- No absolute/local machine paths are used. Paths are resolved relative to the repository root.
- The code never uses the final test labels to fit the model, calibrator, or hyperparameter search.
- Missing-value imputation is fitted on the relevant training partition and then applied to validation/test data.
- One-hot encoding is fitted on the relevant training partition with `handle_unknown='ignore'`.
- SMOTE is applied only inside training folds.
- GroupKFold uses `FirmID`, not row indices.
- The same observation set is used when comparing class weighting and SMOTE.
- Stacking comparisons are explicitly against its **internal base learners**, not the standalone models in Table 3.
- SHAP is calculated from LightGBM raw-margin contributions; the reported 83-feature ranking excludes the six audit-opinion dummy variables.

## 7. Important numerical note

The repository is intended to reproduce the final analysis design, not to embed previously generated result files. Numerical outputs should be generated from the complete minimal dataset supplied with the final repository.

The published reference values include, among others:

- out-of-time raw AUC: 0.7668;
- out-of-time calibrated AUC: 0.7642;
- out-of-time PR-AUC: 0.2489;
- out-of-time threshold: 0.17;
- out-of-time Recall: 0.5142;
- GroupKFold calibrated AUC: 0.7873;
- Stacking AUC: 0.7889;
- narrow-definition AUC: 0.7989;
- narrow-definition PR-AUC: 0.0376.

Because the final repository code removes several implementation ambiguities from the working scripts, the author should run the complete package once on the final dataset and verify the numerical lock before the GitHub/Zenodo deposit is made.

## 8. Data availability and licensing

The minimal dataset is provided only to the extent permitted by the authors' data-access rights and the applicable terms governing the underlying CSMAR data. The repository does not redistribute the original CSMAR database.

The code is released under the MIT License. The dataset is not automatically covered by the software license; its use remains subject to the applicable data-access terms.

## 9. Citation

Please cite the associated PLOS ONE article and, after publication of the archived repository, the Zenodo DOI.
