# Final pre-deposit checklist

## Dataset
- [ ] `data/PLOS_ONE_minimal_dataset.csv` is the final minimal dataset.
- [ ] The file contains 48,328 rows and the expected columns.
- [ ] `FirmID` is stable across years and contains no direct personal identifiers.
- [ ] The dataset contains 84 raw predictors.
- [ ] `TypeAuditOpin` has the six audit-opinion categories used in the analysis.

## Code
- [ ] `python -m compileall .` completes without errors.
- [ ] `python run_all.py` completes without errors.
- [ ] `outputs/Optuna_metrics.csv` gives the expected primary TimeSplit results.
- [ ] `outputs/Table3_MainResults.csv` matches the manuscript Table 3.
- [ ] `outputs/DeLong_ClassWeight_vs_SMOTE.csv` matches the manuscript Table 4 comparison.
- [ ] `outputs/DeLong_Stacking_vs_Internal_BaseLearners.csv` uses internal Stacking base learners.
- [ ] `outputs/SHAP_global_importance.csv` contains exactly 83 reported features.
- [ ] `TextualSimilarity` is rank 10/83.
- [ ] `outputs/Robustness_Checks.csv` matches Table 5.
- [ ] `outputs/Threshold_Cost_Summary.csv` matches Table 6.

## Repository
- [ ] No absolute machine paths remain.
- [ ] No passwords, API keys, tokens, or private files are present.
- [ ] No original restricted CSMAR database files are included.
- [ ] No temporary notebooks or exploratory scripts are included unless intentionally archived.
- [ ] GitHub repository URL is added to `CITATION.cff`.
- [ ] Zenodo DOI is added to `CITATION.cff` and the manuscript Data Availability Statement.
- [ ] GitHub release/tag is created before the Zenodo archive is published.
