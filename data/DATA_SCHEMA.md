# Minimal dataset schema

The final CSV should contain 48,328 firm-year observations and the following logical groups.

## Identification/outcomes

`FirmID`, `year`, `Fraud`, `Fraud_narrow`, `fraud_types`, `fraud_count`, `first_fraud`, `repeat_fraud`

## Robustness metadata

`IndustryName`, `IndustryCategory`, `is_ST`, `is_manufacturing`, `set`

## Raw predictors

Exactly 84 columns:

- 40 financial predictors
- 18 non-financial predictors
- 26 MD&A predictors

The categorical variable `TypeAuditOpin` must be present among the 18 non-financial predictors.

After one-hot encoding, the expected model input dimensionality is 89 (83 original numeric/binary variables + 6 audit-opinion dummy variables).
