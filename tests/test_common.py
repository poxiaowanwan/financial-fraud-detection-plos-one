from pathlib import Path
import sys
PROJECT_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_ON_PATH))

import numpy as np
import pandas as pd

from src.common import (
    MDA_FEATURES,
    best_f1_threshold,
    bootstrap_ci,
    compute_metrics,
    delong_test,
    get_base_full_features,
    numeric_feature_columns,
)


def test_feature_partition():
    df = pd.DataFrame({
        "Stkcd": ["000001", "000002"],
        "year": [2020, 2021],
        "Fraud": [0, 1],
        "TypeAuditOpin": ["A", "B"],
        "x1": [1.0, 2.0],
        "TextualSimilarity": [0.9, 0.8],
    })
    base, full = get_base_full_features(df)
    assert full == ["x1", "TextualSimilarity"]
    assert base == ["x1"]
    assert len(MDA_FEATURES) == 26


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.7, 0.9])
    threshold = best_f1_threshold(y, p)
    metrics = compute_metrics(y, p, threshold)
    assert 0 <= threshold <= 1
    assert 0 <= metrics["AUC"] <= 1
    assert metrics["Recall"] == 1.0


def test_bootstrap_and_delong():
    y = np.array([0, 0, 0, 1, 1, 1])
    p1 = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])
    p2 = np.array([0.10, 0.15, 0.25, 0.65, 0.75, 0.95])
    lo, hi = bootstrap_ci(y, p1, metric="auc", n_boot=100, seed=42)
    assert lo <= hi
    result = delong_test(y, p1, p2)
    assert 0 <= result["p_value"] <= 1


def test_numeric_feature_exclusion():
    df = pd.DataFrame({
        "Stkcd": ["1"], "year": [2020], "Fraud": [0],
        "fraud_count": [0], "first_fraud": [0], "repeat_fraud": [0],
        "x": [1.0], "TextualSimilarity": [0.9],
    })
    cols = numeric_feature_columns(df)
    assert cols == ["x", "TextualSimilarity"]
