from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

MDA_FEATURES = [
    "TextualSimilarity",
    "PositiveVocabularyNum",
    "NegativeVocabularyNum",
    "EmotionTone1",
    "EmotionTone2",
    "PosRatio",
    "NegRatio",
    "SentLenAvg",
    "SentLenStd",
    "ComplexWordRatio",
    "DigitDensity",
    "PuncDensity",
    "TTR",
    "Jaccard_prev",
    "EditSim_prev",
    "TFIDF_Cosine_prev",
    "DLUT_PosNum",
    "DLUT_NegNum",
    "DLUT_PosRatio",
    "DLUT_NegRatio",
    "DLUT_PosIntensity",
    "DLUT_NegIntensity",
    "DLUT_EmotionScore",
    "DLUT_EmotionTone",
    "DLUT_NegAfterNeg",
    "DLUT_PosAfterNeg",
]

EXCLUDE_COLS = [
    "Stkcd", "year", "Fraud", "ShortName", "IndustryName1",
    "ViolationTypeID", "ViolationID", "DeclareDate", "DisposalDate", "Enddate", "set",
    "fraud_types", "fraud_count", "first_fraud", "repeat_fraud", "Fraud_narrow", "is_manufacturing",
]

LGBM_OPTUNA_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.0176,
    num_leaves=25,
    min_child_samples=21,
    subsample=0.998,
    subsample_freq=1,
    colsample_bytree=0.824,
    reg_alpha=0.250,
    reg_lambda=0.277,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

LGBM_DEFAULT_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=10,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

LASSO_PARAMS = dict(
    penalty="l1",
    solver="saga",
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
    n_jobs=-1,
)


def normalize_stkcd(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(6)
    )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def numeric_feature_columns(
    df: pd.DataFrame,
    exclude: Sequence[str] = EXCLUDE_COLS,
) -> list[str]:
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def get_base_full_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_numeric = numeric_feature_columns(df)
    mda_cols = [c for c in MDA_FEATURES if c in all_numeric]
    base_cols = [c for c in all_numeric if c not in mda_cols]
    return base_cols, all_numeric


def one_hot_features(
    df: pd.DataFrame,
    exclude: Sequence[str] = EXCLUDE_COLS,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    feature_cols = [c for c in out.columns if c not in exclude]
    obj_cols = [c for c in feature_cols if out[c].dtype == "object"]
    if obj_cols:
        out = pd.get_dummies(
            out,
            columns=obj_cols,
            prefix=obj_cols,
            dummy_na=False,
        )
    feature_cols = [c for c in out.columns if c not in exclude]
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, feature_cols, obj_cols


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, threshold = precision_recall_curve(y_true, prob)
    if len(threshold) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(threshold[np.argmax(f1)])


def compute_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()
    return {
        "AUC": roc_auc_score(y_true, prob),
        "PR_AUC": average_precision_score(y_true, prob),
        "Brier": brier_score_loss(y_true, prob),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Type_I_Error": fp / (fp + tn) if (fp + tn) else np.nan,
        "Type_II_Error": fn / (fn + tp) if (fn + tp) else np.nan,
        "Threshold": threshold,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(pred_sorted: np.ndarray, label_1_count: int):
    m = label_1_count
    n = pred_sorted.shape[1] - m
    pos = pred_sorted[:, :m]
    neg = pred_sorted[:, m:]
    k = pred_sorted.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])

    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(pred_sorted[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    order = np.argsort(-y_true)
    m = int(y_true.sum())
    pred_sorted = np.vstack([p1, p2])[:, order]
    aucs, cov = fast_delong(pred_sorted, m)
    l = np.array([[1.0, -1.0]])
    variance = float(np.dot(np.dot(l, cov), l.T).ravel()[0])
    z = float(abs(aucs[0] - aucs[1]) / np.sqrt(max(variance, 1e-18)))
    p = float(2 * stats.norm.sf(z))
    return {
        "AUC_1": float(aucs[0]),
        "AUC_2": float(aucs[1]),
        "z": z,
        "p_value": p,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    prob: np.ndarray,
    metric: str = "auc",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        p_b = prob[idx]
        if metric == "auc":
            values.append(roc_auc_score(y_b, p_b))
        elif metric == "pr_auc":
            values.append(average_precision_score(y_b, p_b))
        elif metric == "brier":
            values.append(brier_score_loss(y_b, p_b))
        else:
            raise ValueError(metric)
    if not values:
        return np.nan, np.nan
    return tuple(np.percentile(values, [2.5, 97.5]))


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "Fraud",
):
    X = df[list(feature_cols)].copy()
    y = df[label_col].astype(int).to_numpy()
    return X, y
