#!/usr/bin/env python3
"""
Data Fusion Contest 2026 Task 2: quality-first training pipeline + EDA.
Kaggle-friendly script (no argparse).

Pipeline:
1) Load (or download) data
2) Run EDA report
3) Feature hygiene (constant + ultra-missing drop)
4) Train two ensembles:
   - Multi-label CatBoost (MultiLogloss)
   - One-vs-Rest CatBoost (Logloss/AUC)
5) Learn best blend weight on OOF by Macro ROC-AUC
6) Build submission.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# =========================
# CONFIG (edit on Kaggle)
# =========================
DATA_DIR = Path("/kaggle/input/data-fusion-contest-2026/data")
OUTPUT_PATH = Path("submission.parquet")
EDA_REPORT_PATH = Path("eda_report.md")

# If DATA_DIR does not exist, try kagglehub download.
USE_KAGGLEHUB_DOWNLOAD = False
KAGGLE_DATASET_ID = "hatab123/data-fusion-contest-2026"

MULTI_FOLDS = 5
OVR_FOLDS = 5
MULTI_SEEDS = [42, 1337]
OVR_SEEDS = [2026]

# If True, blend weight between Multi and OvR is optimized by OOF macro ROC-AUC.
AUTO_TUNE_BLEND_WEIGHT = True
DEFAULT_BLEND_WEIGHT_MULTI = 0.65

# Feature hygiene
DROP_CONST_FEATURES = True
MISSING_RATE_THRESHOLD = 0.997  # drop columns with >99.7% missing


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def resolve_data_dir() -> Path:
    if DATA_DIR.exists():
        return DATA_DIR

    if not USE_KAGGLEHUB_DOWNLOAD:
        raise FileNotFoundError(
            f"DATA_DIR not found: {DATA_DIR}. Set correct path or enable USE_KAGGLEHUB_DOWNLOAD."
        )

    import kagglehub

    downloaded_root = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    print(f"Downloaded dataset root: {downloaded_root}")

    if (downloaded_root / "data").exists():
        return downloaded_root / "data"
    return downloaded_root


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_main = pd.read_parquet(data_dir / "train_main_features.parquet")
    train_extra = pd.read_parquet(data_dir / "train_extra_features.parquet")
    test_main = pd.read_parquet(data_dir / "test_main_features.parquet")
    test_extra = pd.read_parquet(data_dir / "test_extra_features.parquet")
    target = pd.read_parquet(data_dir / "train_target.parquet")
    sample_submit = pd.read_parquet(data_dir / "sample_submit.parquet")
    return train_main, train_extra, test_main, test_extra, target, sample_submit


def merge_features(main_df: pd.DataFrame, extra_df: pd.DataFrame) -> pd.DataFrame:
    cols_to_add = [c for c in extra_df.columns if c not in main_df.columns]
    return pd.concat([main_df, extra_df[cols_to_add]], axis=1)


def run_eda(train_full: pd.DataFrame, test_full: pd.DataFrame, target: pd.DataFrame, report_path: Path) -> None:
    feature_cols = [c for c in train_full.columns if c != "customer_id"]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c.startswith("num_feature")]

    miss_train = train_full[feature_cols].isna().mean().sort_values(ascending=False)
    miss_test = test_full[feature_cols].isna().mean().sort_values(ascending=False)

    target_cols = [c for c in target.columns if c != "customer_id"]
    pos_rate = target[target_cols].mean().sort_values(ascending=False)

    lines = []
    lines.append("# EDA report\n")
    lines.append(f"- train_full shape: {train_full.shape}")
    lines.append(f"- test_full shape: {test_full.shape}")
    lines.append(f"- target shape: {target.shape}")
    lines.append(f"- feature count: {len(feature_cols)}")
    lines.append(f"- categorical features: {len(cat_cols)}")
    lines.append(f"- numerical features: {len(num_cols)}\n")

    lines.append("## Missingness")
    lines.append(f"- mean missing rate (train): {miss_train.mean():.4f}")
    lines.append(f"- mean missing rate (test): {miss_test.mean():.4f}")
    lines.append("- top-20 missing features (train):")
    for name, val in miss_train.head(20).items():
        lines.append(f"  - {name}: {val:.4f}")

    lines.append("\n## Target prevalence")
    lines.append(f"- mean positive rate across targets: {pos_rate.mean():.4f}")
    lines.append("- top-10 most frequent targets:")
    for name, val in pos_rate.head(10).items():
        lines.append(f"  - {name}: {val:.4f}")
    lines.append("- top-10 rarest targets:")
    for name, val in pos_rate.tail(10).items():
        lines.append(f"  - {name}: {val:.4f}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA report saved to: {report_path}")


def apply_feature_hygiene(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    feature_cols = [c for c in train_df.columns if c != "customer_id"]
    drop_cols = set()

    if DROP_CONST_FEATURES:
        const_cols = [c for c in feature_cols if train_df[c].nunique(dropna=False) <= 1]
        drop_cols.update(const_cols)
    else:
        const_cols = []

    missing_rate = train_df[feature_cols].isna().mean()
    ultra_missing = missing_rate[missing_rate > MISSING_RATE_THRESHOLD].index.tolist()
    drop_cols.update(ultra_missing)

    keep_cols = [c for c in train_df.columns if c not in drop_cols]
    train_df = train_df[keep_cols].copy()
    test_df = test_df[keep_cols].copy()

    stats = {
        "dropped_total": len(drop_cols),
        "dropped_const": len(const_cols),
        "dropped_ultra_missing": len(ultra_missing),
    }
    print(f"Feature hygiene stats: {stats}")
    return train_df, test_df, stats


def preprocess_cats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[str]]:
    feature_cols = [c for c in train_df.columns if c != "customer_id"]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]

    # Fill NA BEFORE astype(str) to avoid literal "nan" artifacts.
    for col in cat_cols:
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

    cat_indices = [feature_cols.index(c) for c in cat_cols]
    return train_df, test_df, cat_indices, feature_cols


def train_multilabel_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_indices: List[int],
    seeds: List[int],
    n_folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_targets = y_train.shape[1]
    test_pred = np.zeros((x_test.shape[0], n_targets), dtype=np.float64)
    oof_pred = np.zeros((x_train.shape[0], n_targets), dtype=np.float64)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2026)
    total_models = len(seeds) * n_folds
    model_counter = 0

    for seed in seeds:
        for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), start=1):
            model_counter += 1
            print(f"[Multi] seed={seed} fold={fold}/{n_folds} model={model_counter}/{total_models}")

            x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            train_pool = Pool(x_tr, label=y_tr, cat_features=cat_indices)
            valid_pool = Pool(x_va, label=y_va, cat_features=cat_indices)
            test_pool = Pool(x_test, cat_features=cat_indices)

            model = CatBoostClassifier(
                loss_function="MultiLogloss",
                eval_metric="MultiLogloss",
                iterations=8000,
                learning_rate=0.028,
                depth=8,
                l2_leaf_reg=9.0,
                random_strength=1.2,
                bagging_temperature=0.8,
                border_count=254,
                bootstrap_type="Bayesian",
                leaf_estimation_iterations=5,
                od_type="Iter",
                od_wait=350,
                random_seed=seed,
                task_type="GPU",
                verbose=300,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            oof_pred[va_idx] += model.predict(valid_pool, prediction_type="RawFormulaVal") / max(len(seeds), 1)
            test_pred += model.predict(test_pool, prediction_type="RawFormulaVal")

    test_pred /= float(total_models)
    return oof_pred, test_pred


def train_ovr_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_indices: List[int],
    seeds: List[int],
    n_folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_targets = y_train.shape[1]
    test_pred = np.zeros((x_test.shape[0], n_targets), dtype=np.float64)
    oof_pred = np.zeros((x_train.shape[0], n_targets), dtype=np.float64)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2027)

    for target_idx, target_name in enumerate(y_train.columns):
        print(f"[OvR] target {target_idx + 1}/{n_targets}: {target_name}")
        pred_acc_test = np.zeros(x_test.shape[0], dtype=np.float64)
        y_col = y_train[target_name].values

        models_count = 0
        for seed in seeds:
            for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), start=1):
                models_count += 1
                print(f"  seed={seed} fold={fold}/{n_folds}")

                x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
                y_tr, y_va = y_col[tr_idx], y_col[va_idx]

                train_pool = Pool(x_tr, label=y_tr, cat_features=cat_indices)
                valid_pool = Pool(x_va, label=y_va, cat_features=cat_indices)
                test_pool = Pool(x_test, cat_features=cat_indices)

                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    iterations=5000,
                    learning_rate=0.03,
                    depth=7,
                    l2_leaf_reg=8.0,
                    random_strength=1.0,
                    bagging_temperature=0.7,
                    border_count=254,
                    bootstrap_type="Bayesian",
                    leaf_estimation_iterations=5,
                    od_type="Iter",
                    od_wait=300,
                    auto_class_weights="Balanced",
                    random_seed=seed,
                    task_type="GPU",
                    verbose=300,
                )
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

                oof_pred[va_idx, target_idx] += (
                    model.predict(valid_pool, prediction_type="RawFormulaVal").reshape(-1) / max(len(seeds), 1)
                )
                pred_acc_test += model.predict(test_pool, prediction_type="RawFormulaVal").reshape(-1)

        test_pred[:, target_idx] = pred_acc_test / float(models_count)

    return oof_pred, test_pred


def find_best_blend_weight(y_true: pd.DataFrame, pred_multi_raw: np.ndarray, pred_ovr_raw: np.ndarray) -> Tuple[float, float]:
    y_arr = y_true.values
    weights = np.linspace(0.0, 1.0, 21)
    best_w, best_auc = 0.5, -1.0

    p_multi = sigmoid(pred_multi_raw)
    p_ovr = sigmoid(pred_ovr_raw)

    for w in weights:
        p_blend = w * p_multi + (1.0 - w) * p_ovr
        auc = roc_auc_score(y_arr, p_blend, average="macro")
        print(f"Blend weight={w:.2f} | OOF macro AUC={auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)

    return best_w, best_auc


def build_submission(sample_submit: pd.DataFrame, preds: np.ndarray, out_path: Path) -> None:
    sub = sample_submit.copy()
    sub.iloc[:, 1:] = preds
    sub["customer_id"] = sub["customer_id"].astype("int32")
    sub.to_parquet(out_path, index=False)
    print(f"Saved submission: {out_path} | shape={sub.shape}")


def main() -> None:
    print("CONFIG:")
    print(f"  DATA_DIR={DATA_DIR}")
    print(f"  OUTPUT_PATH={OUTPUT_PATH}")
    print(f"  MULTI_FOLDS={MULTI_FOLDS}, OVR_FOLDS={OVR_FOLDS}")
    print(f"  MULTI_SEEDS={MULTI_SEEDS}, OVR_SEEDS={OVR_SEEDS}")
    print(f"  AUTO_TUNE_BLEND_WEIGHT={AUTO_TUNE_BLEND_WEIGHT}")

    data_dir = resolve_data_dir()
    print(f"Using data dir: {data_dir}")

    print("Loading data...")
    train_main, train_extra, test_main, test_extra, target, sample_submit = load_data(data_dir)

    print("Merging features...")
    train_full = merge_features(train_main, train_extra)
    test_full = merge_features(test_main, test_extra)

    print("Running EDA...")
    run_eda(train_full, test_full, target, EDA_REPORT_PATH)

    print("Applying feature hygiene...")
    train_full, test_full, _ = apply_feature_hygiene(train_full, test_full)

    print("Preprocessing categorical features...")
    train_full, test_full, cat_indices, feature_cols = preprocess_cats(train_full, test_full)

    x_train = train_full[feature_cols]
    x_test = test_full[feature_cols]
    y_train = target.drop(columns=["customer_id"])

    print("Training MultiLogloss ensemble...")
    oof_multi_raw, test_multi_raw = train_multilabel_ensemble(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        cat_indices=cat_indices,
        seeds=MULTI_SEEDS,
        n_folds=MULTI_FOLDS,
    )

    print("Training OvR ensemble...")
    oof_ovr_raw, test_ovr_raw = train_ovr_ensemble(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        cat_indices=cat_indices,
        seeds=OVR_SEEDS,
        n_folds=OVR_FOLDS,
    )

    if AUTO_TUNE_BLEND_WEIGHT:
        best_w, best_auc = find_best_blend_weight(y_train, oof_multi_raw, oof_ovr_raw)
        blend_w = best_w
        print(f"Best OOF blend weight: {blend_w:.3f} | macro AUC={best_auc:.6f}")
    else:
        blend_w = DEFAULT_BLEND_WEIGHT_MULTI
        print(f"Using default blend weight: {blend_w:.3f}")

    p_multi_test = sigmoid(test_multi_raw)
    p_ovr_test = sigmoid(test_ovr_raw)
    p_blend_test = blend_w * p_multi_test + (1.0 - blend_w) * p_ovr_test

    # Keep submission format consistent with previous solution (raw margins).
    pred_final = logit(p_blend_test)
    build_submission(sample_submit, pred_final, OUTPUT_PATH)


if __name__ == "__main__":
    main()
