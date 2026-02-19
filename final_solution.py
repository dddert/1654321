#!/usr/bin/env python3
"""
High-quality baseline+ensemble for Data Fusion Contest 2026 Task 2.
Kaggle-friendly version (no argparse).

How to use on Kaggle:
1) Set CONFIG paths and hyperparameters below.
2) Run: python final_solution.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold


# =========================
# CONFIG (edit on Kaggle)
# =========================
DATA_DIR = Path("/kaggle/input/data-fusion-2026-task-2/data")
OUTPUT_PATH = Path("submission.parquet")

MULTI_FOLDS = 5
OVR_FOLDS = 5
MULTI_SEEDS = [42, 1337]
OVR_SEEDS = [2026]
BLEND_WEIGHT_MULTI = 0.65


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


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


def preprocess_cats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[str]]:
    feature_cols = [c for c in train_df.columns if c != "customer_id"]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]

    for col in cat_cols:
        train_df[col] = train_df[col].astype(str).fillna("__MISSING__")
        test_df[col] = test_df[col].astype(str).fillna("__MISSING__")

    cat_indices = [feature_cols.index(c) for c in cat_cols]
    return train_df, test_df, cat_indices, feature_cols


def train_multilabel_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_indices: List[int],
    seeds: List[int],
    n_folds: int,
) -> np.ndarray:
    n_targets = y_train.shape[1]
    test_pred = np.zeros((x_test.shape[0], n_targets), dtype=np.float64)

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
                iterations=7000,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=8.0,
                random_strength=1.2,
                bagging_temperature=0.7,
                border_count=254,
                grow_policy="SymmetricTree",
                bootstrap_type="Bayesian",
                leaf_estimation_iterations=5,
                od_type="Iter",
                od_wait=300,
                random_seed=seed,
                task_type="GPU",
                verbose=200,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            test_pred += model.predict(test_pool, prediction_type="RawFormulaVal")

    test_pred /= float(total_models)
    return test_pred


def train_ovr_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_indices: List[int],
    seeds: List[int],
    n_folds: int,
) -> np.ndarray:
    n_targets = y_train.shape[1]
    test_pred = np.zeros((x_test.shape[0], n_targets), dtype=np.float64)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2027)

    for target_idx, target_name in enumerate(y_train.columns):
        print(f"[OvR] target {target_idx + 1}/{n_targets}: {target_name}")
        pred_acc = np.zeros(x_test.shape[0], dtype=np.float64)
        models_count = 0

        y_col = y_train[target_name].values

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
                    iterations=4500,
                    learning_rate=0.035,
                    depth=7,
                    l2_leaf_reg=7.0,
                    random_strength=1.0,
                    bagging_temperature=0.6,
                    border_count=254,
                    grow_policy="SymmetricTree",
                    bootstrap_type="Bayesian",
                    leaf_estimation_iterations=5,
                    od_type="Iter",
                    od_wait=250,
                    auto_class_weights="Balanced",
                    random_seed=seed,
                    task_type="GPU",
                    verbose=200,
                )
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
                pred_acc += model.predict(test_pool, prediction_type="RawFormulaVal").reshape(-1)

        test_pred[:, target_idx] = pred_acc / float(models_count)

    return test_pred


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
    print(f"  BLEND_WEIGHT_MULTI={BLEND_WEIGHT_MULTI}")

    print("Loading data...")
    train_main, train_extra, test_main, test_extra, target, sample_submit = load_data(DATA_DIR)

    print("Merging features...")
    train_full = merge_features(train_main, train_extra)
    test_full = merge_features(test_main, test_extra)

    print("Preprocessing categorical features...")
    train_full, test_full, cat_indices, feature_cols = preprocess_cats(train_full, test_full)

    x_train = train_full[feature_cols]
    x_test = test_full[feature_cols]
    y_train = target.drop(columns=["customer_id"])

    print("Training MultiLogloss ensemble...")
    pred_multi = train_multilabel_ensemble(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        cat_indices=cat_indices,
        seeds=MULTI_SEEDS,
        n_folds=MULTI_FOLDS,
    )

    print("Training OvR ensemble...")
    pred_ovr = train_ovr_ensemble(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        cat_indices=cat_indices,
        seeds=OVR_SEEDS,
        n_folds=OVR_FOLDS,
    )

    print(f"Blending predictions with weight_multi={BLEND_WEIGHT_MULTI:.3f}")

    p_multi = sigmoid(pred_multi)
    p_ovr = sigmoid(pred_ovr)
    p_blend = BLEND_WEIGHT_MULTI * p_multi + (1.0 - BLEND_WEIGHT_MULTI) * p_ovr
    pred_final = logit(p_blend)

    build_submission(sample_submit, pred_final, OUTPUT_PATH)


if __name__ == "__main__":
    main()
