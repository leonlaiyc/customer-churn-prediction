"""Train the two-stage Ridge -> XGBoost churn model.

Stage 1 fits a linear model on engineered numeric features plus one-hot encoded
categoricals. Stage 2 adds the Ridge prediction as a feature and lets XGBoost
model the remaining non-linear signal.
"""

import gc
import os

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from feature_engineering import load_and_engineer_features

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ORIG_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"

N_SPLITS = 5
INNER_SPLITS = 5
RANDOM_STATE = 42
EARLY_STOP_ROUNDS = 500
RIDGE_ALPHA = 10.0

XGB_PARAMS = {
    "n_estimators": 50000,
    "learning_rate": 0.0063,
    "max_depth": 5,
    "subsample": 0.81,
    "colsample_bytree": 0.32,
    "min_child_weight": 6,
    "reg_alpha": 3.5017,
    "reg_lambda": 1.2925,
    "gamma": 0.790,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": EARLY_STOP_ROUNDS,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "enable_categorical": True,
    "device": "cuda",
    "verbosity": 0,
}


def main() -> None:
    """Train the two-stage model and save OOF/test predictions."""
    train, test, _, feature_dict = load_and_engineer_features(
        TRAIN_PATH,
        TEST_PATH,
        ORIG_PATH,
        TARGET_COL,
    )

    numeric_features = feature_dict["NUMS"]
    engineered_numeric_features = feature_dict["NEW_NUMS"]
    categorical_features = feature_dict["CATS"]
    target_encoding_columns = feature_dict["TE_COLUMNS"]
    ngram_columns = feature_dict["NGRAM_COLS"]
    numeric_as_categorical = feature_dict["NUM_AS_CAT"]

    y_all = train[TARGET_COL].values
    target_encoding_stats = ["mean", "std", "min", "max"]
    n_train = len(train)

    ridge_oof = np.zeros(n_train, dtype=np.float32)
    xgb_oof = np.zeros(n_train, dtype=np.float32)
    ridge_test_pred = np.zeros(len(test), dtype=np.float32)
    xgb_test_pred = np.zeros(len(test), dtype=np.float32)

    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("\n[Starting two-stage training: Ridge -> XGBoost]")

    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(train, y_all), start=1):
        print(f"\n{'=' * 60}\n[FOLD {fold}/{N_SPLITS}] train={len(train_idx):,} valid={len(valid_idx):,}\n{'=' * 60}")

        x_train = train.iloc[train_idx].reset_index(drop=True).copy()
        y_train = y_all[train_idx]
        x_valid = train.iloc[valid_idx].reset_index(drop=True).copy()
        y_valid = y_all[valid_idx]
        x_test = test.copy()

        x_train[TARGET_COL] = y_train
        for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
            x_train_inner = x_train.iloc[inner_train_idx]
            for col in target_encoding_columns:
                stats = x_train_inner.groupby(col, observed=False)[TARGET_COL].agg(target_encoding_stats)
                stats.columns = [f"TE_{col}_{stat}" for stat in target_encoding_stats]
                merged = x_train.iloc[inner_valid_idx][[col]].merge(stats, on=col, how="left")
                for stat in target_encoding_stats:
                    feature_name = f"TE_{col}_{stat}"
                    x_train.loc[x_train.index[inner_valid_idx], feature_name] = merged[feature_name].values

        for col in target_encoding_columns:
            stats = x_train.groupby(col, observed=False)[TARGET_COL].agg(target_encoding_stats)
            stats.columns = [f"TE_{col}_{stat}" for stat in target_encoding_stats]
            for feature_name in stats.columns:
                x_valid[feature_name] = x_valid[[col]].merge(stats[[feature_name]], on=col, how="left")[feature_name].values
                x_test[feature_name] = x_test[[col]].merge(stats[[feature_name]], on=col, how="left")[feature_name].values
                x_train[feature_name] = x_train[feature_name].fillna(0.5)
                x_valid[feature_name] = x_valid[feature_name].fillna(0.5).astype("float32")
                x_test[feature_name] = x_test[feature_name].fillna(0.5).astype("float32")

        for col in ngram_columns:
            ngram_mean = x_train.groupby(col, observed=False)[TARGET_COL].mean()
            feature_name = f"TE_ng_{col}"
            x_train[feature_name] = x_train[col].map(ngram_mean).fillna(0.5).astype("float32")
            x_valid[feature_name] = x_valid[col].map(ngram_mean).fillna(0.5).astype("float32")
            x_test[feature_name] = x_test[col].map(ngram_mean).fillna(0.5).astype("float32")

        x_train = x_train.drop(columns=[TARGET_COL])
        te_feature_cols = [col for col in x_train.columns if col.startswith("TE_")]

        print("  [Stage 1] Training Ridge...")
        ridge_numeric_cols = numeric_features + engineered_numeric_features + te_feature_cols

        scaler = StandardScaler()
        x_train_ridge_num = scaler.fit_transform(x_train[ridge_numeric_cols].fillna(0))
        x_valid_ridge_num = scaler.transform(x_valid[ridge_numeric_cols].fillna(0))
        x_test_ridge_num = scaler.transform(x_test[ridge_numeric_cols].fillna(0))

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        x_train_ridge_cat = encoder.fit_transform(x_train[categorical_features].astype(str))
        x_valid_ridge_cat = encoder.transform(x_valid[categorical_features].astype(str))
        x_test_ridge_cat = encoder.transform(x_test[categorical_features].astype(str))

        x_train_ridge = sparse.hstack([x_train_ridge_num, x_train_ridge_cat]).tocsr()
        x_valid_ridge = sparse.hstack([x_valid_ridge_num, x_valid_ridge_cat]).tocsr()
        x_test_ridge = sparse.hstack([x_test_ridge_num, x_test_ridge_cat]).tocsr()

        ridge_model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
        ridge_model.fit(x_train_ridge, y_train)

        ridge_train_pred = np.clip(ridge_model.predict(x_train_ridge), 0, 1)
        ridge_valid_pred = np.clip(ridge_model.predict(x_valid_ridge), 0, 1)
        ridge_test_fold_pred = np.clip(ridge_model.predict(x_test_ridge), 0, 1)

        ridge_oof[valid_idx] = ridge_valid_pred
        ridge_test_pred += ridge_test_fold_pred / N_SPLITS
        print(f"    -> Ridge fold AUC: {roc_auc_score(y_valid, ridge_valid_pred):.6f}")

        print("  [Stage 2] Training XGBoost...")
        x_train["ridge_pred"] = ridge_train_pred.astype("float32")
        x_valid["ridge_pred"] = ridge_valid_pred.astype("float32")
        x_test["ridge_pred"] = ridge_test_fold_pred.astype("float32")

        for df in (x_train, x_valid, x_test):
            for col in categorical_features:
                df[col] = df[col].astype("category")

        raw_drop_cols = [col for col in ngram_columns + numeric_as_categorical if col in x_train.columns]
        for df in (x_train, x_valid, x_test):
            df.drop(columns=[col for col in raw_drop_cols if col in df.columns], inplace=True, errors="ignore")

        feature_cols = [col for col in x_train.columns if col not in ["id", TARGET_COL]]

        xgb_model = XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(
            x_train[feature_cols],
            y_train,
            eval_set=[(x_valid[feature_cols], y_valid)],
            verbose=False,
        )

        xgb_valid_pred = xgb_model.predict_proba(x_valid[feature_cols])[:, 1].astype(np.float32)
        xgb_oof[valid_idx] = xgb_valid_pred
        xgb_test_pred += xgb_model.predict_proba(x_test[feature_cols])[:, 1].astype(np.float32) / N_SPLITS
        print(f"    -> XGBoost fold AUC: {roc_auc_score(y_valid, xgb_valid_pred):.6f}")

        del ridge_model, xgb_model, x_train, x_valid, x_test, x_train_ridge, x_valid_ridge, x_test_ridge
        gc.collect()

    print("\n" + "=" * 60)
    print("[Final OOF summary]")
    print(f"Ridge OOF AUC : {roc_auc_score(y_all, ridge_oof):.6f}")
    print(f"XGB   OOF AUC : {roc_auc_score(y_all, xgb_oof):.6f}")

    os.makedirs("output", exist_ok=True)
    np.save("output/oof_v24_xgb.npy", xgb_oof)
    np.save("output/pred_v24_xgb.npy", xgb_test_pred)
    print("\nSaved XGBoost OOF and test predictions to output/")


if __name__ == "__main__":
    main()
