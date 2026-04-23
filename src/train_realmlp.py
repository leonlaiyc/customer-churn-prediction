"""Train the RealMLP churn model on the engineered tabular feature set."""

import gc
import os

import numpy as np
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_engineering import load_and_engineer_features

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ORIG_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"

N_SPLITS = 5
INNER_SPLITS = 5
RANDOM_STATE = 42

REALMLP_PARAMS = dict(
    random_state=RANDOM_STATE,
    verbosity=1,
    val_metric_name="1-auc_ovr",
    n_ens=8,
    n_epochs=256,
    batch_size=256,
    use_early_stopping=True,
    early_stopping_additive_patience=10,
    early_stopping_multiplicative_patience=1,
    lr=0.075,
    wd=0.0236,
    sq_mom=0.988,
    lr_sched="flat_anneal",
    first_layer_lr_factor=0.25,
    add_front_scale=False,
    embedding_size=8,
    max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128],
    act="silu",
    p_drop=0.05,
    p_drop_sched="flat_cos",
    plr_hidden_1=16,
    plr_hidden_2=8,
    plr_act_name="gelu",
    plr_lr_factor=0.1151,
    plr_sigma=2.33,
    ls_eps=0.02,
    ls_eps_sched="cos",
    tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
)


def main() -> None:
    """Train RealMLP with leakage-aware target encoding and save predictions."""
    train, test, _, feature_dict = load_and_engineer_features(
        TRAIN_PATH,
        TEST_PATH,
        ORIG_PATH,
        TARGET_COL,
    )

    categorical_features = feature_dict["CATS"]
    target_encoding_columns = feature_dict["TE_COLUMNS"]
    ngram_columns = feature_dict["NGRAM_COLS"]
    numeric_as_categorical = feature_dict["NUM_AS_CAT"]

    y_all = train[TARGET_COL].values
    target_encoding_stats = ["mean", "std", "min", "max"]
    n_train = len(train)

    oof = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)

    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("\n[Starting RealMLP training]")

    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(train, y_all), start=1):
        print(f"\n{'=' * 60}\n[FOLD {fold}/{N_SPLITS}]\n{'=' * 60}")

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

        raw_drop_cols = [col for col in ngram_columns + numeric_as_categorical if col in x_train.columns]
        for df in (x_train, x_valid, x_test):
            df.drop(columns=[col for col in raw_drop_cols if col in df.columns], inplace=True, errors="ignore")
            for col in ["id", TARGET_COL]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        numeric_feature_cols = [col for col in x_train.columns if col not in categorical_features]
        for df in (x_train, x_valid, x_test):
            df[numeric_feature_cols] = df[numeric_feature_cols].fillna(0).astype("float32")
            for col in categorical_features:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        print(f"  Training RealMLP with {x_train.shape[1]} features...")
        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(x_train, y_train, x_valid, y_valid)

        valid_pred = model.predict_proba(x_valid)[:, 1].astype(np.float32)
        oof[valid_idx] = valid_pred
        test_pred += model.predict_proba(x_test)[:, 1].astype(np.float32) / N_SPLITS
        print(f"    -> RealMLP fold AUC: {roc_auc_score(y_valid, valid_pred):.6f}")

        del model, x_train, x_valid, x_test
        gc.collect()

    print("\n" + "=" * 60)
    print("[Final OOF summary]")
    print(f"RealMLP OOF AUC : {roc_auc_score(y_all, oof):.6f}")

    os.makedirs("output", exist_ok=True)
    np.save("output/oof_v26_realmlp.npy", oof)
    np.save("output/pred_v26_realmlp.npy", test_pred)
    print("\nSaved RealMLP OOF and test predictions to output/")


if __name__ == "__main__":
    main()
