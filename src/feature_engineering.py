"""Feature engineering utilities for the customer churn portfolio project.

This module loads the competition data, applies preprocessing, and builds the
hand-crafted feature sets used by both training pipelines.

The goal is not to expose every experiment from the Kaggle notebook. Instead,
it provides a reproducible feature layer that reflects the main modeling ideas
used in the final solution.
"""

from itertools import combinations

import numpy as np
import pandas as pd


def percentile_rank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return percentile ranks of values against a reference distribution."""
    reference_sorted = np.sort(reference)
    return (np.searchsorted(reference_sorted, values) / len(reference_sorted)).astype("float32")


def zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return z-scores of values using the mean and std of a reference array."""
    mean = np.mean(reference)
    std = np.std(reference)
    if std == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mean) / std).astype("float32")


def load_and_engineer_features(
    train_path: str,
    test_path: str,
    orig_path: str,
    target_col: str = "Churn",
):
    """Load raw data and return engineered train/test dataframes plus metadata.

    Returns
    -------
    train : pandas.DataFrame
        Training data with engineered features and numeric target.
    test : pandas.DataFrame
        Test data with engineered features.
    orig : pandas.DataFrame
        Original Telco churn reference dataset used for feature construction.
    feature_dict : dict
        Metadata describing feature groups used downstream.
    """
    print("\n[Loading data]")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_path)

    train[target_col] = train[target_col].map({"No": 0, "Yes": 1}).astype(int)
    orig[target_col] = orig[target_col].map({"No": 0, "Yes": 1}).astype(int)

    for df in (train, test, orig):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in orig.columns:
        orig = orig.drop(columns=["customerID"])

    for df in (train, test, orig):
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    top_cats_for_ngram = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
    ]
    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    print("\n[Building engineered features]")
    engineered_numeric_features: list[str] = []

    for col in numeric_features:
        frequency = pd.concat([train[col], orig[col]]).value_counts(normalize=True)
        for df in (train, test):
            df[f"FREQ_{col}"] = df[col].map(frequency).fillna(0).astype("float32")
        engineered_numeric_features.append(f"FREQ_{col}")

    for df in (train, test):
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
    engineered_numeric_features += [
        "charges_deviation",
        "monthly_to_total_ratio",
        "avg_monthly_charges",
    ]

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for df in (train, test):
        df["service_count"] = (df[service_columns] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
    engineered_numeric_features += ["service_count", "has_internet", "has_phone"]

    for col in categorical_features + numeric_features:
        mapping = orig.groupby(col)[target_col].mean()
        feature_name = f"ORIG_proba_{col}"
        train = train.merge(mapping.rename(feature_name), on=col, how="left")
        test = test.merge(mapping.rename(feature_name), on=col, how="left")
        for df in (train, test):
            df[feature_name] = df[feature_name].fillna(0.5).astype("float32")
        engineered_numeric_features.append(feature_name)

    orig_churn_total = orig.loc[orig[target_col] == 1, "TotalCharges"].values
    orig_nonchurn_total = orig.loc[orig[target_col] == 0, "TotalCharges"].values
    orig_total = orig["TotalCharges"].values
    internet_service_monthly_charge = orig.groupby("InternetService")["MonthlyCharges"].mean()

    for df in (train, test):
        total_charges = df["TotalCharges"].values
        df["pctrank_nonchurner_TC"] = percentile_rank_against(total_charges, orig_nonchurn_total)
        df["pctrank_churner_TC"] = percentile_rank_against(total_charges, orig_churn_total)
        df["pctrank_orig_TC"] = percentile_rank_against(total_charges, orig_total)
        df["zscore_churn_gap_TC"] = (
            np.abs(zscore_against(total_charges, orig_churn_total))
            - np.abs(zscore_against(total_charges, orig_nonchurn_total))
        ).astype("float32")
        df["zscore_nonchurner_TC"] = zscore_against(total_charges, orig_nonchurn_total)
        df["pctrank_churn_gap_TC"] = (
            percentile_rank_against(total_charges, orig_churn_total)
            - percentile_rank_against(total_charges, orig_nonchurn_total)
        ).astype("float32")
        df["resid_IS_MC"] = (
            df["MonthlyCharges"] - df["InternetService"].map(internet_service_monthly_charge).fillna(0)
        ).astype("float32")

        for cat_col, out_col in [("InternetService", "cond_pctrank_IS_TC"), ("Contract", "cond_pctrank_C_TC")]:
            conditional_values = np.zeros(len(df), dtype="float32")
            for category_value in orig[cat_col].unique():
                mask = df[cat_col] == category_value
                reference = orig.loc[orig[cat_col] == category_value, "TotalCharges"].values
                if len(reference) > 0 and mask.sum() > 0:
                    conditional_values[mask] = percentile_rank_against(
                        df.loc[mask, "TotalCharges"].values,
                        reference,
                    )
            df[out_col] = conditional_values

    engineered_numeric_features += [
        "pctrank_nonchurner_TC",
        "zscore_churn_gap_TC",
        "pctrank_churn_gap_TC",
        "resid_IS_MC",
        "cond_pctrank_IS_TC",
        "zscore_nonchurner_TC",
        "pctrank_orig_TC",
        "pctrank_churner_TC",
        "cond_pctrank_C_TC",
    ]

    for quantile_label, quantile_value in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        churn_quantile = np.quantile(orig_churn_total, quantile_value)
        nonchurn_quantile = np.quantile(orig_nonchurn_total, quantile_value)
        for df in (train, test):
            df[f"dist_To_ch_{quantile_label}"] = np.abs(df["TotalCharges"] - churn_quantile).astype("float32")
            df[f"dist_To_nc_{quantile_label}"] = np.abs(df["TotalCharges"] - nonchurn_quantile).astype("float32")
            df[f"qdist_gap_To_{quantile_label}"] = (
                df[f"dist_To_nc_{quantile_label}"] - df[f"dist_To_ch_{quantile_label}"]
            ).astype("float32")

    engineered_numeric_features += [
        "qdist_gap_To_q50",
        "dist_To_ch_q50",
        "dist_To_nc_q50",
        "dist_To_nc_q25",
        "qdist_gap_To_q25",
        "dist_To_nc_q75",
        "dist_To_ch_q75",
        "qdist_gap_To_q75",
    ]

    digit_features = [
        "tenure_first_digit",
        "tenure_last_digit",
        "tenure_second_digit",
        "tenure_mod10",
        "tenure_mod12",
        "tenure_num_digits",
        "tenure_is_multiple_10",
        "tenure_rounded_10",
        "tenure_dev_from_round10",
        "mc_first_digit",
        "mc_last_digit",
        "mc_second_digit",
        "mc_mod10",
        "mc_mod100",
        "mc_num_digits",
        "mc_is_multiple_10",
        "mc_is_multiple_50",
        "mc_rounded_10",
        "mc_fractional",
        "mc_dev_from_round10",
        "tc_first_digit",
        "tc_last_digit",
        "tc_second_digit",
        "tc_mod10",
        "tc_mod100",
        "tc_num_digits",
        "tc_is_multiple_10",
        "tc_is_multiple_100",
        "tc_rounded_100",
        "tc_fractional",
        "tc_dev_from_round100",
        "tenure_years",
        "tenure_months_in_year",
        "mc_per_digit",
        "tc_per_digit",
    ]

    for df in (train, test):
        tenure_str = df["tenure"].astype(str)
        df["tenure_first_digit"] = tenure_str.str[0].astype(int)
        df["tenure_last_digit"] = tenure_str.str[-1].astype(int)
        df["tenure_second_digit"] = tenure_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["tenure_mod10"] = df["tenure"] % 10
        df["tenure_mod12"] = df["tenure"] % 12
        df["tenure_num_digits"] = tenure_str.str.len()
        df["tenure_is_multiple_10"] = (df["tenure"] % 10 == 0).astype("float32")
        df["tenure_rounded_10"] = np.round(df["tenure"] / 10) * 10
        df["tenure_dev_from_round10"] = np.abs(df["tenure"] - df["tenure_rounded_10"])

        monthly_charge_str = df["MonthlyCharges"].astype(str).str.replace(".", "", regex=False)
        df["mc_first_digit"] = monthly_charge_str.str[0].astype(int)
        df["mc_last_digit"] = monthly_charge_str.str[-1].astype(int)
        df["mc_second_digit"] = monthly_charge_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["mc_mod10"] = np.floor(df["MonthlyCharges"]) % 10
        df["mc_mod100"] = np.floor(df["MonthlyCharges"]) % 100
        df["mc_num_digits"] = np.floor(df["MonthlyCharges"]).astype(int).astype(str).str.len()
        df["mc_is_multiple_10"] = (np.floor(df["MonthlyCharges"]) % 10 == 0).astype("float32")
        df["mc_is_multiple_50"] = (np.floor(df["MonthlyCharges"]) % 50 == 0).astype("float32")
        df["mc_rounded_10"] = np.round(df["MonthlyCharges"] / 10) * 10
        df["mc_fractional"] = df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])
        df["mc_dev_from_round10"] = np.abs(df["MonthlyCharges"] - df["mc_rounded_10"])

        total_charge_str = df["TotalCharges"].astype(str).str.replace(".", "", regex=False)
        df["tc_first_digit"] = total_charge_str.str[0].astype(int)
        df["tc_last_digit"] = total_charge_str.str[-1].astype(int)
        df["tc_second_digit"] = total_charge_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["tc_mod10"] = np.floor(df["TotalCharges"]) % 10
        df["tc_mod100"] = np.floor(df["TotalCharges"]) % 100
        df["tc_num_digits"] = np.floor(df["TotalCharges"]).astype(int).astype(str).str.len()
        df["tc_is_multiple_10"] = (np.floor(df["TotalCharges"]) % 10 == 0).astype("float32")
        df["tc_is_multiple_100"] = (np.floor(df["TotalCharges"]) % 100 == 0).astype("float32")
        df["tc_rounded_100"] = np.round(df["TotalCharges"] / 100) * 100
        df["tc_fractional"] = df["TotalCharges"] - np.floor(df["TotalCharges"])
        df["tc_dev_from_round100"] = np.abs(df["TotalCharges"] - df["tc_rounded_100"])

        df["tenure_years"] = df["tenure"] // 12
        df["tenure_months_in_year"] = df["tenure"] % 12
        df["mc_per_digit"] = df["MonthlyCharges"] / (df["mc_num_digits"] + 0.001)
        df["tc_per_digit"] = df["TotalCharges"] / (df["tc_num_digits"] + 0.001)

        for feature in digit_features:
            df[feature] = df[feature].astype("float32")
    engineered_numeric_features += digit_features

    bigram_columns = []
    trigram_columns = []
    for first_col, second_col in combinations(top_cats_for_ngram, 2):
        col_name = f"BG_{first_col}_{second_col}"
        for df in (train, test):
            df[col_name] = df[first_col].astype(str) + "_" + df[second_col].astype(str)
        bigram_columns.append(col_name)

    top_four_for_trigrams = top_cats_for_ngram[:4]
    for first_col, second_col, third_col in combinations(top_four_for_trigrams, 3):
        col_name = f"TG_{first_col}_{second_col}_{third_col}"
        for df in (train, test):
            df[col_name] = (
                df[first_col].astype(str) + "_" + df[second_col].astype(str) + "_" + df[third_col].astype(str)
            )
        trigram_columns.append(col_name)

    ngram_columns = bigram_columns + trigram_columns

    numeric_as_categorical = [f"CAT_{col}" for col in numeric_features]
    for encoded_col, raw_col in zip(numeric_as_categorical, numeric_features):
        for df in (train, test):
            df[encoded_col] = df[raw_col].astype(str)

    target_encoding_columns = numeric_as_categorical + categorical_features

    print("[Feature engineering complete]")
    feature_dict = {
        "NUMS": numeric_features,
        "NEW_NUMS": engineered_numeric_features,
        "CATS": categorical_features,
        "TE_COLUMNS": target_encoding_columns,
        "NGRAM_COLS": ngram_columns,
        "NUM_AS_CAT": numeric_as_categorical,
    }
    return train, test, orig, feature_dict
