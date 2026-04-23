"""Blend saved model predictions using out-of-fold performance."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN_PATH = "data/train.csv"
SAMPLE_SUB_PATH = "data/sample_submission.csv"
TARGET_COL = "Churn"
XGB_OOF_PATH = "output/oof_v24_xgb.npy"
XGB_TEST_PATH = "output/pred_v24_xgb.npy"
REALMLP_OOF_PATH = "output/oof_v26_realmlp.npy"
REALMLP_TEST_PATH = "output/pred_v26_realmlp.npy"


def main() -> None:
    """Search simple ensemble weights on OOF predictions and save submission."""
    print("\n[Loading predictions for ensemble]")

    train = pd.read_csv(TRAIN_PATH)
    y_true = train[TARGET_COL].map({"No": 0, "Yes": 1}).values

    required_files = [XGB_OOF_PATH, XGB_TEST_PATH, REALMLP_OOF_PATH, REALMLP_TEST_PATH]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}. Run train_twostage.py and train_realmlp.py first."
            )

    oof_xgb = np.load(XGB_OOF_PATH)
    pred_xgb = np.load(XGB_TEST_PATH)
    oof_realmlp = np.load(REALMLP_OOF_PATH)
    pred_realmlp = np.load(REALMLP_TEST_PATH)

    print(f"XGBoost OOF AUC   : {roc_auc_score(y_true, oof_xgb):.6f}")
    print(f"RealMLP OOF AUC   : {roc_auc_score(y_true, oof_realmlp):.6f}")
    print(f"OOF correlation   : {np.corrcoef(oof_xgb, oof_realmlp)[0, 1]:.6f}")

    print("\n[Searching ensemble weights on OOF predictions]")
    candidate_weights = [round(weight, 2) for weight in np.arange(0.50, 1.01, 0.05)]
    best_auc = -1.0
    best_weight = None
    best_test_pred = None

    for weight in candidate_weights:
        oof_blend = weight * oof_xgb + (1 - weight) * oof_realmlp
        auc = roc_auc_score(y_true, oof_blend)
        print(f"  XGB={weight:.2f}, RealMLP={1 - weight:.2f} -> OOF AUC={auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_weight = weight
            best_test_pred = weight * pred_xgb + (1 - weight) * pred_realmlp

    print("\n" + "=" * 40)
    print("BEST BLEND FOUND")
    print("=" * 40)
    print(f"XGBoost weight : {best_weight:.2f}")
    print(f"RealMLP weight : {1 - best_weight:.2f}")
    print(f"Best OOF AUC   : {best_auc:.6f}")

    submission = pd.read_csv(SAMPLE_SUB_PATH)
    submission[TARGET_COL] = best_test_pred
    output_path = f"output/submission_ensemble_xgb{int(best_weight * 100)}_mlp{int((1 - best_weight) * 100)}.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSaved final Kaggle submission to {output_path}")


if __name__ == "__main__":
    main()
