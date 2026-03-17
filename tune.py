import argparse
import random

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.training import KNNTuner, NNTuner, RandomForestTuner

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

mlflow.set_tracking_uri("https://mlflow-server-production-c6e7.up.railway.app/")


def main(args):
    df = pd.read_csv("data/bank_marketing_data_processed.csv")

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

    kwargs = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "cv": cv,
    }

    if args.model in ["knn", "all"]:
        knn_tuner = KNNTuner(experiment_name="knn_bmd", **kwargs)  # ty: ignore[invalid-argument-type]
        knn_tuner.tune(max_evals=20, register_best_model=True)

    if args.model in ["nn", "all"]:
        nn_tuner = NNTuner(experiment_name="nn_bmd", **kwargs)  # ty: ignore[invalid-argument-type]
        nn_tuner.tune(max_evals=20, register_best_model=True)

    if args.model in ["rf", "all"]:
        rf_tuner = RandomForestTuner(experiment_name="rf_bmd", **kwargs)  # ty: ignore[invalid-argument-type]
        rf_tuner.tune(max_evals=20, register_best_model=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["knn", "nn", "rf", "all"],
        help="Model to tune: knn, nn, rf, or all",
    )
    args = parser.parse_args()
    main(args)
