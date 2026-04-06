import argparse
import os
import random

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.training import KNNTuner, NNTuner, RandomForestTuner, XGBTuner

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_URL") or "")


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

    if args.model in ["knn"]:
        tuner = KNNTuner

    if args.model in ["nn"]:
        tuner = NNTuner

    if args.model in ["rf"]:
        tuner = RandomForestTuner

    if args.model in ["xgb"]:
        tuner = XGBTuner

    experiment_name = "BMD_Tuning"
    tuner = tuner(experiment_name=experiment_name, **kwargs)  # ty: ignore[invalid-argument-type]
    tuner.tune(n_trials=args.n_trials, register_best_model=args.register_best_model)
    tuner.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["knn", "nn", "rf", "xgb"],
        help="Model to tune: knn, nn, rf, xgb",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of trials for hyperparameter tuning",
    )
    parser.add_argument(
        "--register_best_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to register the best model after tuning",
    )
    args = parser.parse_args()
    main(args)
