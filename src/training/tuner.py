from abc import ABC, abstractmethod
from typing import Any, Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.client import MlflowClient
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

from src.training.utils import get_dvc_hash, get_git_revision_short_hash

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]
            ),
            make_column_selector(dtype_include=np.number),
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ohe",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            make_column_selector(dtype_exclude=np.number),
        ),
    ]
)


class BaseTuner(ABC):
    """Base class for Hyperparameter tuning with Optuna and MLflow."""

    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: StratifiedKFold,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv

        self.client = MlflowClient(
            "https://mlflow-server-production-c6e7.up.railway.app/"
        )
        exp = mlflow.set_experiment(experiment_name)
        self.experiment_id = exp.experiment_id
        self.trial_count = 0

    @abstractmethod
    def get_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters using optuna trial.
        """
        pass

    @abstractmethod
    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        """
        Create a pipeline with the given parameters.
        """
        pass

    def objective(self, trial: optuna.Trial) -> float:
        self.trial_count += 1
        run_name = f"{self.model_name}_trial_{self.trial_count}"

        with mlflow.start_run(
            run_name=run_name, experiment_id=self.experiment_id, nested=True
        ) as run:
            params = self.get_params(trial)
            pipeline = self.create_pipeline(params)

            cv_scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=self.cv,
                scoring="average_precision",
            )
            cv_mean = cv_scores.mean()

            pipeline.fit(self.X_train, self.y_train)
            y_probs = pipeline.predict_proba(self.X_test)[:, 1]
            test_auc = roc_auc_score(self.y_test, y_probs)

            mlflow.log_params(params)
            mlflow.log_metric("cv_average_precision", cv_mean)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.set_tag("trial_index", self.trial_count)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=self.model_name,
                serialization_format="skops",
                pip_requirements=["scikit-learn", "skops"],
                skops_trusted_types=[
                    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
                    "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
                    "scipy.sparse._csr.csr_matrix",
                    "numpy.dtype",
                    "numpy.number",
                    "sklearn.compose._column_transformer.make_column_selector",
                    "xgboost.sklearn.XGBClassifier",
                    "xgboost.core.Booster",
                ],
            )

            # Save the run ID to the trial so we can retrieve it for the best model later
            trial.set_user_attr("run_id", run.info.run_id)

        # Optuna maximizes by default if we specify direction="maximize"
        return cv_mean

    def tune(self, n_trials: int = 30, register_best_model: bool = True) -> None:
        with mlflow.start_run(run_name=f"{self.experiment_name}_parent_run"):
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=n_trials)

            best_trial = study.best_trial
            best_run_id = best_trial.user_attrs["run_id"]
            best_cv_ap = best_trial.value or 0.0

            self.best_params = study.best_params

            if register_best_model:
                self.register_best_model(best_run_id, best_cv_ap)

            print(f"Experiment {self.experiment_name} Completed")
            print(f"Model {self.model_name}")
            print(f"Best Run ID: {best_run_id}")
            print(f"Best Parameters Found: {best_trial.params}")
            print(f"Best CV Average Precision: {best_cv_ap}")

    def register_best_model(self, best_run_id: str, cv_auc: float) -> None:
        """Registers the best model from the trials into MLflow Model Registry."""

        model_uri = f"runs:/{best_run_id}/{self.model_name}"
        mv = mlflow.register_model(model_uri, self.model_name)

        # Hardcoded file path for now
        gov_tags = {
            "git_sha": get_git_revision_short_hash(),
            "data_hash": get_dvc_hash("data/bank_marketing_data_processed.csv.dvc"),
            "team": "kyles-team",
            "cv_auc_score": str(round(cv_auc, 4)),
        }

        for key, value in gov_tags.items():
            self.client.set_model_version_tag(
                name=self.model_name, version=mv.version, key=key, value=value
            )

        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=f"{self.model_name}_{self.experiment_name}_tuned",
            version=mv.version,
        )

    def report(self):
        final_model = self.create_pipeline(self.best_params)

        final_model.fit(self.X_train, self.y_train)
        y_pred = final_model.predict(self.X_test)
        y_prob = final_model.predict_proba(self.X_test)[:, 1]

        target_names = ["No (0)", "Yes (1)"]

        print("--- Test Set Results ---")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_prob):.4f}")
        print(
            f"Average Precision (PR-AUC): {average_precision_score(self.y_test, y_prob):.4f}"
        )
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        custom_threshold = 0.3
        y_pred_custom = (
            final_model.predict_proba(self.X_test)[:, 1] >= custom_threshold
        ).astype(int)
        print(f"\nClassification Report ({custom_threshold} Threshold):")
        print(
            classification_report(self.y_test, y_pred_custom, target_names=target_names)
        )

        y_random = np.random.random_sample(y_pred.shape).round().astype(int)

        print("\nClassification Report (Random):")
        print(classification_report(self.y_test, y_random, target_names=target_names))


class KNNTuner(BaseTuner):
    def __init__(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: StratifiedKFold,
        **kwargs,
    ) -> None:
        super().__init__("KNN", experiment_name, X_train, y_train, X_test, y_test, cv)

    def get_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "minkowski"]
            ),
            "p": trial.suggest_float("p", 1.0, 2.0),
            "leaf_size": trial.suggest_categorical("leaf_size", [10, 20, 30, 40, 50]),
        }

    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=params["n_neighbors"],
                        weights=params["weights"],
                        metric=params["metric"],
                        p=params.get("p", 2),
                        leaf_size=params["leaf_size"],
                    ),
                ),
            ]
        )


class NNTuner(BaseTuner):
    def __init__(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: StratifiedKFold,
        **kwargs,
    ) -> None:
        super().__init__(
            "NeuralNetwork", experiment_name, X_train, y_train, X_test, y_test, cv
        )

    def get_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                ["8", "16_8", "8_16_8"],
            ),
            "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
            "solver": trial.suggest_categorical("solver", ["sgd", "adam"]),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.05, log=True),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "adaptive"]
            ),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 0.0001, 0.01, log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }

    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:

        hls_tuple = tuple(int(x) for x in params["hidden_layer_sizes"].split("_"))

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=hls_tuple,
                        activation=params["activation"],
                        solver=params["solver"],
                        alpha=params["alpha"],
                        learning_rate=params["learning_rate"],
                        learning_rate_init=params["learning_rate_init"],
                        batch_size=params["batch_size"],
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )


class RandomForestTuner(BaseTuner):
    def __init__(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: StratifiedKFold,
        **kwargs,
    ) -> None:
        super().__init__(
            "RandomForest", experiment_name, X_train, y_train, X_test, y_test, cv
        )

    def get_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "weight_multiplier": trial.suggest_float("weight_multiplier", 1.0, 8.0),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight={0: 1, 1: params["weight_multiplier"]},
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )


class XGBTuner(BaseTuner):
    def __init__(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: StratifiedKFold,
        **kwargs,
    ) -> None:
        super().__init__(
            "XGBoost", experiment_name, X_train, y_train, X_test, y_test, cv
        )

    def get_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }

    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        gamma=params["gamma"],
                        min_child_weight=params["min_child_weight"],
                        scale_pos_weight=params["scale_pos_weight"],
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
