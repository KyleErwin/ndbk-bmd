import random
from abc import ABC, abstractmethod
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.base import Trials
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from mlflow.client import MlflowClient
from mlflow.entities import Run
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.training.utils import get_git_revision_short_hash, get_dvc_hash


class BaseTuner(ABC):
    """Base class for Hyperparameter tuning with Hyperopt and MLflow."""

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
        self.space = {}

    @abstractmethod
    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        """
        Create a pipeline with the given parameters.
        """
        pass

    def objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.trial_count += 1
        run_name = f"{self.model_name}_trial_{self.trial_count}"

        with mlflow.start_run(
            run_name=run_name, experiment_id=self.experiment_id, nested=True
        ) as run:
            pipeline = self.create_pipeline(params)

            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train, cv=self.cv, scoring="roc_auc"
            )
            cv_mean = cv_scores.mean()

            pipeline.fit(self.X_train, self.y_train)
            y_probs = pipeline.predict_proba(self.X_test)[:, 1]
            test_auc = roc_auc_score(self.y_test, y_probs)

            mlflow.log_params(params)
            mlflow.log_metric("cv_auc_mean", cv_mean)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("overfit_gap", cv_mean - test_auc)
            mlflow.set_tag("trial_index", self.trial_count)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=self.model_name,
                serialization_format="skops",
                pip_requirements=["scikit-learn", "skops"],
                skops_trusted_types=[
                    "imblearn.over_sampling._smote.base.SMOTE",
                    "imblearn.pipeline.Pipeline",
                    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
                    "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
                    "scipy.sparse._csr.csr_matrix",
                ],
            )

        return {"loss": 1 - cv_mean, "status": STATUS_OK, "run_id": run.info.run_id}

    def tune(self, max_evals: int = 20, register_best_model: bool = True) -> None:
        with mlflow.start_run(run_name=f"{self.experiment_name}_parent_run"):
            trials = Trials()
            best_params = fmin(
                fn=self.objective,
                space=self.space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
            )

            self.trials = trials
            best_trial = sorted(trials.results, key=lambda x: x["loss"])[0]
            best_run_id = best_trial["run_id"]
            best_cv_auc = 1 - best_trial["loss"]

            if register_best_model:
                self.register_best_model(best_run_id, best_cv_auc)

            print(f"Experiment {self.experiment_name} Completed")
            print(f"Model {self.model_name}")
            print(f"Best Run ID: {best_run_id}")
            print(f"Best Parameters Found: {best_params}")
            print(f"Best CV AUC: {best_cv_auc}")

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
        self.space = {
            "n_neighbors": hp.choice("n_neighbors", range(3, 8)),  # Test 3 to 8
            "weights": hp.choice("weights", ["uniform", "distance"]),
            "metric": hp.choice("metric", ["euclidean", "manhattan", "minkowski"]),
            "p": hp.uniform("p", 1, 2),  # 1 is Manhattan, 2 is Euclidean
        }

    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                # ("one_hot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=params["n_neighbors"],
                        weights=params["weights"],
                        metric=params["metric"],
                        p=params.get("p", 2),
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
        self.space = {
            "hidden_layer_sizes": hp.choice(
                "hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50)]
            ),
            "activation": hp.choice("activation", ["tanh", "relu"]),
            "solver": hp.choice("solver", ["sgd", "adam"]),
            "alpha": hp.loguniform("alpha", np.log(0.0001), np.log(0.05)),
            "learning_rate": hp.choice("learning_rate", ["constant", "adaptive"]),
        }

    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                # ("one_hot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=params["hidden_layer_sizes"],
                        activation=params["activation"],
                        solver=params["solver"],
                        alpha=params["alpha"],
                        learning_rate=params["learning_rate"],
                        max_iter=500,
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
        self.space = {
            "n_estimators": hp.choice("n_estimators", [50, 100, 200]),
            "max_depth": hp.choice("max_depth", [None, 10, 20, 30]),
            "min_samples_split": hp.choice("min_samples_split", [2, 5, 10]),
            "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 4]),
        }

    def create_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                # ("one_hot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("smote", SMOTE(random_state=42)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        random_state=42,
                    ),
                ),
            ]
        )
