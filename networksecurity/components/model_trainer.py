import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from xgboost import XGBClassifier  

import mlflow
import dagshub

dagshub.init(repo_owner='atulkanttiwari', repo_name='End-to-End-Phishing-Detection-System', mlflow=True)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):

        #  Models including XGBoost
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(max_iter=1000, verbose=1),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(
                eval_metric='logloss',
                verbosity=1
            ),
        }

        #  Hyperparameters including XGBoost
        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20],
            },
            "Random Forest": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01],
                'n_estimators': [100, 200],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [0.1, 0.01],
                'n_estimators': [100, 200],
            },
            "XGBoost": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.01],
                'subsample': [0.8, 1.0],
            }
        }

        #  Evaluate all models
        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params
        )

        best_model_score = max(model_report.values())

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        logging.info(f"Best model selected: {best_model_name}")

        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        classification_train_metric = get_classification_score(
            y_true=y_train,
            y_pred=y_train_pred
        )

        classification_test_metric = get_classification_score(
            y_true=y_test,
            y_pred=y_test_pred
        )

        #  MLflow Tracking
        with mlflow.start_run():

            mlflow.log_param("best_model", best_model_name)

            mlflow.log_metric("train_f1_score", classification_train_metric.f1_score)
            mlflow.log_metric("train_precision", classification_train_metric.precision_score)
            mlflow.log_metric("train_recall", classification_train_metric.recall_score)

            mlflow.log_metric("test_f1_score", classification_test_metric.f1_score)
            mlflow.log_metric("test_precision", classification_test_metric.precision_score)
            mlflow.log_metric("test_recall", classification_test_metric.recall_score)

            mlflow.sklearn.log_model(best_model, "model")

        # Load preprocessor
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        # Create model directory
        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        # Wrap model + preprocessor
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model
        )

        # Save wrapped model
        save_object(
            self.model_trainer_config.trained_model_file_path,
            obj=network_model
        )

        # Optional final_model save
        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")

        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(
                X_train, y_train, X_test, y_test
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
