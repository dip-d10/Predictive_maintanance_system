import json
import os
import sys
from importlib import import_module

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from src.logger import logging

try:
	XGBClassifier = import_module("xgboost").XGBClassifier
except Exception:
	XGBClassifier = None


class ModelTrainer:
	def __init__(self, config: ModelTrainerConfig):
		self.config = config

	def _load_data(self) -> pd.DataFrame:
		if not os.path.exists(self.config.training_data_path):
			raise FileNotFoundError(f"Training data not found at {self.config.training_data_path}")

		df = pd.read_csv(self.config.training_data_path)
		if self.config.datetime_column in df.columns:
			df[self.config.datetime_column] = pd.to_datetime(df[self.config.datetime_column])
		return df

	def _resolve_target_column(self, df: pd.DataFrame) -> str:
		if self.config.target_column in df.columns:
			return self.config.target_column

		fallback_columns = [
			column
			for column in df.columns
			if column.startswith("failure_within_next_") or column.startswith("target_failure_")
		]
		if fallback_columns:
			logging.warning(
				f"Configured target '{self.config.target_column}' not found. Using '{fallback_columns[0]}' instead."
			)
			return fallback_columns[0]

		raise KeyError(f"Target column '{self.config.target_column}' not found in training data")

	def _chronological_split(self, df: pd.DataFrame, target_column: str):
		working_df = df.copy()

		if self.config.datetime_column in working_df.columns:
			working_df = working_df.sort_values(self.config.datetime_column)
		else:
			working_df = working_df.sort_index()

		if self.config.max_training_rows and len(working_df) > self.config.max_training_rows:
			logging.info(
				f"Limiting training data to the first {self.config.max_training_rows} chronological rows "
				f"out of {len(working_df)} total rows for the dry-run."
			)
			working_df = working_df.iloc[: self.config.max_training_rows].copy()

		working_df = working_df.reset_index(drop=True)
		split_index = int(len(working_df) * (1 - self.config.test_size))
		split_index = max(1, min(split_index, len(working_df) - 1))

		train_df = working_df.iloc[:split_index].copy().reset_index(drop=True)
		test_df = working_df.iloc[split_index:].copy().reset_index(drop=True)

		train_y = train_df[target_column].astype(int)
		test_y = test_df[target_column].astype(int)

		train_x = train_df.drop(columns=[target_column])
		test_x = test_df.drop(columns=[target_column])

		drop_columns = [self.config.datetime_column, "machineID"]
		train_x = train_x.drop(columns=[column for column in drop_columns if column in train_x.columns])
		test_x = test_x.drop(columns=[column for column in drop_columns if column in test_x.columns])

		train_x = pd.get_dummies(train_x, drop_first=False)
		test_x = pd.get_dummies(test_x, drop_first=False)
		test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
		train_x = train_x.loc[:, ~train_x.columns.duplicated()].copy()
		test_x = test_x.loc[:, ~test_x.columns.duplicated()].copy()

		train_df.to_csv(self.config.train_data_path, index=False)
		test_df.to_csv(self.config.test_data_path, index=False)

		return train_x, test_x, train_y, test_y

	def _build_candidate_models(self):
		model_params = self.config.model_params or {}

		models = {
			"logistic_regression": LogisticRegression(
				**model_params.get("logistic_regression", {}),
			),
			"random_forest": RandomForestClassifier(
				random_state=self.config.random_state,
				**model_params.get("random_forest", {}),
			),
		}

		if XGBClassifier is not None:
			xgb_params = dict(model_params.get("xgboost", {}))
			xgb_params.setdefault("random_state", self.config.random_state)
			xgb_params.setdefault("objective", "binary:logistic")
			xgb_params.setdefault("eval_metric", "logloss")
			models["xgboost"] = XGBClassifier(**xgb_params)
		else:
			logging.warning("xgboost is not available in the current environment; skipping the XGBoost candidate.")

		return models

	def _evaluate_model(self, y_true, y_pred, y_probability=None):
		metrics = {
			"accuracy": accuracy_score(y_true, y_pred),
			"precision": precision_score(y_true, y_pred, zero_division=0),
			"recall": recall_score(y_true, y_pred, zero_division=0),
			"f1": f1_score(y_true, y_pred, zero_division=0),
		}

		if y_probability is not None:
			try:
				metrics["roc_auc"] = roc_auc_score(y_true, y_probability)
			except Exception:
				metrics["roc_auc"] = None
		else:
			metrics["roc_auc"] = None

		return metrics

	def initiate_model_trainer(self):
		try:
			logging.info("Starting model training...")

			os.makedirs(self.config.root_dir, exist_ok=True)
			os.makedirs(self.config.trained_model_dir, exist_ok=True)

			df = self._load_data()
			target_column = self._resolve_target_column(df)
			train_x, test_x, train_y, test_y = self._chronological_split(df, target_column)

			candidate_models = self._build_candidate_models()
			metrics_report = {}
			

			for model_name, model in candidate_models.items():
				logging.info(f"Training candidate model: {model_name}")
				model.fit(train_x, train_y)

				predictions = model.predict(test_x)
				probabilities = None
				if hasattr(model, "predict_proba"):
					probabilities = model.predict_proba(test_x)[:, 1]

				metrics = self._evaluate_model(test_y, predictions, probabilities)
				metrics_report[model_name] = metrics

				model_path = os.path.join(self.config.trained_model_dir, f"{model_name}.joblib")
				joblib.dump(model, model_path)
				metrics_report[model_name]["model_path"] = model_path

				# Trainer no longer chooses a best model. It only trains and saves candidates.

			metrics_payload = {
				"target_column": target_column,
				"models": metrics_report,
			}

			with open(self.config.metrics_file_path, "w", encoding="utf-8") as metrics_file:
				json.dump(metrics_payload, metrics_file, indent=4)

			logging.info("Model training completed successfully.")

			return ModelTrainerArtifact(
				trained_model_dir=self.config.trained_model_dir,
				train_data_path=self.config.train_data_path,
				test_data_path=self.config.test_data_path,
				model_metrics_path=self.config.metrics_file_path,
			)

		except Exception as e:
			logging.exception(f"Model training failed: {e}")
			raise MyException(e, sys)
