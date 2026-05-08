import json
import os
import sys
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	average_precision_score,
	confusion_matrix,
)

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.exception import MyException
from src.logger import logging


class ModelEvaluation:
	def __init__(self, config: ModelEvaluationConfig):
		self.config = config

	def _read_df(self, path: str) -> pd.DataFrame:
		if not os.path.exists(path):
			raise FileNotFoundError(f"Data file not found: {path}")
		df = pd.read_csv(path)
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

		raise KeyError(f"Target column '{self.config.target_column}' not found in data")

	def _preprocess(self, df: pd.DataFrame, reference_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
		target_col = self._resolve_target_column(df)
		y = df[target_col].astype(int)
		x = df.drop(columns=[target_col])

		drop_columns = [self.config.datetime_column, "machineID"]
		x = x.drop(columns=[c for c in drop_columns if c in x.columns])

		x = pd.get_dummies(x, drop_first=False)

		# Remove duplicate columns generated during get_dummies before any reindexing
		x = x.loc[:, ~x.columns.duplicated()].copy()

		if reference_columns is not None:
			# ensure reference_columns are unique to avoid pandas reindex errors
			unique_ref_cols = list(dict.fromkeys(reference_columns))
			x = x.reindex(columns=unique_ref_cols, fill_value=0)

		return x, y

	def _load_models(self) -> Dict[str, Any]:
		models = {}
		if not os.path.exists(self.config.trained_model_dir):
			raise FileNotFoundError(f"Trained models directory not found: {self.config.trained_model_dir}")

		for fname in os.listdir(self.config.trained_model_dir):
			if fname.endswith(".joblib"):
				path = os.path.join(self.config.trained_model_dir, fname)
				try:
					model = joblib.load(path)
					models[fname.rsplit(".", 1)[0]] = {"model": model, "path": path}
				except Exception:
					logging.exception(f"Failed to load model file: {path}")
		return models

	def _get_probabilities(self, model, X: pd.DataFrame) -> np.ndarray:
		if hasattr(model, "predict_proba"):
			try:
				probs = model.predict_proba(X)[:, 1]
				return probs
			except Exception:
				pass

		if hasattr(model, "decision_function"):
			try:
				scores = model.decision_function(X)
				probs = 1 / (1 + np.exp(-scores))
				return probs
			except Exception:
				pass

		# Fallback: use predictions as probabilities (0 or 1)
		preds = model.predict(X)
		return preds.astype(float)

	def _evaluate_at_threshold(self, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
		preds = (probs >= threshold).astype(int)
		precision = float(precision_score(y_true, preds, zero_division=0))
		recall = float(recall_score(y_true, preds, zero_division=0))
		f1 = float(f1_score(y_true, preds, zero_division=0))
		tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

		return {
			"threshold": threshold,
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"false_positives": int(fp),
			"false_negatives": int(fn),
		}

	def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
		try:
			logging.info("Starting model evaluation...")

			# Load reference train and test datasets for consistent encoding
			train_df = self._read_df(self.config.train_data_path)
			test_df = self._read_df(self.config.test_data_path)

			# derive reference feature columns from training data
			train_x, train_y = self._preprocess(train_df, reference_columns=None)
			reference_columns = list(train_x.columns)

			test_x, test_y = self._preprocess(test_df, reference_columns=reference_columns)

			models = self._load_models()
			if not models:
				raise FileNotFoundError("No candidate models found for evaluation")

			thresholds = [0.1 * i for i in range(1, 10)]

			evaluation_report: Dict[str, Any] = {
				"models": {},
			}
			threshold_report: Dict[str, Any] = {}

			# For selection across models
			model_selection_summaries = []

			for model_name, info in models.items():
				model = info["model"]
				logging.info(f"Evaluating model: {model_name}")

				probs = self._get_probabilities(model, test_x)

				roc_auc = None
				pr_auc = None
				try:
					roc_auc = float(roc_auc_score(test_y, probs))
				except Exception:
					roc_auc = None
				try:
					pr_auc = float(average_precision_score(test_y, probs))
				except Exception:
					pr_auc = None

				# Evaluate thresholds for business cost
				thresholds_info = []
				best_for_model = None
				best_cost = float("inf")

				for t in thresholds:
					metrics = self._evaluate_at_threshold(test_y.to_numpy(), probs, t)
					fp = metrics["false_positives"]
					fn = metrics["false_negatives"]
					business_cost = (
						fn * self.config.downtime_cost + fp * self.config.maintenance_cost
					)
					metrics["business_cost"] = float(business_cost)
					thresholds_info.append(metrics)

					if business_cost < best_cost:
						best_cost = business_cost
						best_for_model = metrics

				evaluation_report["models"][model_name] = {
					"roc_auc": roc_auc,
					"pr_auc": pr_auc,
					"best_threshold": best_for_model,
					"model_path": info.get("path"),
				}

				threshold_report[model_name] = thresholds_info

				model_selection_summaries.append(
					{
						"model_name": model_name,
						"best_business_cost": float(best_cost),
						"pr_auc": pr_auc if pr_auc is not None else 0.0,
						"recall_at_best_threshold": float(best_for_model.get("recall", 0.0)),
						"best_threshold": float(best_for_model.get("threshold", 0.5)),
					}
				)

			# Choose best model: primary lowest business cost, secondary highest PR-AUC, tertiary highest recall
			model_selection_summaries = sorted(
				model_selection_summaries,
				key=lambda x: (x["best_business_cost"], -x["pr_auc"], -x["recall_at_best_threshold"]),
			)

			chosen = model_selection_summaries[0]
			best_model_name = chosen["model_name"]
			best_threshold = chosen["best_threshold"]
			best_business_cost = chosen["best_business_cost"]

			approved = False
			chosen_pr_auc = next((m["pr_auc"] for m in model_selection_summaries if m["model_name"] == best_model_name), 0.0)
			if chosen_pr_auc >= self.config.min_pr_auc_threshold:
				approved = True

			# Save reports
			os.makedirs(os.path.dirname(self.config.evaluation_report_path) or ".", exist_ok=True)
			with open(self.config.evaluation_report_path, "w", encoding="utf-8") as f:
				json.dump(evaluation_report, f, indent=4)

			os.makedirs(os.path.dirname(self.config.threshold_report_path) or ".", exist_ok=True)
			with open(self.config.threshold_report_path, "w", encoding="utf-8") as f:
				json.dump(threshold_report, f, indent=4)

			# Save best model artifact
			best_model_obj = models[best_model_name]["model"]
			os.makedirs(os.path.dirname(self.config.best_model_path) or ".", exist_ok=True)
			joblib.dump(best_model_obj, self.config.best_model_path)

			logging.info(f"Model evaluation completed. Best model: {best_model_name}, approved: {approved}")

			return ModelEvaluationArtifact(
				best_model_path=self.config.best_model_path,
				threshold_path=self.config.threshold_report_path,
				evaluation_report_path=self.config.evaluation_report_path,
				approved_model=approved,
				best_model_name=best_model_name,
				best_threshold=best_threshold,
				best_business_cost=best_business_cost,
			)

		except Exception as e:
			logging.exception(f"Model evaluation failed: {e}")
			raise MyException(e, sys)
