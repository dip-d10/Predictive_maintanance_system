import json
import os
import re
import sys
from datetime import datetime

from src.cloud_storage.azure_blob import AzureBlobClient
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig
from src.exception import MyException
from src.logger import logging


class ModelPusher:
	def __init__(
		self,
		config: ModelPusherConfig,
		azure_blob_client: AzureBlobClient,
		model_evaluation_artifact: ModelEvaluationArtifact,
	):
		self.config = config
		self.azure_blob_client = azure_blob_client
		self.model_evaluation_artifact = model_evaluation_artifact

	def _build_rejected_artifact(self, reason: str, current_version: str = "") -> ModelPusherArtifact:
		logging.info(f"Model deployment rejected: {reason}")
		production_uri = self.azure_blob_client.get_blob_uri("production/model.joblib")
		return ModelPusherArtifact(
			production_blob_uri=production_uri,
			archived_blob_uri="",
			deployment_status="rejected",
			deployed_model_version=current_version,
		)

	def _load_new_model_pr_auc(self) -> float:
		report_path = self.model_evaluation_artifact.evaluation_report_path
		if not os.path.exists(report_path):
			raise FileNotFoundError(f"Evaluation report file not found: {report_path}")

		with open(report_path, "r", encoding="utf-8") as report_file:
			report = json.load(report_file)

		model_name = self.model_evaluation_artifact.best_model_name
		model_entry = report.get("models", {}).get(model_name)
		if not model_entry:
			raise KeyError(f"Best model '{model_name}' not found in evaluation report")

		pr_auc = model_entry.get("pr_auc")
		if pr_auc is None:
			return 0.0
		return float(pr_auc)

	def _read_current_production_metadata(self) -> dict:
		metadata_blob = "production/metadata.json"
		local_path = self.config.metadata_local_path

		self.azure_blob_client.download_file(metadata_blob, local_path)

		try:
			with open(local_path, "r", encoding="utf-8") as metadata_file:
				return json.load(metadata_file)
		except Exception as exc:
			raise ValueError(f"Production metadata is corrupted: {exc}") from exc

	def _extract_version_number(self, version: str) -> int:
		if not version:
			return 0
		match = re.match(r"^v(\d+)$", str(version).strip())
		if not match:
			return 0
		return int(match.group(1))

	def _next_model_version(self, current_version: str) -> str:
		current_num = self._extract_version_number(current_version)
		return f"v{current_num + 1}" if current_num > 0 else "v2"

	def _first_model_version(self) -> str:
		return "v1"

	def _build_metadata(self, model_version: str, pr_auc: float) -> dict:
		return {
			"model_name": self.model_evaluation_artifact.best_model_name,
			"model_version": model_version,
			"training_timestamp": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
			"best_threshold": self.model_evaluation_artifact.best_threshold,
			"business_cost": self.model_evaluation_artifact.best_business_cost,
			"pr_auc": pr_auc,
			"approved_model": bool(self.model_evaluation_artifact.approved_model),
		}

	def _write_local_metadata(self, metadata: dict):
		os.makedirs(os.path.dirname(self.config.metadata_local_path) or ".", exist_ok=True)
		with open(self.config.metadata_local_path, "w", encoding="utf-8") as metadata_file:
			json.dump(metadata, metadata_file, indent=4)

	def _upload_production(self):
		self.azure_blob_client.upload_file(
			self.model_evaluation_artifact.best_model_path,
			"production/model.joblib",
		)
		self.azure_blob_client.upload_file(
			self.config.metadata_local_path,
			"production/metadata.json",
		)

	def _archive_current_production(self, archive_timestamp: str) -> str:
		archive_model = f"archive/{archive_timestamp}/model.joblib"
		archive_metadata = f"archive/{archive_timestamp}/metadata.json"

		self.azure_blob_client.move_blob("production/model.joblib", archive_model)
		self.azure_blob_client.move_blob("production/metadata.json", archive_metadata)

		archive_uri = self.azure_blob_client.get_blob_uri(f"archive/{archive_timestamp}/")
		logging.info(f"Archived current production model to archive/{archive_timestamp}/")
		return archive_uri

	def _cleanup_archive_retention(self, keep_latest: int = 2):
		archive_blobs = self.azure_blob_client.list_blobs(prefix="archive/")
		archive_folders = sorted({
			blob_name.split("/")[1]
			for blob_name in archive_blobs
			if blob_name.count("/") >= 2
		})

		if len(archive_folders) <= keep_latest:
			return

		folders_to_delete = archive_folders[: len(archive_folders) - keep_latest]
		for folder in folders_to_delete:
			folder_prefix = f"archive/{folder}/"
			folder_blobs = self.azure_blob_client.list_blobs(prefix=folder_prefix)
			for blob_name in folder_blobs:
				self.azure_blob_client.delete_blob(blob_name)
			logging.info(f"Removed archived model folder due to retention policy: {folder_prefix}")

	def initiate_model_pusher(self) -> ModelPusherArtifact:
		try:
			logging.info("Starting model pusher stage...")
			self.azure_blob_client.connect()

			if not self.model_evaluation_artifact.approved_model:
				return self._build_rejected_artifact("Model was not approved by evaluation stage")

			new_business_cost = float(self.model_evaluation_artifact.best_business_cost)
			new_pr_auc = self._load_new_model_pr_auc()

			prod_model_exists = self.azure_blob_client.blob_exists("production/model.joblib")
			prod_meta_exists = self.azure_blob_client.blob_exists("production/metadata.json")

			if not prod_model_exists and not prod_meta_exists:
				logging.info("No production model found. Performing first deployment.")
				metadata = self._build_metadata(self._first_model_version(), new_pr_auc)
				self._write_local_metadata(metadata)
				self._upload_production()

				production_uri = self.azure_blob_client.get_blob_uri("production/model.joblib")
				return ModelPusherArtifact(
					production_blob_uri=production_uri,
					archived_blob_uri="",
					deployment_status="deployed",
					deployed_model_version=metadata["model_version"],
				)

			if prod_model_exists != prod_meta_exists:
				raise FileNotFoundError(
					"Production blobs are in an inconsistent state: model and metadata must both exist"
				)

			current_metadata = self._read_current_production_metadata()
			current_business_cost = float(current_metadata.get("business_cost"))
			current_pr_auc = float(current_metadata.get("pr_auc", 0.0))
			current_version = str(current_metadata.get("model_version", "v1"))

			logging.info(
				"Comparing models for deployment decision: "
				f"new(cost={new_business_cost}, pr_auc={new_pr_auc}) vs "
				f"current(cost={current_business_cost}, pr_auc={current_pr_auc})"
			)

			should_deploy = (
				new_business_cost < current_business_cost
				or (new_business_cost == current_business_cost and new_pr_auc > current_pr_auc)
			)

			if not should_deploy:
				return self._build_rejected_artifact(
					"New model is not better than production model by deployment policy",
					current_version=current_version,
				)

			archive_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
			archive_uri = self._archive_current_production(archive_timestamp)

			next_version = self._next_model_version(current_version)
			metadata = self._build_metadata(next_version, new_pr_auc)
			self._write_local_metadata(metadata)
			self._upload_production()

			self._cleanup_archive_retention(keep_latest=2)
			logging.info(f"Production deployment successful with model version {next_version}")

			production_uri = self.azure_blob_client.get_blob_uri("production/model.joblib")
			return ModelPusherArtifact(
				production_blob_uri=production_uri,
				archived_blob_uri=archive_uri,
				deployment_status="deployed",
				deployed_model_version=next_version,
			)

		except Exception as exc:
			logging.exception(f"Model pusher stage failed: {exc}")
			raise MyException(exc, sys)
