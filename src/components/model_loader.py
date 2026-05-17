import os
import sys
import json
import joblib
from datetime import datetime

from src.entity.config_entity import ModelLoaderConfig
from src.entity.artifact_entity import ModelLoaderArtifact
from src.exception import MyException
from src.logger import logging
from src.cloud_storage.azure_blob import AzureBlobClient
from src.configuration.config_manager import ConfigurationManager


class ModelLoader:
    """
    Load production model from Azure Blob Storage.
    Fetch model, metadata, and threshold for inference.
    """
    
    def __init__(self, config: ModelLoaderConfig):
        self.config = config
        config_manager = ConfigurationManager()
        azure_config = config_manager.get_azure_blob_config()
        self.azure_client = AzureBlobClient(config=azure_config)
    
    def _download_model_from_blob(self) -> str:
        """Download model.joblib from Azure Blob"""
        try:
            logging.info(f"Downloading production model from Azure Blob: {self.config.production_model_blob_path}")
            
            os.makedirs(self.config.model_cache_dir, exist_ok=True)
            
            local_model_path = os.path.join(self.config.model_cache_dir, "production_model.joblib")
            
            # Download from Azure Blob
            self.azure_client.download_file(
                blob_name=self.config.production_model_blob_path,
                local_path=local_model_path
            )
            
            logging.info(f"Model downloaded to: {local_model_path}")
            return local_model_path
            
        except Exception as e:
            logging.exception(f"Failed to download model from Azure Blob: {e}")
            raise MyException(e, sys)
    
    def _download_metadata_from_blob(self) -> dict:
        """Download model metadata from Azure Blob"""
        try:
            logging.info(f"Downloading model metadata from Azure Blob: {self.config.production_metadata_blob_path}")
            
            os.makedirs(self.config.model_cache_dir, exist_ok=True)
            
            local_metadata_path = os.path.join(self.config.model_cache_dir, "production_metadata.json")
            
            # Download from Azure Blob
            self.azure_client.download_file(
                blob_name=self.config.production_metadata_blob_path,
                local_path=local_metadata_path
            )
            
            # Load metadata
            with open(local_metadata_path, "r") as f:
                metadata = json.load(f)
            
            logging.info(f"Metadata loaded. Model version: {metadata.get('model_version')}")
            return metadata
            
        except Exception as e:
            logging.warning(f"Could not download metadata from Azure Blob: {e}")
            # Return default metadata if not found
            return {
                "model_version": "unknown",
                "threshold": self.config.default_threshold,
                "training_date": "unknown"
            }
    
    def load_model(self) -> object:
        """Load model from local cache or download if not exists"""
        try:
            logging.info("Loading production model for inference")
            
            # Check if model exists locally
            local_model_path = os.path.join(self.config.model_cache_dir, "production_model.joblib")
            
            if os.path.exists(local_model_path):
                logging.info(f"Loading model from local cache: {local_model_path}")
                model = joblib.load(local_model_path)
            else:
                logging.info("Model not in cache, downloading from Azure Blob")
                model_path = self._download_model_from_blob()
                model = joblib.load(model_path)
            
            logging.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logging.exception(f"Failed to load model: {e}")
            raise MyException(e, sys)
    
    def initiate_model_loading(self) -> ModelLoaderArtifact:
        """
        Main entry point for model loading.
        Load model, metadata, and threshold.
        """
        try:
            logging.info("Starting model loading for prediction")
            
            # Load model
            model = self.load_model()
            
            # Load metadata
            metadata = self._download_metadata_from_blob()
            
            # Extract threshold from metadata or use default
            threshold = metadata.get("threshold", self.config.default_threshold)
            model_version = metadata.get("model_version", "unknown")
            
            os.makedirs(self.config.model_cache_dir, exist_ok=True)
            
            artifact = ModelLoaderArtifact(
                model=model,
                model_path=os.path.join(self.config.model_cache_dir, "production_model.joblib"),
                metadata=metadata,
                threshold=threshold,
                model_version=model_version,
                load_timestamp=datetime.utcnow().isoformat(),
                is_load_successful=True,
                message="Model loaded successfully"
            )
            
            logging.info(f"Model loading completed. Version: {model_version}, Threshold: {threshold}")
            return artifact
            
        except Exception as e:
            logging.exception(f"Error in model loading: {e}")
            raise MyException(e, sys)
