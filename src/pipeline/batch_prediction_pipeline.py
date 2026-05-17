from datetime import datetime

from src.configuration.config_manager import ConfigurationManager
from src.components.prediction_data_ingestion import PredictionDataIngestion
from src.components.prediction_feature_engineering import PredictionFeatureEngineering
from src.components.model_loader import ModelLoader
from src.components.predictor import Predictor
from src.components.prediction_storage import PredictionStorage
from src.components.alert_manager import AlertManager
from src.logger import logging


class BatchPredictionPipeline:
    """
    Hourly batch prediction pipeline:
    
    Fetch latest telemetry
    → Engineer features
    → Load production model
    → Generate predictions
    → Store predictions
    → Trigger alerts
    
    Runs every hour automatically.
    """
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.start_time = datetime.utcnow()
    
    def start_prediction_data_ingestion(self):
        """Fetch latest hourly telemetry and historical data"""
        try:
            logging.info("=" * 80)
            logging.info("BATCH PREDICTION PIPELINE - Step 1: Data Ingestion")
            logging.info("=" * 80)
            
            data_ingestion_config = self.config_manager.get_prediction_data_ingestion_config()
            data_ingestion = PredictionDataIngestion(config=data_ingestion_config)
            artifact = data_ingestion.initiate_prediction_data_ingestion()
            
            logging.info(f"[OK] Data Ingestion Complete: {artifact.records_count} records fetched")
            return artifact
            
        except Exception as e:
            logging.exception(f"[ERROR] Data Ingestion Failed: {e}")
            raise
    
    def start_prediction_feature_engineering(self, data_ingestion_artifact):
        """Engineer features from raw telemetry"""
        try:
            logging.info("=" * 80)
            logging.info("BATCH PREDICTION PIPELINE - Step 2: Feature Engineering")
            logging.info("=" * 80)
            
            feature_engineering_config = self.config_manager.get_prediction_feature_engineering_config()
            feature_engineering = PredictionFeatureEngineering(config=feature_engineering_config)
            
            artifact = feature_engineering.initiate_prediction_feature_engineering(
                telemetry_data=data_ingestion_artifact.telemetry_data,
                machines_data=data_ingestion_artifact.machines_data,
                errors_data=data_ingestion_artifact.errors_data,
                maintenance_data=data_ingestion_artifact.maintenance_data
            )
            
            logging.info(f"[OK] Feature Engineering Complete: {artifact.engineered_features_count} features created")
            return artifact
            
        except Exception as e:
            logging.exception(f"[ERROR] Feature Engineering Failed: {e}")
            raise
    
    def start_model_loading(self):
        """Load production model from Azure Blob"""
        try:
            logging.info("=" * 80)
            logging.info("BATCH PREDICTION PIPELINE - Step 3: Model Loading")
            logging.info("=" * 80)
            
            model_loader_config = self.config_manager.get_model_loader_config()
            model_loader = ModelLoader(config=model_loader_config)
            artifact = model_loader.initiate_model_loading()
            
            logging.info(f"[OK] Model Loading Complete: Version {artifact.model_version}, Threshold: {artifact.threshold}")
            return artifact
            
        except Exception as e:
            logging.exception(f"[ERROR] Model Loading Failed: {e}")
            raise
    
    def start_prediction(self, feature_engineering_artifact, model_loader_artifact):
        """Generate predictions using loaded model"""
        try:
            logging.info("=" * 80)
            logging.info("BATCH PREDICTION PIPELINE - Step 4: Prediction")
            logging.info("=" * 80)
            
            predictor_config = self.config_manager.get_predictor_config()
            predictor = Predictor(config=predictor_config)
            
            artifact = predictor.initiate_prediction(
                model=model_loader_artifact.model,
                features_df=feature_engineering_artifact.features_dataframe,
                threshold=model_loader_artifact.threshold,
                model_version=model_loader_artifact.model_version
            )
            
            logging.info(f"[OK] Prediction Complete: {artifact.predictions_count} predictions, {artifact.maintenance_alerts_count} alerts")
            return artifact
            
        except Exception as e:
            logging.exception(f"[ERROR] Prediction Failed: {e}")
            raise
    
    def start_prediction_storage(self, prediction_artifact):
        """Store predictions into MongoDB"""
        try:
            logging.info("=" * 80)
            logging.info("BATCH PREDICTION PIPELINE - Step 5: Prediction Storage")
            logging.info("=" * 80)
            
            storage_config = self.config_manager.get_prediction_storage_config()
            storage = PredictionStorage(config=storage_config)
            
            artifact = storage.initiate_prediction_storage(
                predictions_df=prediction_artifact.predictions_dataframe,
                predictions_count=prediction_artifact.predictions_count,
                maintenance_alerts_count=prediction_artifact.maintenance_alerts_count,
                model_version=prediction_artifact.model_version
            )
            
            logging.info(f"[OK] Storage Complete: {artifact.predictions_stored_count} predictions stored")
            return artifact
            
        except Exception as e:
            logging.exception(f"[ERROR] Storage Failed: {e}")
            raise
    
    def start_alert_manager(self, prediction_artifact):
        """Trigger alerts for maintenance requirements"""
        try:
            logging.info("=" * 80)
            logging.info("BATCH PREDICTION PIPELINE - Step 6: Alert Management")
            logging.info("=" * 80)
            
            alert_config = self.config_manager.get_alert_config()
            alert_manager = AlertManager(config=alert_config)
            
            artifact = alert_manager.initiate_alert_manager(
                predictions_df=prediction_artifact.predictions_dataframe
            )
            
            logging.info(f"[OK] Alert Management Complete: {artifact.alerts_triggered} alerts triggered")
            logging.info(f"  - HIGH risk: {artifact.high_risk_alerts}")
            logging.info(f"  - MEDIUM risk: {artifact.medium_risk_alerts}")
            
            return artifact
            
        except Exception as e:
            logging.exception(f"[ERROR] Alert Management Failed: {e}")
            raise
    # Note: orchestration removed — stage methods remain for external orchestration
