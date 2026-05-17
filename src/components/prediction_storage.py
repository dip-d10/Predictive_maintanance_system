import os
import sys
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

from src.entity.config_entity import PredictionStorageConfig
from src.entity.artifact_entity import PredictionStorageArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.env_loader import MONGO_URI


class PredictionStorage:
    """
    Store prediction results into MongoDB predictions collection.
    Save prediction metadata and risk indicators.
    """
    
    def __init__(self, config: PredictionStorageConfig):
        self.config = config
        self.mongo_uri = MONGO_URI
    
    def _connect_mongodb(self):
        """Connect to MongoDB Atlas"""
        try:
            logging.info("Connecting to MongoDB for storing predictions")
            client = MongoClient(self.mongo_uri)
            database = client[self.config.database_name]
            logging.info("MongoDB connection successful")
            return database
        except Exception as e:
            logging.exception(f"MongoDB connection failed: {e}")
            raise MyException(e, sys)
    
    def store_predictions(
        self,
        predictions_df: pd.DataFrame,
        model_version: str
    ) -> int:
        """
        Store predictions into MongoDB predictions collection
        """
        try:
            logging.info("Storing predictions into MongoDB")
            
            if predictions_df.empty:
                logging.warning("No predictions to store")
                return 0
            
            database = self._connect_mongodb()
            predictions_collection = database[self.config.predictions_collection]
            
            # Prepare documents for insertion
            documents = []
            for _, row in predictions_df.iterrows():
                doc = {
                    "machine_id": str(row['machineID']),
                    "timestamp": str(row['datetime']),
                    "failure_probability": float(row['failure_probability']),
                    "risk_level": str(row['risk_level']),
                    "maintenance_required": bool(row['maintenance_required']),
                    "model_version": model_version,
                    "stored_at": datetime.utcnow().isoformat(),
                    "prediction_batch_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                }
                documents.append(doc)
            
            # Bulk insert
            if documents:
                result = predictions_collection.insert_many(documents)
                inserted_count = len(result.inserted_ids)
                logging.info(f"Inserted {inserted_count} prediction records into MongoDB")
                return inserted_count
            
            return 0
            
        except Exception as e:
            logging.exception(f"Error storing predictions: {e}")
            raise MyException(e, sys)
    
    def store_prediction_summary(
        self,
        predictions_count: int,
        maintenance_alerts_count: int,
        model_version: str
    ) -> dict:
        """
        Store prediction batch summary metadata
        """
        try:
            logging.info("Storing prediction batch summary")
            
            database = self._connect_mongodb()
            summary_collection = database[self.config.prediction_summary_collection]
            
            # Ensure values are native Python types (avoid numpy types causing BSON errors)
            total_preds = int(predictions_count)
            maintenance_alerts = int(maintenance_alerts_count)
            alert_rate = float((maintenance_alerts / total_preds * 100) if total_preds > 0 else 0.0)

            summary_doc = {
                "batch_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                "total_predictions": total_preds,
                "maintenance_alerts": maintenance_alerts,
                "alert_rate": alert_rate,
                "model_version": str(model_version),
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "batch_status": "completed"
            }

            result = summary_collection.insert_one(summary_doc)
            logging.info(f"Stored prediction summary. Batch ID: {summary_doc['batch_id']}")
            
            return summary_doc
            
        except Exception as e:
            logging.exception(f"Error storing prediction summary: {e}")
            # Don't fail the entire pipeline if summary storage fails
            logging.warning("Continuing despite summary storage error")
            return {}
    
    def initiate_prediction_storage(
        self,
        predictions_df: pd.DataFrame,
        predictions_count: int,
        maintenance_alerts_count: int,
        model_version: str
    ) -> PredictionStorageArtifact:
        """
        Main entry point for prediction storage
        """
        try:
            logging.info("Starting prediction storage")
            
            # Store individual predictions
            inserted_count = self.store_predictions(predictions_df, model_version)
            
            # Store batch summary
            summary = self.store_prediction_summary(
                predictions_count,
                maintenance_alerts_count,
                model_version
            )
            
            artifact = PredictionStorageArtifact(
                predictions_stored_count=inserted_count,
                maintenance_alerts_stored=maintenance_alerts_count,
                batch_id=summary.get('batch_id', 'unknown'),
                storage_timestamp=datetime.utcnow().isoformat(),
                is_storage_successful=True,
                message=f"Stored {inserted_count} predictions successfully"
            )
            
            logging.info("Prediction storage completed successfully")
            return artifact
            
        except Exception as e:
            logging.exception(f"Error in prediction storage: {e}")
            raise MyException(e, sys)
