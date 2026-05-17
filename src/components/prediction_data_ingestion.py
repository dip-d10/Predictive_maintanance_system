import os
import sys
import pandas as pd
from datetime import datetime

from src.entity.config_entity import PredictionDataIngestionConfig
from src.entity.artifact_entity import PredictionDataIngestionArtifact
from src.exception import MyException
from src.logger import logging


class PredictionDataIngestion:
    """
    Load prediction input dataset from CSV file (demo mode).
    
    The prediction dataset is generated during the data validation stage
    and contains a lookback window of recent telemetry data suitable for
    running hourly predictions.
    """
    
    def __init__(self, config: PredictionDataIngestionConfig):
        self.config = config
    
    def load_prediction_data(self) -> pd.DataFrame:
        """
        Load prediction input dataset from CSV file.
        
        This file is created by the data validation component during training
        and contains recent telemetry records ready for inference.
        
        Returns: DataFrame with prediction input data
        """
        try:
            if not os.path.exists(self.config.prediction_input_path):
                raise FileNotFoundError(
                    f"Prediction input dataset not found at: {self.config.prediction_input_path}\n"
                    f"Please run the training pipeline first to generate this file."
                )
            
            logging.info(f"Loading prediction input dataset from: {self.config.prediction_input_path}")
            
            df = pd.read_csv(self.config.prediction_input_path)
            
            logging.info(f"Loaded {len(df)} records from prediction dataset")
            
            # Ensure datetime column is properly formatted
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values(by=['machineID', 'datetime']).reset_index(drop=True)
                logging.info(f"Data sorted by machineID and datetime")
            
            # Log data summary
            unique_machines = df['machineID'].nunique() if 'machineID' in df.columns else 0
            logging.info(f"Prediction dataset contains {unique_machines} unique machines")
            
            return df
            
        except Exception as e:
            logging.exception(f"Error loading prediction data: {e}")
            raise MyException(e, sys)
    
    def initiate_prediction_data_ingestion(self) -> PredictionDataIngestionArtifact:
        """
        Main entry point for prediction data ingestion.
        
        Loads the prediction dataset from CSV file and returns artifact
        containing the data for downstream feature engineering.
        """
        try:
            logging.info("Starting prediction data ingestion (CSV mode)")
            
            # Load data from CSV
            telemetry_data = self.load_prediction_data()
            
            # Ensure output directory exists
            os.makedirs(self.config.prediction_data_dir, exist_ok=True)
            
            # Save a copy for traceability and debugging
            traceability_file = os.path.join(
                self.config.prediction_data_dir,
                "prediction_ingestion_trace.csv"
            )
            telemetry_data.to_csv(traceability_file, index=False)
            logging.info(f"Saved ingestion trace to {traceability_file}")
            
            # Create artifact
            artifact = PredictionDataIngestionArtifact(
                telemetry_data=telemetry_data,
                machines_data=pd.DataFrame(),  # Not needed for prediction phase
                errors_data=pd.DataFrame(),     # Not needed for prediction phase
                maintenance_data=pd.DataFrame(), # Not needed for prediction phase
                prediction_data_dir=self.config.prediction_data_dir,
                ingestion_timestamp=datetime.utcnow().isoformat(),
                records_count=len(telemetry_data)
            )
            
            logging.info("Prediction data ingestion completed successfully")
            return artifact
            
        except Exception as e:
            logging.exception(f"Error in prediction data ingestion: {e}")
            raise MyException(e, sys)
