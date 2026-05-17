import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from src.entity.config_entity import PredictorConfig
from src.entity.artifact_entity import PredictionArtifact
from src.exception import MyException
from src.logger import logging


class Predictor:
    """
    Run model inference on engineered features.
    Generate predictions, probabilities, and risk levels.
    """
    
    def __init__(self, config: PredictorConfig):
        self.config = config
    
    def _get_feature_columns(self, features_df: pd.DataFrame) -> list:
        """
        Get feature columns used for prediction.
        Exclude non-feature columns (IDs, timestamps, etc.)
        """
        exclude_cols = {
            'machineID', 'machine_id', 'datetime', 'time',
            '_id', 'id', 'timestamp', 'failure', 'model', 'model_type'
        }

        # Prefer numeric columns only to avoid passing object/string columns to models
        numeric_cols = set(features_df.select_dtypes(include=[np.number]).columns.tolist())

        feature_cols = [col for col in features_df.columns if col in numeric_cols and col not in exclude_cols]
        return feature_cols
    
    def run_inference(self, model, features_df: pd.DataFrame, feature_columns: list) -> np.ndarray:
        """Run model inference and get probabilities"""
        try:
            logging.info(f"Running inference on {len(features_df)} records")
            
            # Determine expected feature order from model if available
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                logging.info(f"Model provides feature names ({len(expected_features)}). Reindexing features to match model.")
                # Reindex the dataframe to the expected features, filling missing with 0
                X_df = features_df.reindex(columns=expected_features, fill_value=0).copy()
            else:
                X_df = features_df[feature_columns].copy()

            # Ensure numeric types; coerce any non-numeric values to NaN then fill
            for col in X_df.columns:
                if not np.issubdtype(X_df[col].dtype, np.number):
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

            # Fill NaNs produced by coercion with 0 (or consider median/impute in future)
            nan_cols = X_df.columns[X_df.isna().any()].tolist()
            if nan_cols:
                logging.warning(f"Non-numeric values coerced to NaN in columns: {nan_cols}; filling with 0")
            X_df = X_df.fillna(0)

            X = X_df.values
            
            # Get predictions
            predictions = model.predict(X)
            
            # Get prediction probabilities (for binary classification)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]  # Probability of positive class
            else:
                # Fallback: use raw predictions
                probabilities = predictions
            
            logging.info(f"Inference completed. Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
            return probabilities
            
        except Exception as e:
            logging.exception(f"Error during inference: {e}")
            raise MyException(e, sys)
    
    def _calculate_risk_level(self, failure_probability: float) -> str:
        """Convert failure probability to risk level"""
        if failure_probability >= self.config.high_risk_threshold:
            return "HIGH"
        elif failure_probability >= self.config.medium_risk_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_predictions(
        self,
        model,
        features_df: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """
        Generate predictions with risk levels and maintenance flags
        """
        try:
            logging.info("Generating predictions with risk levels")
            
            # Get feature columns
            feature_columns = self._get_feature_columns(features_df)
            
            # Run inference
            failure_probabilities = self.run_inference(model, features_df, feature_columns)
            
            # Create predictions dataframe
            predictions_df = features_df[['machineID', 'datetime']].copy()
            predictions_df['failure_probability'] = failure_probabilities
            predictions_df['maintenance_required'] = failure_probabilities >= threshold
            predictions_df['risk_level'] = predictions_df['failure_probability'].apply(
                self._calculate_risk_level
            )
            
            logging.info(f"Predictions generated:")
            logging.info(f"  - Maintenance required: {predictions_df['maintenance_required'].sum()} machines")
            logging.info(f"  - HIGH risk: {(predictions_df['risk_level'] == 'HIGH').sum()}")
            logging.info(f"  - MEDIUM risk: {(predictions_df['risk_level'] == 'MEDIUM').sum()}")
            logging.info(f"  - LOW risk: {(predictions_df['risk_level'] == 'LOW').sum()}")
            
            return predictions_df
            
        except Exception as e:
            logging.exception(f"Error generating predictions: {e}")
            raise MyException(e, sys)
    
    def initiate_prediction(
        self,
        model,
        features_df: pd.DataFrame,
        threshold: float,
        model_version: str
    ) -> PredictionArtifact:
        """
        Main entry point for prediction.
        """
        try:
            logging.info("Starting prediction")
            
            if features_df.empty:
                logging.warning("No features data for prediction. Returning empty artifact.")
                artifact = PredictionArtifact(
                    predictions_dataframe=pd.DataFrame(),
                    predictions_count=0,
                    maintenance_alerts_count=0,
                    model_version=model_version,
                    threshold=threshold,
                    prediction_timestamp=datetime.utcnow().isoformat(),
                    is_prediction_successful=False,
                    message="No features data for prediction"
                )
                return artifact
            
            # Generate predictions
            predictions_df = self.generate_predictions(model, features_df, threshold)
            
            # Count maintenance alerts
            maintenance_count = predictions_df['maintenance_required'].sum()
            
            # Save predictions
            os.makedirs(self.config.predictions_dir, exist_ok=True)
            # Use a filesystem-safe timestamp for filenames (avoid ':' on Windows)
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
            predictions_file = os.path.join(self.config.predictions_dir, f"predictions_{ts}.csv")
            try:
                predictions_df.to_csv(predictions_file, index=False)
            except Exception:
                # Fallback: use a simple incremental name if writing with timestamp fails
                predictions_file = os.path.join(self.config.predictions_dir, "predictions.csv")
                predictions_df.to_csv(predictions_file, index=False)
            
            artifact = PredictionArtifact(
                predictions_dataframe=predictions_df,
                predictions_file_path=predictions_file,
                predictions_count=len(predictions_df),
                maintenance_alerts_count=maintenance_count,
                model_version=model_version,
                threshold=threshold,
                prediction_timestamp=datetime.utcnow().isoformat(),
                is_prediction_successful=True,
                message=f"Prediction completed. {maintenance_count} maintenance alerts triggered."
            )
            
            logging.info("Prediction completed successfully")
            return artifact
            
        except Exception as e:
            logging.exception(f"Error in prediction: {e}")
            raise MyException(e, sys)
