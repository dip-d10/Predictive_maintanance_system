from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    raw_data_dir:str 
    
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: Path
    merged_data_path: Path
    prediction_input_path: Path
    
@dataclass
class FeatureEngineeringArtifact:
    feature_store_path: str
    final_feature_path: str
    is_engineering_successful: bool
    message: str    


@dataclass
class ModelTrainerArtifact:
    trained_model_dir: str
    train_data_path: str
    test_data_path: str
    model_metrics_path: str    


@dataclass
class ModelEvaluationArtifact:
    best_model_path: str
    threshold_path: str
    evaluation_report_path: str
    approved_model: bool
    best_model_name: str
    best_threshold: float
    best_business_cost: float


@dataclass
class ModelPusherArtifact:
    production_blob_uri: str
    archived_blob_uri: str
    deployment_status: str
    deployed_model_version: str

# ============================================================
# PREDICTION PIPELINE ARTIFACTS
# ============================================================

@dataclass
class PredictionDataIngestionArtifact:
    telemetry_data: object  # pd.DataFrame
    machines_data: object  # pd.DataFrame
    errors_data: object  # pd.DataFrame
    maintenance_data: object  # pd.DataFrame
    prediction_data_dir: str
    ingestion_timestamp: str
    records_count: int


@dataclass
class PredictionFeatureEngineeringArtifact:
    features_dataframe: object  # pd.DataFrame
    features_file_path: str
    engineered_features_count: int
    records_count: int
    is_engineering_successful: bool
    message: str


@dataclass
class ModelLoaderArtifact:
    model: object  # Loaded model object
    model_path: str
    metadata: dict
    threshold: float
    model_version: str
    load_timestamp: str
    is_load_successful: bool
    message: str


@dataclass
class PredictionArtifact:
    predictions_dataframe: object  # pd.DataFrame
    predictions_file_path: str = None
    predictions_count: int = 0
    maintenance_alerts_count: int = 0
    model_version: str = None
    threshold: float = None
    prediction_timestamp: str = None
    is_prediction_successful: bool = False
    message: str = ""


@dataclass
class PredictionStorageArtifact:
    predictions_stored_count: int
    maintenance_alerts_stored: int
    batch_id: str
    storage_timestamp: str
    is_storage_successful: bool
    message: str


@dataclass
class AlertArtifact:
    alerts_triggered: int
    high_risk_alerts: int
    medium_risk_alerts: int
    alerts_list: list
    email_notification_sent: bool
    slack_notification_sent: bool
    alert_timestamp: str
    is_alert_successful: bool
    message: str
