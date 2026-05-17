from dataclasses import dataclass
from typing import List
from pathlib import Path



@dataclass
class DataIngestionConfig:
    
    root_dir: str
    database_name: str
    collections: List[str]
    raw_data_dir: str
   
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    raw_data_dir: Path
    schema_file_path: Path
    status_file: Path
    merged_dataset_path: Path
    prediction_input_path: Path
    lag_features: List
    rolling_windows: List
    error_window_hours: int
    maintenance_window_hours: int
    
    
@dataclass
class FeatureEngineeringConfig:
    root_dir: str
    master_data_path: str
    feature_store_path: str
    final_feature_path: str
    lag_features: list
    rolling_windows: list
    prediction_horizons: list
    telemetry_columns: list
    machine_id_column: str
    datetime_column: str
    failure_column: str   
      

@dataclass
class ModelTrainerConfig:
    root_dir: str
    training_data_path: str
    trained_model_dir: str
    train_data_path: str
    test_data_path: str
    metrics_file_path: str
    test_size: float
    random_state: int
    max_training_rows: int
    target_column: str
    datetime_column: str
    model_params: dict      
      

@dataclass
class ModelEvaluationConfig:
    root_dir: str
    trained_model_dir: str
    train_data_path: str
    test_data_path: str
    evaluation_report_path: str
    threshold_report_path: str
    best_model_path: str
    target_column: str
    datetime_column: str
    downtime_cost: int
    maintenance_cost: int
    min_pr_auc_threshold: float


@dataclass
class AzureBlobConfig:
    container_name: str


@dataclass
class ModelPusherConfig:
    root_dir: str
    metadata_local_path: str


# ============================================================
# PREDICTION PIPELINE CONFIGS
# ============================================================

@dataclass
class PredictionDataIngestionConfig:
    prediction_input_path: str
    prediction_data_dir: str


@dataclass
class PredictionFeatureEngineeringConfig:
    prediction_features_dir: str
    telemetry_columns: list
    lag_features: list
    rolling_windows: list
    machine_id_column: str
    datetime_column: str


@dataclass
class ModelLoaderConfig:
    container_name: str
    production_model_blob_path: str
    production_metadata_blob_path: str
    model_cache_dir: str
    default_threshold: float


@dataclass
class PredictorConfig:
    predictions_dir: str
    high_risk_threshold: float
    medium_risk_threshold: float


@dataclass
class PredictionStorageConfig:
    database_name: str
    predictions_collection: str
    prediction_summary_collection: str


@dataclass
class AlertConfig:
    alerts_log_dir: str
    enable_email: bool
    email_recipients: list
    enable_slack: bool
    slack_channel: str
