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
      