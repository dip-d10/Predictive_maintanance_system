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