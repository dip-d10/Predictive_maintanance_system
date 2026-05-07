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