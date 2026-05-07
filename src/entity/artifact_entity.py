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