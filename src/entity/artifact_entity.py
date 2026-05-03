from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_data_dir:str 
    
    
@dataclass
class RawDataValidationArtifact:
    validation_status: bool
    validation_report_path: str   