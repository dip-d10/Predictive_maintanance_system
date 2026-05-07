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