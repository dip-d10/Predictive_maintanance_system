from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
   root_dir: Path #root_dir: 'artifacts/data_ingestion'
   source_URL: str # MongoDB URL
   local_data_path: Path # 5 datasets
   raw_data_path : Path # Merged dataset
   
  
   
   
       
             