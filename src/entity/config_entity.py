from dataclasses import dataclass
from typing import List


@dataclass
class DataIngestionConfig:
    
    root_dir: str
    database_name: str
    collections: List[str]
    raw_data_dir: str
   
  
   
   
       
             