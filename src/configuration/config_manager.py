from src.utils.main_utils import read_yaml, create_directories
from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from pathlib import Path

class ConfigurationManager:
    
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        create_directories([
            self.config.artifacts_root,
            self.config.data_validation.root_dir
        ])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([
            config.root_dir,
            config.raw_data_dir
        ])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            database_name=config.database_name,
            collections=config.collections,
            raw_data_dir=config.raw_data_dir
        )
        
    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Reads the data_validation section from config.yaml and 
        returns a strictly-typed DataValidationConfig object.
        """
        config = self.config.data_validation

        # Ensure the artifacts/data_validation directory exists before the component runs
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            raw_data_dir=Path(config.raw_data_dir),
            schema_file_path=Path(config.schema_file_path),
            status_file=Path(config.status_file),
            merged_dataset_path=Path(config.merged_dataset_path)
        )

        return data_validation_config