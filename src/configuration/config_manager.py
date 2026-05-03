from src.utils.main_utils import read_yaml, create_directories
from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)

        create_directories([
            self.config.artifacts_root
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