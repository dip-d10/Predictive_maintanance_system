from src.utils.main_utils import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig

schema_file_path = ""
config_file_path = ""

schema = read_yaml(schema_file_path)
config = read_yaml(config_file_path)


class ConfigurationManager:
    def __init__(
        self,schema,config ):

        self.config = config
        self.schema = schema

        create_directories([self.config.artifacts_root]) # Creates the root directory for artifacts if it does not exist.

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir]) # creates a data_ingestion folder inside it contain zip folder and unzip file

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        ) # Creates and returns a DataIngestionConfig instance using the settings from the config file.

        return data_ingestion_config
    