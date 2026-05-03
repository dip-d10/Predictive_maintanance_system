from src.configuration.config_manager import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.logger import logging


STAGE_NAME = "DATA INGESTION STAGE"


class TrainingPipeline:

    def __init__(self):
        pass

    def start_data_ingestion(self):
        try:
            logging.info(
                f">>>>>> {STAGE_NAME} started <<<<<<"
            )

            config = ConfigurationManager()

            ingestion_config = (
                config.get_data_ingestion_config()
            )

            ingestion = DataIngestion(
                config=ingestion_config
            )

            artifact = (
                ingestion.initiate_data_ingestion()
            )

            logging.info(
                f"Raw data stored at {artifact.raw_data_dir}"
            )

            logging.info(
                f">>>>>> {STAGE_NAME} completed <<<<<<"
            )

        except Exception as e:
            logging.exception(e)
            raise e