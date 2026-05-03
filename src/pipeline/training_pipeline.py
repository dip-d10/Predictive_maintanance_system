from src.configuration.config_manager import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.raw_data_validation import RawDataValidation
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
        
    def start_raw_data_validation(self):
        try:
            logging.info(
                ">>>>>> RAW DATA VALIDATION STAGE started <<<<<<"
            )

            config = ConfigurationManager()

            validation_config = (
                config.get_raw_data_validation_config()
            )

            validation = RawDataValidation(
                config=validation_config
            )

            validation_artifact = (
                validation.validate_all_files()
            )

            logging.info(
                f"Validation Status: "
                f"{validation_artifact.validation_status}"
            )

            logging.info(
                f"Validation Report Path: "
                f"{validation_artifact.validation_report_path}"
            )

            logging.info(
                ">>>>>> RAW DATA VALIDATION STAGE completed <<<<<<"
            )

        except Exception as e:
            logging.exception(e)
            raise e    