import os

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.configuration.mongo_db_connection import MongoDBConnector
from src.logger import logging


class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            mongo_connector = MongoDBConnector(
                database_name=self.config.database_name,
                collections=self.config.collections
            )

            collection_data = mongo_connector.extract_data()

            for collection_name, df in collection_data.items():

                file_path = os.path.join(
                    self.config.raw_data_dir,
                    f"{collection_name}.csv"
                )

                df.to_csv(file_path, index=False)

                logging.info(
                    f"{collection_name}.csv saved at {file_path}"
                )

            logging.info("Data ingestion completed")

            return DataIngestionArtifact(
                raw_data_dir=self.config.raw_data_dir
            )

        except Exception as e:
            logging.exception(e)
            raise e