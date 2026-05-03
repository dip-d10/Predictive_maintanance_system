import pandas as pd
from pymongo import MongoClient
from src.logger import logging
from src.utils.env_loader import MONGO_URI
import os


class MongoDBConnector:
    
    def __init__(self, database_name, collections):
        self.mongo_url = MONGO_URI
        self.database_name = database_name
        self.collections = collections

    def connect_to_database(self):
        try:
            logging.info("Connecting to MongoDB Atlas")

            client = MongoClient(self.mongo_url)
            database = client[self.database_name]

            logging.info("MongoDB connection successful")

            return database

        except Exception as e:
            logging.exception(e)
            raise e

    def extract_data(self):
        try:
            database = self.connect_to_database()

            collection_data = {}

            for collection_name in self.collections:

                logging.info(
                    f"Fetching {collection_name}"
                )

                collection = database[collection_name]
                data = list(collection.find())

                df = pd.DataFrame(data)

                if "_id" in df.columns:
                    df.drop(columns=["_id"], inplace=True)

                collection_data[collection_name] = df

                logging.info(
                    f"{collection_name}: {len(df)} rows extracted"
                )

            return collection_data

        except Exception as e:
            logging.exception(e)
            raise e