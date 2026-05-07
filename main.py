from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import logging
from src.exception import MyException
import sys


STAGE_NAME_1 = "Data Ingestion Stage"
STAGE_NAME_2 = "Data Validation and Merging Stage"
STAGE_NAME_3 = "Feature Engineering Stage"

if __name__ == "__main__":
    try:
        # Initialize the pipeline orchestrator
        pipeline = TrainingPipeline()
        
        # ==========================================
        # STAGE 01: Data Ingestion
        # ==========================================
        logging.info(f">>>>>> {STAGE_NAME_1} started <<<<<<")
        pipeline.start_data_ingestion()
        logging.info(f">>>>>> {STAGE_NAME_1} completed successfully <<<<<<\n\nx==========x")
        
        pipeline = TrainingPipeline()

        logging.info(f">>>>>> {STAGE_NAME_2} started <<<<<<")
        validation_artifact = pipeline.start_data_validation()

        if not validation_artifact.validation_status:
            logging.error("Pipeline halted: data validation failed. Check the status file.")
            exit(1)

        logging.info(f">>>>>> {STAGE_NAME_2} completed successfully <<<<<<\n\nx==========x")

        logging.info(f">>>>>> {STAGE_NAME_3} started <<<<<<")
        feature_engineering_artifact = pipeline.start_feature_engineering()

        if not feature_engineering_artifact.is_engineering_successful:
            logging.error("Pipeline halted: feature engineering failed.")
            exit(1)

        logging.info(f">>>>>> {STAGE_NAME_3} completed successfully <<<<<<\n\nx==========x")

    except Exception as e:
        raise MyException(e, sys) from e