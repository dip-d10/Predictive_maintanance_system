from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import logging

STAGE_NAME_1 = "Data Ingestion Stage"
STAGE_NAME_2 = "Data Validation and Merging Stage"

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
        
        # ==========================================
        # STAGE 02: Data Validation & Merger
        # ==========================================
        logging.info(f">>>>>> {STAGE_NAME_2} started <<<<<<")
        # Ensure we capture the artifact to check the status if needed
        validation_artifact = pipeline.start_data_validation()
        
        if not validation_artifact.validation_status:
            logging.error("Pipeline Halted: Data Validation failed. Check status.txt")
            exit(1)
            
        logging.info(f">>>>>> {STAGE_NAME_2} completed successfully <<<<<<\n\nx==========x")
        
        # Future stages (Data Transformation, Model Trainer) will naturally drop in here.

    except Exception as e:
        logging.exception("Pipeline failed. Check logs for details.")
        exit(1)