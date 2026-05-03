from src.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    
    pipeline = TrainingPipeline()
    pipeline.start_data_ingestion()
    pipeline.start_raw_data_validation()
    