from src.configuration.config_manager import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def start_data_ingestion(self):
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        return data_ingestion.initiate_data_ingestion()


    def start_data_validation(self):
        """
        Build the DataValidation component and run the full validation flow.
        """
        data_validation_config = self.config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        return data_validation.initiate_data_validation()

    def start_feature_engineering(self):
        """
        Build the FeatureEngineering component and run the full feature engineering flow.
        """
        feature_engineering_config = self.config_manager.get_feature_engineering_config()
        feature_engineering = FeatureEngineering(config=feature_engineering_config)
        return feature_engineering.initiate_feature_engineering()

    def start_model_trainer(self):
        """
        Build the ModelTrainer component and run the model training flow.
        """
        model_trainer_config = self.config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        return model_trainer.initiate_model_trainer()

    def start_model_evaluation(self):
        """
        Build the ModelEvaluation component and run the evaluation flow.
        """
        model_evaluation_config = self.config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        return model_evaluation.initiate_model_evaluation()