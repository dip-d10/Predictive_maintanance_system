from src.utils.main_utils import read_yaml, create_directories
from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, FeatureEngineeringConfig, ModelTrainerConfig
from pathlib import Path

class ConfigurationManager:
    
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        create_directories([
            self.config.artifacts_root,
            self.config.data_validation.root_dir,
            self.config.model_trainer.root_dir,
            self.config.model_trainer.trained_model_dir,
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
        
    def get_data_validation_config(self) -> DataValidationConfig:
        
        config = self.config.data_validation

        # Ensure the artifacts/data_validation directory exists before the component runs
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            raw_data_dir=Path(config.raw_data_dir),
            schema_file_path=SCHEMA_FILE_PATH,
            status_file=Path(config.status_file),
            merged_dataset_path=Path(config.merged_dataset_path)
        )

        return data_validation_config

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:

        config = self.config.feature_engineering

        create_directories([config.root_dir])

        return FeatureEngineeringConfig(
            root_dir=config.root_dir,
            master_data_path=config.master_data_path,
            feature_store_path=config.feature_store_path,
            final_feature_path=config.final_feature_path,
            lag_features=config.lag_features,
            rolling_windows=config.rolling_windows,
            prediction_horizons=config.prediction_horizons,
            telemetry_columns=config.telemetry_columns,
            machine_id_column=config.machine_id_column,
            datetime_column=config.datetime_column,
            failure_column=config.failure_column,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer

        create_directories([
            config.root_dir,
            config.trained_model_dir,
        ])

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            training_data_path=self.config.feature_engineering.final_feature_path,
            trained_model_dir=config.trained_model_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            metrics_file_path=config.metrics_file_path,
            test_size=config.test_size,
            random_state=config.random_state,
            max_training_rows=config.max_training_rows,
            target_column=config.target_column,
            datetime_column=self.config.feature_engineering.datetime_column,
            model_params=config.model_params,
        )