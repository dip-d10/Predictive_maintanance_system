from src.utils.main_utils import read_yaml, create_directories
from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    AzureBlobConfig,
    ModelPusherConfig,
    PredictionDataIngestionConfig,
    PredictionFeatureEngineeringConfig,
    ModelLoaderConfig,
    PredictorConfig,
    PredictionStorageConfig,
    AlertConfig,
)
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
            self.config.model_pusher.root_dir,
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
        prediction_config = self.config.prediction_data

        # Ensure the artifacts/data_validation directory exists before the component runs
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            raw_data_dir=Path(config.raw_data_dir),
            schema_file_path=SCHEMA_FILE_PATH,
            status_file=Path(config.status_file),
            merged_dataset_path=Path(config.merged_dataset_path),
            prediction_input_path=Path(prediction_config.prediction_input_path),
            lag_features=self.config.feature_engineering.lag_features,
            rolling_windows=self.config.feature_engineering.rolling_windows,
            error_window_hours=prediction_config.error_window_hours,
            maintenance_window_hours=prediction_config.maintenance_window_hours,
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

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        # Ensure evaluation artifact directory exists
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            trained_model_dir=self.config.model_trainer.trained_model_dir,
            train_data_path=self.config.model_trainer.train_data_path,
            test_data_path=self.config.model_trainer.test_data_path,
            evaluation_report_path=config.evaluation_report_path,
            threshold_report_path=config.threshold_report_path,
            best_model_path=config.best_model_path,
            target_column=self.config.model_trainer.target_column,
            datetime_column=self.config.feature_engineering.datetime_column,
            downtime_cost=config.downtime_cost,
            maintenance_cost=config.maintenance_cost,
            min_pr_auc_threshold=config.min_pr_auc_threshold,
        )

    def get_azure_blob_config(self) -> AzureBlobConfig:
        config = self.config.azure_blob

        return AzureBlobConfig(
            container_name=config.container_name,
        )

    def get_model_pusher_config(self) -> ModelPusherConfig:
        config = self.config.model_pusher

        create_directories([
            config.root_dir,
            str(Path(config.metadata_local_path).parent),
        ])

        return ModelPusherConfig(
            root_dir=config.root_dir,
            metadata_local_path=config.metadata_local_path,
        )

    # ============================================================
    # PREDICTION PIPELINE CONFIG METHODS
    # ============================================================

    def get_prediction_data_ingestion_config(self) -> PredictionDataIngestionConfig:
        config = self.config.prediction_data_ingestion
        prediction_config = self.config.prediction_data

        create_directories([config.prediction_data_dir])

        return PredictionDataIngestionConfig(
            prediction_input_path=prediction_config.prediction_input_path,
            prediction_data_dir=config.prediction_data_dir,
        )

    def get_prediction_feature_engineering_config(self) -> PredictionFeatureEngineeringConfig:
        config = self.config.prediction_feature_engineering

        create_directories([config.prediction_features_dir])

        return PredictionFeatureEngineeringConfig(
            prediction_features_dir=config.prediction_features_dir,
            telemetry_columns=self.config.feature_engineering.telemetry_columns,
            lag_features=self.config.feature_engineering.lag_features,
            rolling_windows=self.config.feature_engineering.rolling_windows,
            machine_id_column=self.config.feature_engineering.machine_id_column,
            datetime_column=self.config.feature_engineering.datetime_column,
        )

    def get_model_loader_config(self) -> ModelLoaderConfig:
        config = self.config.model_loader

        create_directories([config.model_cache_dir])

        return ModelLoaderConfig(
            container_name=self.config.azure_blob.container_name,
            production_model_blob_path=config.production_model_blob_path,
            production_metadata_blob_path=config.production_metadata_blob_path,
            model_cache_dir=config.model_cache_dir,
            default_threshold=config.default_threshold,
        )

    def get_predictor_config(self) -> PredictorConfig:
        config = self.config.predictor

        create_directories([config.predictions_dir])

        return PredictorConfig(
            predictions_dir=config.predictions_dir,
            high_risk_threshold=config.high_risk_threshold,
            medium_risk_threshold=config.medium_risk_threshold,
        )

    def get_prediction_storage_config(self) -> PredictionStorageConfig:
        config = self.config.prediction_storage

        return PredictionStorageConfig(
            database_name=config.database_name,
            predictions_collection=config.predictions_collection,
            prediction_summary_collection=config.prediction_summary_collection,
        )

    def get_alert_config(self) -> AlertConfig:
        config = self.config.alert_manager

        create_directories([config.alerts_log_dir])

        return AlertConfig(
            alerts_log_dir=config.alerts_log_dir,
            enable_email=config.enable_email,
            email_recipients=config.email_recipients,
            enable_slack=config.enable_slack,
            slack_channel=config.slack_channel,
        )
