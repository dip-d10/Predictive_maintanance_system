import os
import pandas as pd
from src.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.utils.main_utils import read_yaml


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.schema = read_yaml(config.schema_file_path)

    def validate_raw_datasets(self) -> bool:
        """
        Validate each raw CSV before any merge or heavy Pandas operations.
        """
        try:
            logging.info("Starting raw dataset validation...")

            raw_schemas = {
                key: value
                for key, value in self.schema.items()
                if key not in ["Master_Dataset", "target_column"]
            }

            for dataset_name, dataset_schema in raw_schemas.items():
                file_path = os.path.join(self.config.raw_data_dir, f"{dataset_name}.csv")

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{dataset_name}.csv not found at {file_path}")

                df = pd.read_csv(file_path)

                expected_columns = list(dataset_schema["columns"].keys())
                actual_columns = list(df.columns)
                if expected_columns != actual_columns:
                    raise ValueError(
                        f"{dataset_name}: Schema mismatch. Expected {expected_columns}, got {actual_columns}"
                    )

                if df.isnull().sum().sum() > 0:
                    raise ValueError(f"{dataset_name}: Contains null values")

                if df.duplicated().sum() > 0:
                    raise ValueError(f"{dataset_name}: Contains exact duplicate rows")

                if dataset_name == "telemetry" and df.duplicated(subset=["machineID", "datetime"]).sum() > 0:
                    raise ValueError("Telemetry: Duplicate timestamps found for a machine")

                logging.info(f"{dataset_name} passed raw validation.")

            return True

        except Exception as e:
            logging.error(f"Raw data validation failed: {str(e)}")
            raise e

    def _aggregate_sparse_events(self, df: pd.DataFrame, event_column: str, prefix: str | None = None) -> pd.DataFrame:
        """
        One-hot encode sparse event columns and aggregate duplicate timestamps.
        """
        encoded = pd.get_dummies(df, columns=[event_column], prefix=prefix or event_column)
        aggregated = encoded.groupby(["machineID", "datetime"]).sum().reset_index()
        return aggregated

    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge telemetry with machine metadata and sparse event datasets.
        """
        logging.info("Starting dataset merge strategy...")

        telemetry = pd.read_csv(os.path.join(self.config.raw_data_dir, "telemetry.csv"))
        failures = pd.read_csv(os.path.join(self.config.raw_data_dir, "failures.csv"))
        maint = pd.read_csv(os.path.join(self.config.raw_data_dir, "maintenance.csv"))
        errors = pd.read_csv(os.path.join(self.config.raw_data_dir, "errors.csv"))

        for df in [telemetry, failures, maint, errors]:
            df["datetime"] = pd.to_datetime(df["datetime"])

        telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)

        failures_agg = self._aggregate_sparse_events(failures, "failure")
        failure_cols = [col for col in failures_agg.columns if col.startswith("failure_") and col != "failure"]
        failures_agg["failure"] = (failures_agg[failure_cols].sum(axis=1) > 0).astype(int)

        maint_agg = self._aggregate_sparse_events(maint, "comp", prefix="maint")
        errors_agg = self._aggregate_sparse_events(errors, "errorID")

        master_df = telemetry.copy()
        master_df = master_df.merge(pd.read_csv(os.path.join(self.config.raw_data_dir, "machines.csv")), on="machineID", how="left")
        master_df = master_df.merge(failures_agg, on=["machineID", "datetime"], how="left")
        master_df = master_df.merge(maint_agg, on=["machineID", "datetime"], how="left")
        master_df = master_df.merge(errors_agg, on=["machineID", "datetime"], how="left")

        event_columns = [
            col for col in master_df.columns
            if col not in telemetry.columns and col not in ["model", "age"]
        ]
        master_df[event_columns] = master_df[event_columns].fillna(0)

        master_df = master_df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
        return master_df

    def validate_merged_dataset(self, master_df: pd.DataFrame) -> bool:
        """
        Final quality gate for the merged master dataset.
        """
        logging.info("Validating merged master dataset...")

        telemetry = pd.read_csv(os.path.join(self.config.raw_data_dir, "telemetry.csv"))

        if len(master_df) != len(telemetry):
            raise ValueError(
                f"Merge caused row anomalies! Expected {len(telemetry)}, got {len(master_df)}. "
            )

        if master_df.duplicated(subset=["machineID", "datetime"]).sum() > 0:
            raise ValueError("Merged dataset contains duplicate machineID + datetime rows")

        for column in ["volt", "rotate", "pressure", "vibration", "model", "age"]:
            if column not in master_df.columns:
                raise ValueError(f"Merged dataset missing column: {column}")

        telemetry_columns = ["volt", "rotate", "pressure", "vibration"]
        machine_columns = ["model", "age"]
        if master_df[telemetry_columns].isnull().sum().sum() > 0:
            raise ValueError("Merged dataset contains nulls in telemetry columns")
        if master_df[machine_columns].isnull().sum().sum() > 0:
            raise ValueError("Merged dataset contains nulls in machine metadata columns")

        event_columns = [
            col for col in master_df.columns
            if col.startswith("failure_") or col.startswith("maint_") or col.startswith("errorID_") or col == "failure"
        ]
        for column in event_columns:
            if not pd.api.types.is_numeric_dtype(master_df[column]):
                raise ValueError(f"Event column {column} must be numeric")

        logging.info("Master dataset validated successfully.")
        return True

    def calculate_prediction_lookback_window(self) -> int:
        """
        Dynamically calculate required lookback window for prediction.
        
        Considers:
        - Maximum lag features
        - Maximum rolling windows
        - Error event window
        - Maintenance window
        
        Returns: Required lookback in hours
        """
        max_lag = max(self.config.lag_features) if self.config.lag_features else 0
        max_rolling = max(self.config.rolling_windows) if self.config.rolling_windows else 0
        
        # Rolling windows in the config are hourly, so they represent hours directly
        required_window = max(
            max_lag,
            max_rolling,
            self.config.error_window_hours,
            self.config.maintenance_window_hours
        )
        
        logging.info(f"Prediction lookback window calculated: {required_window} hours")
        logging.info(f"  - Max lag features: {max_lag}h")
        logging.info(f"  - Max rolling windows: {max_rolling}h")
        logging.info(f"  - Error window: {self.config.error_window_hours}h")
        logging.info(f"  - Maintenance window: {self.config.maintenance_window_hours}h")
        
        return required_window

    def create_prediction_input_dataset(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract latest records for prediction inference.
        
        This simulates the latest batch of telemetry data that would be
        received in production for real-time inference.
        
        Steps:
        1. Calculate required lookback window
        2. Find latest timestamp in data
        3. Extract records: latest_timestamp - lookback_window to latest_timestamp
        4. Save as prediction_input_dataset.csv
        
        Returns: Prediction dataset
        """
        try:
            logging.info("Creating prediction input dataset for demo mode...")
            
            # Calculate lookback window
            lookback_hours = self.calculate_prediction_lookback_window()
            
            # Find latest timestamp
            master_df['datetime'] = pd.to_datetime(master_df['datetime'])
            latest_timestamp = master_df['datetime'].max()
            lookback_start = latest_timestamp - pd.Timedelta(hours=lookback_hours)
            
            logging.info(f"Latest timestamp in dataset: {latest_timestamp}")
            logging.info(f"Lookback period: {lookback_start} to {latest_timestamp}")
            
            # Extract prediction dataset
            prediction_df = master_df[
                (master_df['datetime'] >= lookback_start) &
                (master_df['datetime'] <= latest_timestamp)
            ].copy()
            
            logging.info(f"Extracted {len(prediction_df)} records for prediction")
            
            # Save prediction dataset
            os.makedirs(os.path.dirname(self.config.prediction_input_path), exist_ok=True)
            prediction_df.to_csv(self.config.prediction_input_path, index=False)
            logging.info(f"Prediction input dataset saved to: {self.config.prediction_input_path}")
            
            # Log sample
            unique_machines = prediction_df['machineID'].nunique()
            logging.info(f"Prediction dataset contains {unique_machines} unique machines")
            
            return prediction_df
            
        except Exception as e:
            logging.exception(f"Error creating prediction input dataset: {e}")
            raise e

    def initiate_data_validation(self):
        """
        Run the full validation and merge flow.
        Also creates prediction input dataset for demo mode.
        """
        try:
            self.validate_raw_datasets()
            master_df = self.merge_datasets()
            final_valid = self.validate_merged_dataset(master_df)

            os.makedirs(os.path.dirname(self.config.merged_dataset_path), exist_ok=True)
            master_df.to_csv(self.config.merged_dataset_path, index=False)

            # Create prediction input dataset for demo mode
            prediction_df = self.create_prediction_input_dataset(master_df)

            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {final_valid}")

            return DataValidationArtifact(
                validation_status=final_valid,
                validation_report_path=self.config.status_file,
                merged_data_path=self.config.merged_dataset_path,
                prediction_input_path=self.config.prediction_input_path
            )

        except Exception as e:
            with open(self.config.status_file, "w") as f:
                f.write("Validation status: False")
            logging.exception(f"Validation component failed: {e}")
            raise e