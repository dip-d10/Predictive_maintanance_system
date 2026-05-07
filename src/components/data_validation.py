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
        Fail-Fast Mechanism: Validates all raw CSVs before any heavy Pandas operations.
        """
        try:
            logging.info("Starting raw dataset validation...")
            # We only validate raw schemas here, skipping Master_Dataset and target_column
            raw_schemas = {k: v for k, v in self.schema.items() if k not in ["Master_Dataset", "target_column"]}
            
            for dataset_name, dataset_schema in raw_schemas.items():
                file_path = os.path.join(self.config.raw_data_dir, f"{dataset_name}.csv")
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{dataset_name}.csv not found at {file_path}")

                df = pd.read_csv(file_path)
                
                # Schema validation
                expected_columns = list(dataset_schema["columns"].keys())
                actual_columns = list(df.columns)
                if expected_columns != actual_columns:
                    raise ValueError(f"{dataset_name}: Schema mismatch. Expected {expected_columns}, got {actual_columns}")

                # Null & Duplicate validation
                if df.isnull().sum().sum() > 0:
                    raise ValueError(f"{dataset_name}: Contains null values [cite: 94]")
                if df.duplicated().sum() > 0:
                    raise ValueError(f"{dataset_name}: Contains exact duplicate rows [cite: 95]")

                # Telemetry specific check (Uniqueness of spine)
                if dataset_name == "telemetry":
                    if df.duplicated(subset=["machineID", "datetime"]).sum() > 0:
                        raise ValueError("Telemetry: Duplicate timestamps found for a machine! ")

                logging.info(f"{dataset_name} passed raw validation.")
                
            return True

        except Exception as e:
            logging.error(f"Raw data validation failed: {str(e)} [cite: 97]")
            raise e

    def _aggregate_sparse_events(self, df: pd.DataFrame, prefix_col: str) -> pd.DataFrame:
        """
        Helper method to aggregate multiple events at the same timestamp.
        """
        # Convert categorical events into binary columns
        encoded = pd.get_dummies(df, columns=[prefix_col])
        # Group by machine and time to prevent row explosion during merge 
        aggregated = encoded.groupby(["machineID", "datetime"]).sum().reset_index()
        return aggregated

    def merge_datasets(self) -> pd.DataFrame:
        """
        Merges all datasets using Telemetry as the master spine.
        """
        logging.info("Starting dataset merge strategy...")
        
        # Load datasets
        telemetry = pd.read_csv(os.path.join(self.config.raw_data_dir, "telemetry.csv"))
        machines = pd.read_csv(os.path.join(self.config.raw_data_dir, "machines.csv"))
        failures = pd.read_csv(os.path.join(self.config.raw_data_dir, "failures.csv"))
        maint = pd.read_csv(os.path.join(self.config.raw_data_dir, "maintenance.csv"))
        errors = pd.read_csv(os.path.join(self.config.raw_data_dir, "errors.csv"))

        # Standardize Datetime
        for df in [telemetry, failures, maint, errors]:
            df["datetime"] = pd.to_datetime(df["datetime"])

        # Aggregate sparse events to resolve duplicate timestamps [cite: 42, 64-68]
        failures_agg = self._aggregate_sparse_events(failures, "failure")
        
        # Create overall failure target column [cite: 66]
        failure_cols = [col for col in failures_agg.columns if "failure_" in col]
        failures_agg["failure"] = (failures_agg[failure_cols].sum(axis=1) > 0).astype(float)

        maint_agg = self._aggregate_sparse_events(maint, "comp")
        errors_agg = self._aggregate_sparse_events(errors, "errorID")

        # Begin Merge on Telemetry Spine [cite: 68]
        master_df = telemetry.copy()
        
        master_df = master_df.merge(machines, on="machineID", how="left")
        master_df = master_df.merge(failures_agg, on=["machineID", "datetime"], how="left")
        master_df = master_df.merge(maint_agg, on=["machineID", "datetime"], how="left")
        master_df = master_df.merge(errors_agg, on=["machineID", "datetime"], how="left")

        # Fill sparse event nulls with 0 [cite: 70]
        event_columns = [col for col in master_df.columns if col not in telemetry.columns and col not in machines.columns]
        master_df[event_columns] = master_df[event_columns].fillna(0.0)

        # Mandatory sort for time-series windowing [cite: 71]
        master_df = master_df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
        
        return master_df

    def validate_merged_dataset(self, master_df: pd.DataFrame) -> bool:
        """
        Final quality gate. Validates the merged dataset against the Master_Dataset schema.
        """
        logging.info("Validating merged master dataset...")
        
        # 1. Check Row Explosion (Most critical join check)
        expected_rows = 876100  # 100 machines * 8761 hours
        if len(master_df) != expected_rows:
            raise ValueError(f"Merge caused row anomalies! Expected {expected_rows}, got {len(master_df)}. ")

        # 2. Schema check against config
        master_schema = self.schema["Master_Dataset"]["columns"]
        for col in master_schema.keys():
            if col not in master_df.columns:
                raise ValueError(f"Merged dataset missing column: {col}")
            
        # 3. Check for lingering nulls (confirms fillna worked)
        if master_df.isnull().sum().sum() > 0:
            raise ValueError("Merged dataset contains nulls. fillna(0) failed on sparse columns. [cite: 73]")

        logging.info("Master dataset validated successfully.")
        return True

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Orchestrates the entire component flow.
        """
        try:
            # Step 1: Validate Raw Files [cite: 103]
            raw_valid = self.validate_raw_datasets()

            # Step 2: Merge logic [cite: 42]
            master_df = self.merge_datasets()

            # Step 3: Validate Final Output [cite: 43]
            final_valid = self.validate_merged_dataset(master_df)

            # Step 4: Save artifact [cite: 74, 87]
            os.makedirs(os.path.dirname(self.config.merged_dataset_path), exist_ok=True)
            master_df.to_csv(self.config.merged_dataset_path, index=False)
            logging.info(f"Master dataset saved to {self.config.merged_dataset_path}")

            # Write status
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {final_valid}")

            return DataValidationArtifact(
                validation_status=final_valid,
                validation_report_path=self.config.status_file,
                merged_data_path=self.config.merged_dataset_path
            )

        except Exception as e:
            # If any stage fails, write False to status and halt
            with open(self.config.status_file, 'w') as f:
                f.write("Validation status: False")
            logging.exception(f"Validation component failed: {e}")
            raise e