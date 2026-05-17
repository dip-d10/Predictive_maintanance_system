import os
import sys
import pandas as pd
import numpy as np

from src.entity.config_entity import PredictionFeatureEngineeringConfig
from src.entity.artifact_entity import PredictionFeatureEngineeringArtifact
from src.exception import MyException
from src.logger import logging


class PredictionFeatureEngineering:
    """
    Generate prediction features using same logic as training.
    Ensures consistency between training and prediction.
    """
    
    def __init__(self, config: PredictionFeatureEngineeringConfig):
        self.config = config
        self.machine_id_column = config.machine_id_column
        self.datetime_column = config.datetime_column
        self.telemetry_columns = list(config.telemetry_columns)
        self.lag_features = list(config.lag_features)
        self.rolling_windows = list(config.rolling_windows)
    
    def _sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by machine and datetime"""
        df = df.copy()
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        return df.sort_values(
            by=[self.machine_id_column, self.datetime_column]
        ).reset_index(drop=True)
    
    def create_machine_features(self, telemetry_df: pd.DataFrame, machines_df: pd.DataFrame) -> pd.DataFrame:
        """Create machine-level features (age, model encoding, current telemetry)"""
        df = self._sort_dataframe(telemetry_df)
        
        try:
            # Merge machine metadata
            if not machines_df.empty:
                df = df.merge(
                    machines_df,
                    left_on=self.machine_id_column,
                    right_on=self.machine_id_column,
                    how="left"
                )
            
            # Create machine age
            if "age" in df.columns:
                df["machine_age"] = df["age"]
            elif "manufacture_year" in df.columns:
                df["machine_age"] = pd.to_datetime(df[self.datetime_column]).dt.year - df["manufacture_year"]
            else:
                df["machine_age"] = 0
            
            # One-hot encode model
            if "model" in df.columns:
                model_dummies = pd.get_dummies(df["model"], prefix="model")
                df = pd.concat([df, model_dummies], axis=1)
            
            # Ensure telemetry columns exist
            for column in self.telemetry_columns:
                if column not in df.columns:
                    logging.warning(f"Missing telemetry column: {column}. Setting to 0.")
                    df[column] = 0
                else:
                    df[f"current_{column}"] = df[column]
            
            return df
            
        except Exception as e:
            logging.exception(f"Error creating machine features: {e}")
            raise MyException(e, sys)
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for each telemetry column"""
        df = df.copy()
        
        try:
            grouped = df.groupby(self.machine_id_column, sort=False)
            
            for column in self.telemetry_columns:
                if column in df.columns:
                    for lag in self.lag_features:
                        df[f"{column}_lag_{lag}"] = grouped[column].shift(lag)
            
            return df
            
        except Exception as e:
            logging.exception(f"Error creating lag features: {e}")
            raise MyException(e, sys)
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean and std features"""
        df = df.copy()
        
        try:
            for column in self.telemetry_columns:
                if column in df.columns:
                    grouped = df.groupby(self.machine_id_column, sort=False)[column]
                    
                    for window in self.rolling_windows:
                        df[f"{column}_rolling_mean_{window}"] = grouped.transform(
                            lambda series, window=window: series.rolling(window=window, min_periods=1).mean()
                        )
                        df[f"{column}_rolling_std_{window}"] = grouped.transform(
                            lambda series, window=window: series.rolling(window=window, min_periods=1).std()
                        )
            
            return df
            
        except Exception as e:
            logging.exception(f"Error creating rolling features: {e}")
            raise MyException(e, sys)
    
    def create_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create change (delta) features"""
        df = df.copy()
        
        try:
            for column in self.telemetry_columns:
                if column in df.columns:
                    if f"{column}_lag_1" in df.columns:
                        df[f"{column}_delta_1"] = df[column] - df[f"{column}_lag_1"]
                    
                    if f"{column}_lag_3" in df.columns:
                        df[f"{column}_delta_3"] = df[column] - df[f"{column}_lag_3"]
            
            return df
            
        except Exception as e:
            logging.exception(f"Error creating delta features: {e}")
            raise MyException(e, sys)
    
    def create_error_features(self, df: pd.DataFrame, errors_df: pd.DataFrame) -> pd.DataFrame:
        """Create error-based features (error frequency, etc.)"""
        df = df.copy()
        
        try:
            if errors_df.empty:
                df["error_count_last_24h"] = 0
                df["error_count_last_7d"] = 0
                return df
            
            # Count errors by machine and time window
            errors_df["datetime"] = pd.to_datetime(errors_df.get("datetime", errors_df.get("time", pd.Series())))
            
            for _, row in df.iterrows():
                machine_id = row[self.machine_id_column]
                current_time = pd.to_datetime(row[self.datetime_column])
                
                # Get errors for this machine in different time windows
                machine_errors = errors_df[errors_df.get(self.machine_id_column) == machine_id]
                
                # Last 24 hours
                last_24h = machine_errors[
                    (machine_errors["datetime"] >= current_time - pd.Timedelta(hours=24)) &
                    (machine_errors["datetime"] <= current_time)
                ]
                df.loc[_, "error_count_last_24h"] = len(last_24h)
                
                # Last 7 days
                last_7d = machine_errors[
                    (machine_errors["datetime"] >= current_time - pd.Timedelta(days=7)) &
                    (machine_errors["datetime"] <= current_time)
                ]
                df.loc[_, "error_count_last_7d"] = len(last_7d)
            
            return df
            
        except Exception as e:
            logging.warning(f"Could not create error features: {e}")
            # Return df with default values
            if "error_count_last_24h" not in df.columns:
                df["error_count_last_24h"] = 0
            if "error_count_last_7d" not in df.columns:
                df["error_count_last_7d"] = 0
            return df
    
    def create_maintenance_features(self, df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
        """Create maintenance-based features"""
        df = df.copy()
        
        try:
            if maintenance_df.empty:
                df["days_since_last_maintenance"] = -1
                df["maintenance_count_last_30d"] = 0
                return df
            
            maintenance_df["datetime"] = pd.to_datetime(
                maintenance_df.get("datetime", maintenance_df.get("time", pd.Series()))
            )
            
            for idx, row in df.iterrows():
                machine_id = row[self.machine_id_column]
                current_time = pd.to_datetime(row[self.datetime_column])
                
                # Get maintenance records for this machine
                machine_maint = maintenance_df[maintenance_df.get(self.machine_id_column) == machine_id]
                
                if not machine_maint.empty:
                    # Days since last maintenance
                    last_maint = machine_maint[machine_maint["datetime"] <= current_time].sort_values("datetime")
                    if not last_maint.empty:
                        days_since = (current_time - last_maint.iloc[-1]["datetime"]).days
                        df.loc[idx, "days_since_last_maintenance"] = days_since
                    else:
                        df.loc[idx, "days_since_last_maintenance"] = -1
                    
                    # Maintenance count in last 30 days
                    last_30d = machine_maint[
                        (machine_maint["datetime"] >= current_time - pd.Timedelta(days=30)) &
                        (machine_maint["datetime"] <= current_time)
                    ]
                    df.loc[idx, "maintenance_count_last_30d"] = len(last_30d)
                else:
                    df.loc[idx, "days_since_last_maintenance"] = -1
                    df.loc[idx, "maintenance_count_last_30d"] = 0
            
            return df
            
        except Exception as e:
            logging.warning(f"Could not create maintenance features: {e}")
            if "days_since_last_maintenance" not in df.columns:
                df["days_since_last_maintenance"] = -1
            if "maintenance_count_last_30d" not in df.columns:
                df["maintenance_count_last_30d"] = 0
            return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values (forward fill, backward fill, drop if necessary)"""
        df = df.copy()
        
        try:
            # Forward fill for lagged/rolling features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df.groupby(self.machine_id_column)[numeric_cols].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(0)
            )
            
            # Fill any remaining NaNs with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logging.warning(f"Error handling missing values: {e}")
            return df.fillna(0)
    
    def initiate_prediction_feature_engineering(
        self,
        telemetry_data: pd.DataFrame,
        machines_data: pd.DataFrame,
        errors_data: pd.DataFrame,
        maintenance_data: pd.DataFrame
    ) -> PredictionFeatureEngineeringArtifact:
        """
        Main entry point for prediction feature engineering.
        
        IMPORTANT: This generates features using historical context, but returns
        ONLY the latest record per machine for prediction scoring.
        
        Flow:
        - Historical rows (e.g., 168 hours × 100 machines = ~16,900 rows)
        - → Generate lag/rolling features using context
        - → Drop NaN rows
        - → Keep ONLY latest row per machine
        - → Return ~100 rows for prediction
        """
        try:
            logging.info("Starting prediction feature engineering")
            
            # Create features in same order as training
            df = self.create_machine_features(telemetry_data, machines_data)
            df = self.create_lag_features(df)
            df = self.create_rolling_features(df)
            df = self.create_delta_features(df)
            df = self.create_error_features(df, errors_data)
            df = self.create_maintenance_features(df, maintenance_data)
            df = self.handle_missing_values(df)
            
            # Log raw feature engineering results
            logging.info(f"Total rows after feature engineering: {len(df)}")
            logging.info(f"Unique machines in data: {df[self.machine_id_column].nunique()}")
            
            # CRITICAL STEP: Keep only latest record per machine for prediction
            # Historical rows were used ONLY for generating lag/rolling features
            logging.info("Filtering to latest row per machine for prediction...")
            
            # Step 1: Ensure proper sorting
            df = df.sort_values(
                by=[self.machine_id_column, self.datetime_column]
            ).reset_index(drop=True)
            logging.info(f"Data sorted by {self.machine_id_column} and {self.datetime_column}")
            
            # Step 2: Drop rows with NaN (lag features create NaNs in early rows)
            rows_before_dropna = len(df)
            df = df.dropna()
            rows_after_dropna = len(df)
            logging.info(f"Dropped {rows_before_dropna - rows_after_dropna} rows with NaN values")
            logging.info(f"Rows after NaN removal: {rows_after_dropna}")
            
            # Step 3: Keep only latest record per machine
            final_prediction_df = (
                df.groupby(self.machine_id_column, sort=False)
                .tail(1)
                .reset_index(drop=True)
            )
            
            logging.info(f"Final prediction rows (after keeping latest per machine): {len(final_prediction_df)}")
            logging.info(f"Unique machines in final dataset: {final_prediction_df[self.machine_id_column].nunique()}")
            
            # Step 4: Validation - ensure one row per machine
            unique_machines = final_prediction_df[self.machine_id_column].nunique()
            assert unique_machines == len(final_prediction_df), \
                f"Data validation failed: Expected {len(final_prediction_df)} rows for {unique_machines} machines, but got {len(final_prediction_df)} rows"
            logging.info(f"[VALIDATED] One row per machine confirmed: {unique_machines} machines -> {len(final_prediction_df)} predictions")
            
            # Save only the final prediction features (latest per machine)
            os.makedirs(self.config.prediction_features_dir, exist_ok=True)
            features_file = os.path.join(self.config.prediction_features_dir, "prediction_features.csv")
            final_prediction_df.to_csv(features_file, index=False)
            logging.info(f"Saved prediction features to {features_file}")
            logging.info(f"Feature matrix shape: {final_prediction_df.shape} (rows, features)")
            
            # Use ONLY final_prediction_df for the artifact
            artifact = PredictionFeatureEngineeringArtifact(
                features_dataframe=final_prediction_df,
                features_file_path=features_file,
                engineered_features_count=len(final_prediction_df.columns),
                records_count=len(final_prediction_df),
                is_engineering_successful=True,
                message=f"Prediction feature engineering completed successfully. {len(final_prediction_df)} predictions ready ({unique_machines} machines)"
            )
            
            logging.info("Prediction feature engineering completed")
            return artifact
            
        except Exception as e:
            logging.exception(f"Error in prediction feature engineering: {e}")
            raise MyException(e, sys)
