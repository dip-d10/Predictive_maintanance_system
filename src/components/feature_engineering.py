import os
import sys

import pandas as pd

from src.constants import SCHEMA_FILE_PATH
from src.entity.artifact_entity import FeatureEngineeringArtifact
from src.entity.config_entity import FeatureEngineeringConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.schema = read_yaml(SCHEMA_FILE_PATH)
        self.machine_id_column = config.machine_id_column
        self.datetime_column = config.datetime_column
        self.failure_column = config.failure_column
        self.telemetry_columns = list(config.telemetry_columns)
        self.lag_features = list(config.lag_features)
        self.rolling_windows = list(config.rolling_windows)
        self.prediction_horizons = list(config.prediction_horizons or [24])

    def _sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        return df.sort_values(
            by=[self.machine_id_column, self.datetime_column]
        ).reset_index(drop=True)

    def create_machine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._sort_dataframe(df)

        if "age" in df.columns:
            df["machine_age"] = df["age"]
        elif "manufacture_year" in df.columns:
            df["machine_age"] = pd.to_datetime(df[self.datetime_column]).dt.year - df["manufacture_year"]

        if "model" in df.columns:
            model_dummies = pd.get_dummies(df["model"], prefix="model")
            df = pd.concat([df, model_dummies], axis=1)

        for column in self.telemetry_columns:
            if column not in df.columns:
                raise KeyError(f"Missing telemetry column: {column}")
            df[f"current_{column}"] = df[column]

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        grouped = df.groupby(self.machine_id_column, sort=False)

        for column in self.telemetry_columns:
            for lag in self.lag_features:
                df[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for column in self.telemetry_columns:
            grouped = df.groupby(self.machine_id_column, sort=False)[column]

            for window in self.rolling_windows:
                df[f"{column}_rolling_mean_{window}"] = grouped.transform(
                    lambda series, window=window: series.rolling(window=window).mean()
                )
                df[f"{column}_rolling_std_{window}"] = grouped.transform(
                    lambda series, window=window: series.rolling(window=window).std()
                )

        return df

    def create_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for column in self.telemetry_columns:
            if f"{column}_lag_1" in df.columns:
                df[f"{column}_delta_1"] = df[column] - df[f"{column}_lag_1"]

            if f"{column}_lag_3" in df.columns:
                df[f"{column}_delta_3"] = df[column] - df[f"{column}_lag_3"]

        return df

    def create_maintenance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        maintenance_columns = [column for column in df.columns if column.startswith("maint_")]

        if not maintenance_columns:
            df["time_since_last_maintenance"] = pd.NA
            df["maintenance_count_last_30_days"] = pd.NA
            df["component_replaced_last_30_days"] = pd.NA
            return df

        maintenance_flag = df[maintenance_columns].fillna(0).sum(axis=1) > 0
        df["_maintenance_flag"] = maintenance_flag.astype(int)
        df["_maintenance_datetime"] = df[self.datetime_column].where(maintenance_flag)
        df["_maintenance_datetime"] = df.groupby(self.machine_id_column)["_maintenance_datetime"].ffill()
        df["time_since_last_maintenance"] = (
            pd.to_datetime(df[self.datetime_column]) - pd.to_datetime(df["_maintenance_datetime"])
        ).dt.total_seconds() / 3600

        grouped_flag = df.groupby(self.machine_id_column, sort=False)["_maintenance_flag"]
        df["maintenance_count_last_30_days"] = grouped_flag.transform(
            lambda series: series.rolling(window=720, min_periods=1).sum()
        )
        df["component_replaced_last_30_days"] = grouped_flag.transform(
            lambda series: series.rolling(window=720, min_periods=1).sum()
        )

        df.drop(columns=["_maintenance_flag", "_maintenance_datetime"], inplace=True)
        return df

    def create_error_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        error_columns = [column for column in df.columns if column.startswith("errorID_")]

        if not error_columns:
            df["error_count_last_24h"] = pd.NA
            df["error_count_last_7d"] = pd.NA
            df["recent_error_type"] = "no_error"
            return df

        error_flag = df[error_columns].fillna(0).sum(axis=1) > 0
        df["_error_flag"] = error_flag.astype(int)
        df["error_count_last_24h"] = df.groupby(self.machine_id_column, sort=False)["_error_flag"].transform(
            lambda series: series.rolling(window=24, min_periods=1).sum()
        )
        df["error_count_last_7d"] = df.groupby(self.machine_id_column, sort=False)["_error_flag"].transform(
            lambda series: series.rolling(window=168, min_periods=1).sum()
        )

        recent_error_type = pd.Series(index=df.index, dtype="object")
        current_error_mask = error_flag
        if current_error_mask.any():
            recent_error_type.loc[current_error_mask] = df.loc[current_error_mask, error_columns].idxmax(axis=1)

        recent_error_type = recent_error_type.ffill()
        recent_error_type = recent_error_type.fillna("no_error")
        df["recent_error_type"] = recent_error_type

        df.drop(columns=["_error_flag"], inplace=True)
        return df

    def create_failure_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.failure_column not in df.columns:
            df["prior_failure_count"] = pd.NA
            df["time_since_last_failure"] = pd.NA
            return df

        df["_failure_flag"] = df[self.failure_column].fillna(0).astype(int)
        df["prior_failure_count"] = df.groupby(self.machine_id_column, sort=False)["_failure_flag"].cumsum().shift(1).fillna(0)

        df["_failure_datetime"] = df[self.datetime_column].where(df["_failure_flag"] > 0)
        df["_failure_datetime"] = df.groupby(self.machine_id_column)["_failure_datetime"].ffill()
        df["time_since_last_failure"] = (
            pd.to_datetime(df[self.datetime_column]) - pd.to_datetime(df["_failure_datetime"])
        ).dt.total_seconds() / 3600

        df.drop(columns=["_failure_flag", "_failure_datetime"], inplace=True)
        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.failure_column not in df.columns:
            raise KeyError(f"Missing failure column: {self.failure_column}")

        df[self.failure_column] = df[self.failure_column].fillna(0).astype(int)

        grouped_failure = df.groupby(self.machine_id_column, sort=False)[self.failure_column]
        for horizon in self.prediction_horizons:
            target_column = f"failure_within_next_{horizon}h"
            df[target_column] = grouped_failure.transform(
                lambda series, horizon=horizon: series.shift(-horizon).rolling(horizon).max()
            )

        return df

    def handle_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["recent_error_type"] = df["recent_error_type"].fillna("no_error")
        df = df.dropna().reset_index(drop=True)
        return df

    def initiate_feature_engineering(self):
        try:
            logging.info("Starting feature engineering...")

            df = pd.read_csv(self.config.master_data_path)

            df = self.create_machine_features(df)
            df = self.create_lag_features(df)
            df = self.create_rolling_features(df)
            df = self.create_delta_features(df)
            df = self.create_maintenance_features(df)
            df = self.create_error_features(df)
            df = self.create_failure_history_features(df)
            df = self.create_target(df)
            df = self.handle_null_values(df)

            os.makedirs(os.path.dirname(self.config.feature_store_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.final_feature_path), exist_ok=True)

            df.to_csv(self.config.feature_store_path, index=False)
            df.to_csv(self.config.final_feature_path, index=False)

            logging.info("Feature engineering completed successfully.")

            return FeatureEngineeringArtifact(
                feature_store_path=self.config.feature_store_path,
                final_feature_path=self.config.final_feature_path,
                is_engineering_successful=True,
                message="Feature engineering completed successfully",
            )

        except Exception as e:
            logging.exception(f"Feature engineering failed: {e}")
            raise MyException(e, sys)