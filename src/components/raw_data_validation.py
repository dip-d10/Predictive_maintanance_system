import os
import yaml
import pandas as pd

from src.entity.config_entity import RawDataValidationConfig
from src.entity.artifact_entity import RawDataValidationArtifact
from src.utils.main_utils import read_yaml
from src.logger import logging


class RawDataValidation:

    def __init__(self, config: RawDataValidationConfig):
        self.config = config
        self.schema = read_yaml(config.schema_file_path)

    def validate_all_files(self):

        validation_results = {}

        try:
            for dataset_name, dataset_schema in self.schema.items():

                file_path = os.path.join(
                    self.config.raw_data_dir,
                    f"{dataset_name}.csv"
                )

                # file existence check
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"{dataset_name}.csv not found"
                    )

                df = pd.read_csv(file_path)

                # schema validation
                expected_columns = list(
                    dataset_schema["columns"].keys()
                )

                actual_columns = list(df.columns)

                if expected_columns != actual_columns:
                    raise ValueError(
                        f"{dataset_name}: schema mismatch"
                    )

                # null validation
                if df.isnull().sum().sum() > 0:
                    raise ValueError(
                        f"{dataset_name}: contains null values"
                    )

                # duplicate validation
                if df.duplicated().sum() > 0:
                    raise ValueError(
                        f"{dataset_name}: contains duplicates"
                    )

                validation_results[dataset_name] = "passed"

                logging.info(
                    f"{dataset_name} validation passed"
                )

            report = {
                "validation_status": True,
                "datasets": validation_results
            }

        except Exception as e:

            report = {
                "validation_status": False,
                "error": str(e)
            }

            logging.exception(
                "Raw data validation failed"
            )

            raise e

        finally:
            report_path = os.path.join(
                self.config.root_dir,
                "validation_report.yaml"
            )

            with open(report_path, "w") as file:
                yaml.dump(report, file)

            logging.info(
                f"Validation report saved at {report_path}"
            )

        logging.info("All datasets validated successfully")

        return RawDataValidationArtifact(
            validation_status=report["validation_status"],
            validation_report_path=report_path
        )