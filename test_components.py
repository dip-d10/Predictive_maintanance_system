from src.configuration.config_manager import ConfigurationManager
from src.components.raw_data_validation import RawDataValidation


STAGE_NAME = "RAW DATA VALIDATION STAGE"

try:
    print(f">>>>>> {STAGE_NAME} started <<<<<<")

    config = ConfigurationManager()

    validation_config = (
        config.get_raw_data_validation_config()
    )

    validation = RawDataValidation(
        config=validation_config
    )

    validation.validate_all_files()

    print(f">>>>>> {STAGE_NAME} completed <<<<<<")

except Exception as e:
    print(e)