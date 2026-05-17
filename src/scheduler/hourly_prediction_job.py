"""
Hourly Batch Prediction Job
=============================

Schedule: Every hour (00 * * * *)
Trigger: Azure Container Instances / Kubernetes CronJob

Responsibilities:
- Fetch latest telemetry data
- Engineer features
- Load production model
- Generate predictions
- Store predictions in MongoDB
- Trigger maintenance alerts

To use with Azure Container Instances (ACI):
    az container create \
        --resource-group <rg> \
        --name hourly-prediction-job \
        --image <container-image> \
        --command-line "python src/scheduler/hourly_prediction_job.py" \
        --schedule "0 * * * *"  # Every hour

To use with Kubernetes CronJob:
    See: kubernetes/hourly_prediction_cronjob.yaml

To use with Azure Functions:
    See: azure_functions/prediction_timer_trigger/__init__.py
"""

import sys
from datetime import datetime

from src.pipeline.batch_prediction_pipeline import BatchPredictionPipeline
from src.logger import logging
from src.exception import MyException


def execute_hourly_prediction_job():
    """
    Execute hourly batch prediction pipeline
    Scheduled to run every hour automatically
    """
    try:
        start_time = datetime.utcnow()

        logging.info("=" * 80)
        logging.info("Step 1/6: Prediction Data Ingestion")
        logging.info("=" * 80)
        pipeline = BatchPredictionPipeline()
        data_ingestion_artifact = pipeline.start_prediction_data_ingestion()
        logging.info("[OK] Data Ingestion Complete")

        logging.info("=" * 80)
        logging.info("Step 2/6: Prediction Feature Engineering")
        logging.info("=" * 80)
        feature_engineering_artifact = pipeline.start_prediction_feature_engineering(data_ingestion_artifact)
        logging.info("[OK] Feature Engineering Complete")

        logging.info("=" * 80)
        logging.info("Step 3/6: Model Loading")
        logging.info("=" * 80)
        model_loader_artifact = pipeline.start_model_loading()
        logging.info("[OK] Model Loading Complete")

        logging.info("=" * 80)
        logging.info("Step 4/6: Prediction")
        logging.info("=" * 80)
        prediction_artifact = pipeline.start_prediction(
            feature_engineering_artifact,
            model_loader_artifact
        )
        logging.info("[OK] Prediction Complete")

        logging.info("=" * 80)
        logging.info("Step 5/6: Prediction Storage")
        logging.info("=" * 80)
        storage_artifact = pipeline.start_prediction_storage(prediction_artifact)
        logging.info("[OK] Prediction Storage Complete")

        logging.info("=" * 80)
        logging.info("Step 6/6: Alert Management")
        logging.info("=" * 80)
        alert_artifact = pipeline.start_alert_manager(prediction_artifact)
        logging.info("[OK] Alert Management Complete")

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        logging.info("=" * 80)
        logging.info("[OK] HOURLY PREDICTION JOB COMPLETED SUCCESSFULLY")
        logging.info(f"Predictions: {prediction_artifact.predictions_count}")
        logging.info(f"Alerts: {alert_artifact.alerts_triggered}")
        logging.info(f"Duration: {duration:.2f}s")
        logging.info("=" * 80)

        return {
            "pipeline_status": "SUCCESS",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "predictions_generated": prediction_artifact.predictions_count,
            "maintenance_alerts": alert_artifact.alerts_triggered
        }
        
    except Exception as e:
        logging.exception(f"HOURLY PREDICTION JOB FAILED: {e}")
        logging.info("=" * 80)
        logging.info("HOURLY PREDICTION JOB FAILED")
        logging.info(f"   Error: {str(e)}")
        logging.info("=" * 80)
        raise MyException(e, sys)


if __name__ == "__main__":
    """
    Entry point for hourly prediction job
    Called by scheduler (Azure Container Instances, Kubernetes CronJob, etc.)
    """
    try:
        result = execute_hourly_prediction_job()
        sys.exit(0)  # Success
    except Exception as e:
        logging.exception(f"Fatal error in hourly prediction job: {e}")
        sys.exit(1)  # Failure
