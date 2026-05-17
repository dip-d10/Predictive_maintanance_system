# Predictive Maintenance MLOps System 

This project is an end-to-end **Machine Learning + MLOps system** built to predict whether an industrial machine is likely to fail so maintenance teams can take action before breakdown happens.

The system simulates a real-world production ML workflow with automated retraining, batch prediction, cloud model storage, and scheduled prediction jobs.

---

## Business Problem

In manufacturing industries, unexpected machine failures can cause:

- Production downtime  
- Revenue loss  
- High repair costs  
- Poor maintenance planning  

This project helps companies move from reactive maintenance to **predictive maintenance** using machine sensor data.

---

## Problem Statement

The model predicts:

- **1 → Machine likely to fail**
- **0 → Machine healthy**

This allows teams to identify high-risk machines before actual failure occurs.

---

## Dataset Used

Microsoft Predictive Maintenance Dataset

Files used:

- `PdM_machines.csv`
- `PdM_maint.csv`
- `PdM_errors.csv`
- `PdM_failures.csv`

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- MongoDB Atlas  
- Microsoft Azure Blob Storage  
- Git/GitHub  
- Cron Jobs / Scheduling  

---

## Project Workflow

```bash
Data Ingestion
→ Data Validation
→ Feature Engineering
→ Model Training
→ Model Evaluation
→ Model Storage (Azure)
→ Batch Prediction
→ Prediction Storage (MongoDB)
→ Automated Retraining
```

---

## Key Features

- Built modular ML pipeline architecture  
- Engineered **79 predictive features** using rolling windows, lag features, and machine history trends  
- Automated weekly retraining pipeline  
- Hourly prediction pipeline for real-time monitoring simulation  
- Batch prediction system for latest machine health status  
- Stores predictions in MongoDB Atlas  
- Stores production models in Azure Blob Storage  
- Detailed logging for debugging and monitoring  

---

## Architecture Overview

```bash
Raw Machine Data
      ↓
Data Pipeline
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Azure Model Storage
      ↓
Hourly Predictions
      ↓
MongoDB Prediction Storage
```

---

## Project Structure

```bash
predictive-maintenance/
│
├── notebook/
├── src/
├── pipeline/
├── artifacts/
├── config/
├── training_pipeline.py
├── batch_prediction_pipeline.py
├── hourly_prediction_job.py
├── weekly_training_job.py
└── requirements.txt
```

---

## How to Run

```bash
git clone <your-repo-link>
cd predictive-maintenance

pip install -r requirements.txt

python training_pipeline.py
```

Run prediction pipeline:

```bash
python hourly_prediction_job.py
```

---

## Current Status

✅ Training pipeline completed  
✅ Feature engineering completed  
✅ Azure model storage completed  
✅ Batch prediction completed  
✅ Hourly prediction scheduling completed  
✅ Weekly retraining completed  

> Project is still improving with better model experimentation and monitoring features.

---

## Future Improvements

- MLflow experiment tracking  
- Hyperparameter tuning  
- FastAPI deployment  
- Docker setup  
- CI/CD pipeline  
- Monitoring dashboard  

---

## Author

**Sumanta Jyoti**  
Aspiring Data Scientist | ML Engineer
