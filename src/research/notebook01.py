import os
import pandas as pd
from pymongo import MongoClient

# ---------------------------
# MongoDB Atlas Connection URI
# ---------------------------
MONGO_URI = "mongodb+srv://sumantapa2503mth393_db_user:sumantapa2503mth393_db_user@cluster0.rnbdegu.mongodb.net/?appName=Cluster0"
# Connect to MongoDB Atlas (Cluster0)
client = MongoClient(MONGO_URI)

# Create/select database inside Cluster0
db = client["predictive_maintenance_db"]

print(" Connected to MongoDB Atlas successfully")

# ---------------------------
# Get path of current script
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

# data folder path
data_dir = os.path.join(current_dir, "data")

# ---------------------------
# CSV file mapping
# ---------------------------
files = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
    "maintenance": "PdM_maint.csv"
}

# ---------------------------
# Upload CSVs to MongoDB
# ---------------------------
for collection_name, file_name in files.items():
    file_path = os.path.join(data_dir, file_name)

    print(f"\nUploading {file_name} → {collection_name}")

    # Read CSV
    df = pd.read_csv(file_path)

    print(f"Rows found: {len(df)}")

    # Convert dataframe rows to MongoDB documents
    records = df.to_dict("records")


    # Insert new data
    result = db[collection_name].insert_many(records)

    print(f" Inserted {len(result.inserted_ids)} records into '{collection_name}'")

print("\n All datasets uploaded successfully to Cluster0!")