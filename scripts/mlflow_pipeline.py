
# import sys
# import os
# import pandas as pd

# # Add 'src' folder to Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# import mlflow
# import mlflow.sklearn

# from pipeline_components import (
#     extract_data_mlflow,
#     preprocess_data_mlflow,
#     train_model_mlflow,
#     evaluate_model_mlflow
# )

# # ---------------------------
# # CONFIGURATION
# # ---------------------------
# RAW_DATA = "data/raw_data.csv"
# EXTRACTED_DATA = "data/extracted_data.csv"
# PROCESSED_DATA = "data/processed_data.csv"
# MODEL_PATH = "models/rf_model.pkl"

# # Ensure directories exist
# os.makedirs(os.path.dirname(RAW_DATA), exist_ok=True)
# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# # ---------------------------
# # PIPELINE EXECUTION
# # ---------------------------
# if __name__ == "__main__":
#     # Initialize MLflow experiment
#     mlflow.set_experiment("Boston_Housing_Pipeline")

#     with mlflow.start_run(run_name="RF_MLflow_Run"):

#         # Step 1: Extract data
#         print("Extracting data...")
#         extract_data_mlflow(RAW_DATA, EXTRACTED_DATA)

#         # Step 2: Preprocess data
#         print("Preprocessing data...")
#         X_train, X_test, y_train, y_test, feature_columns = preprocess_data_mlflow(
#             EXTRACTED_DATA, PROCESSED_DATA
#         )

#         # Step 3: Train model
#         print("Training model...")
#         model = train_model_mlflow(PROCESSED_DATA, MODEL_PATH)

#         # Step 4: Evaluate model
#         print("Evaluating model...")
#         mse = evaluate_model_mlflow(model, X_test, y_test, feature_columns)
#         print(f"MSE: {mse}")

#         # Step 5: Log metrics and model to MLflow
#         mlflow.log_metric("mse", mse)
#         mlflow.sklearn.log_model(model, name="random_forest_model")

#         print("Run complete. Metrics and model logged.")
import sys
import os
import pandas as pd

# Add 'src' folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import mlflow
import mlflow.sklearn

from pipeline_components import (
    extract_data_mlflow,
    preprocess_data_mlflow,
    train_model_mlflow,
    evaluate_model_mlflow
)

# ---------------------------
# CONFIGURATION
# ---------------------------
RAW_DATA = "data/raw_data.csv"
EXTRACTED_DATA = "data/extracted_data.csv"
PROCESSED_DATA = "data/processed_data.csv"
MODEL_PATH = "models/rf_model.pkl"

# Ensure directories exist
os.makedirs(os.path.dirname(RAW_DATA), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ---------------------------
# PIPELINE EXECUTION
# ---------------------------
if __name__ == "__main__":
    # Initialize MLflow experiment
    mlflow.set_experiment("Boston_Housing_Pipeline")

    with mlflow.start_run(run_name="RF_MLflow_Run"):

        # Step 1: Extract data
        print("Extracting data...")
        extract_data_mlflow(RAW_DATA, EXTRACTED_DATA)

        # Step 2: Preprocess data
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, feature_columns = preprocess_data_mlflow(
            EXTRACTED_DATA, PROCESSED_DATA
        )

        # Step 3: Train model
        print("Training model...")
        model = train_model_mlflow(PROCESSED_DATA, MODEL_PATH)

        # Step 4: Evaluate model
        print("Evaluating model...")
        mse = evaluate_model_mlflow(model, X_test, y_test, feature_columns)
        print(f"MSE: {mse}")

        # Step 5: Log metrics and model to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, name="random_forest_model")

        print("Run complete. Metrics and model logged.")
