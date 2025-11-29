
# import pandas as pd
# import joblib
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# # ---------------------------
# # DATA EXTRACTION
# # ---------------------------

# def extract_data_mlflow(raw_data_path: str, extracted_output_path: str):
#     """
#     Copies raw CSV to extracted output path.
#     """
#     import shutil
#     os.makedirs(os.path.dirname(extracted_output_path), exist_ok=True)
#     shutil.copy(raw_data_path, extracted_output_path)

# # ---------------------------
# # DATA PREPROCESSING
# # ---------------------------

# def preprocess_data_mlflow(extracted_input_path: str, processed_output_path: str):
#     df = pd.read_csv(extracted_input_path)

#     # Separate features and target
#     X = df.drop("MEDV", axis=1)
#     y = df["MEDV"]

#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

#     # Ensure output folder exists
#     os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

#     # Save processed training data with original column names
#     processed_df = pd.DataFrame(X_train, columns=X.columns)
#     processed_df["label"] = y_train.values
#     processed_df.to_csv(processed_output_path, index=False)

#     return X_train, X_test, y_train, y_test, X.columns  # return columns for evaluation

# # ---------------------------
# # MODEL TRAINING
# # ---------------------------

# def train_model_mlflow(processed_data_path: str, model_output_path: str):
#     df = pd.read_csv(processed_data_path)

#     X_train = df.drop("label", axis=1)
#     y_train = df["label"]

#     model = RandomForestRegressor(n_estimators=50, random_state=42)
#     model.fit(X_train, y_train)

#     # Ensure model folder exists
#     os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
#     joblib.dump(model, model_output_path)

#     return model

# # ---------------------------
# # MODEL EVALUATION
# # ---------------------------

# def evaluate_model_mlflow(model, X_test, y_test, feature_columns):
#     # Convert X_test back to DataFrame with original column names
#     X_test_df = pd.DataFrame(X_test, columns=feature_columns)
#     preds = model.predict(X_test_df)
#     mse = mean_squared_error(y_test, preds)
#     return mse
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# ---------------------------
# DATA EXTRACTION
# ---------------------------

def extract_data_mlflow(raw_data_path: str, extracted_output_path: str):
    """
    Copies raw CSV to extracted output path.
    """
    import shutil
    os.makedirs(os.path.dirname(extracted_output_path), exist_ok=True)
    shutil.copy(raw_data_path, extracted_output_path)

# ---------------------------
# DATA PREPROCESSING
# ---------------------------

def preprocess_data_mlflow(extracted_input_path: str, processed_output_path: str):
    df = pd.read_csv(extracted_input_path)

    # Separate features and target
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Ensure output folder exists
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

    # Save processed training data with original column names
    processed_df = pd.DataFrame(X_train, columns=X.columns)
    processed_df["label"] = y_train.values
    processed_df.to_csv(processed_output_path, index=False)

    # Return feature columns for evaluation
    return X_train, X_test, y_train, y_test, X.columns

# ---------------------------
# MODEL TRAINING
# ---------------------------

def train_model_mlflow(processed_data_path: str, model_output_path: str):
    df = pd.read_csv(processed_data_path)

    X_train = df.drop("label", axis=1)
    y_train = df["label"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Ensure model folder exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)

    return model

# ---------------------------
# MODEL EVALUATION
# ---------------------------

def evaluate_model_mlflow(model, X_test, y_test, feature_columns):
    # Convert X_test back to DataFrame with original column names
    X_test_df = pd.DataFrame(X_test, columns=feature_columns)
    preds = model.predict(X_test_df)
    mse = mean_squared_error(y_test, preds)
    return mse
