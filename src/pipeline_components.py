import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


# ---------------------------
# COMPONENT 1 — DATA EXTRACTION
# ---------------------------

@component(
    base_image="python:3.11",
    output_component_file="components/extract_data.yaml",
)
def extract_data(raw_data_path: Input[Dataset], extracted_output: Output[Dataset]):
    """
    Reads the versioned dataset from the DVC-tracked file.
    Copies it into the component output path.
    """
    import shutil
    shutil.copy(raw_data_path.path, extracted_output.path)


# ---------------------------
# COMPONENT 2 — DATA PREPROCESSING
# ---------------------------

@component(
    base_image="python:3.11",
    output_component_file="components/preprocess_data.yaml",
)
def preprocess_data(
    extracted_input: Input[Dataset],
    processed_data: Output[Dataset]
):
    """
    Loads raw CSV, cleans, scales, and splits into train/test.
    Saves processed data into a new CSV.
    """
    df = pd.read_csv(extracted_input.path)

    # Example target column "MEDV"
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    processed_df = pd.DataFrame(X_train)
    processed_df["label"] = y_train.values
    processed_df.to_csv(processed_data.path, index=False)


# ---------------------------
# COMPONENT 3 — MODEL TRAINING
# ---------------------------

@component(
    base_image="python:3.11",
    output_component_file="components/train_model.yaml",
)
def train_model(processed_data: Input[Dataset], model_output: Output[Model]):
    """
    Train Random Forest on processed data.
    """
    df = pd.read_csv(processed_data.path)

    X_train = df.drop("label", axis=1)
    y_train = df["label"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_output.path)


# ---------------------------
# COMPONENT 4 — MODEL EVALUATION
# ---------------------------

@component(
    base_image="python:3.11",
    output_component_file="components/evaluate_model.yaml",
)
def evaluate_model(
    model_input: Input[Model],
    extracted_input: Input[Dataset],
    metrics_output: Output[Metrics]
):
    """
    Loads model and evaluates using test split.
    """
    df = pd.read_csv(extracted_input.path)

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(model_input.path)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    metrics_output.log_metric("mse", mse)
