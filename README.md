# MLOps Kubeflow & MLflow Assignment

## Project Overview
This project demonstrates an end-to-end MLOps workflow using **Kubeflow Pipelines**, **MLflow**, **DVC**, and **Jenkins**. The project focuses on the **Boston Housing dataset**, performing regression to predict housing prices (`MEDV`) based on multiple features.  

The workflow includes:  
1. **Data Extraction**: Fetch versioned data from DVC remote storage.  
2. **Data Preprocessing**: Clean, scale, and split the dataset into training and testing sets.  
3. **Model Training**: Train a Random Forest Regressor using the processed data.  
4. **Model Evaluation**: Evaluate the model and log metrics (MSE) using MLflow.  
5. **CI/CD**: Automate pipeline execution using Jenkins.  

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ayan-create/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Install Python Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Setup DVC Remote Storage
1. Ensure you have DVC installed:
```bash
pip install dvc
```
2. Pull versioned dataset:
```bash
dvc pull
```

### 4. (Optional) Minikube & Kubeflow Setup
> **Note:** For this assignment, MLflow is used for pipeline execution instead of Minikube.  
If you want to use Kubeflow Pipelines locally:
1. Install [Minikube](https://minikube.sigs.k8s.io/docs/start/).  
2. Deploy [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/installation/).  

---

## Pipeline Walkthrough

### 1. Compile Kubeflow Pipeline
```bash
python pipeline.py
```
This generates YAML files for each pipeline component in `components/`.

### 2. Run MLflow Pipeline
```bash
python scripts/mlflow_pipeline.py
```
- The pipeline executes all stages: data extraction, preprocessing, model training, and evaluation.  
- MLflow logs metrics, parameters, and model artifacts under `mlruns/`.  

### 3. Run via Jenkins
- Ensure Jenkins is installed and connected to this repository.  
- Jenkinsfile stages:
  1. Checkout
  2. Environment Setup
  3. Data Setup (DVC Pull)
  4. Pipeline Compilation
  5. Trigger MLflow Pipeline
- Execute the pipeline manually or via webhook.  
- Pipeline logs and artifacts will be available in MLflow.  

---

## Project Structure
```
.
├─ components/          # Kubeflow component YAMLs
├─ data/                # Raw and processed data
├─ dvc_remote/          # DVC remote cache
├─ mlruns/              # MLflow experiment runs
├─ models/              # Trained model artifacts
├─ scripts/             # MLflow pipeline scripts & helper scripts
├─ src/                 # Python modules for components
├─ Dockerfile           # Dockerfile (if needed for components)
├─ Jenkinsfile          # Jenkins CI/CD pipeline definition
├─ requirements.txt     # Python dependencies
├─ pipeline.py          # Pipeline compiler script (generates YAML)
└─ README.md            # Project documentation
```

---

## References
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)  
- [MLflow](https://mlflow.org/)  
- [DVC](https://dvc.org/)  
- [Jenkins Pipeline](https://www.jenkins.io/doc/book/pipeline/)

---

## Deliverables
1. Fully functional ML pipeline with MLflow logging.  
2. CI/CD automation with Jenkins.  
3. Screenshots of:
   - Jenkins pipeline run console output  
   - Repository main page with README.md  
4. GitHub repository URL: `<your-github-repo-url>`

