pipeline {
    agent any

    environment {
        PYTHON_BIN = "python"   // Windows usually uses "python" instead of "python3"
        ML_PIPELINE_SCRIPT = "scripts\\mlflow_pipeline.py"
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                echo 'Setting up Python environment and installing dependencies...'
                bat "${env.PYTHON_BIN} -m pip install --upgrade pip"
                bat "${env.PYTHON_BIN} -m pip install -r requirements.txt"
            }
        }

        stage('Data Setup') {
            steps {
                echo 'Pulling DVC-tracked dataset...'
                bat 'dvc pull'
            }
        }

        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow pipeline to YAML...'
                bat "${env.PYTHON_BIN} pipeline.py"
            }
        }

        stage('Trigger MLflow Pipeline') {
            steps {
                echo 'Running MLflow pipeline...'
                bat "${env.PYTHON_BIN} ${env.ML_PIPELINE_SCRIPT}"
            }
        }
    }

    post {
        success {
            echo 'Jenkins pipeline completed successfully!'
        }
        failure {
            echo 'Jenkins pipeline failed. Check the console output for errors.'
        }
    }
}
