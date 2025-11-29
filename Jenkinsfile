pipeline {
    agent any

    environment {
        PYTHON = "python3.11" // Adjust if your Jenkins node uses a different Python
    }

    stages {
        stage('Checkout Code') {
            steps {
                // Clone the repository
                git branch: 'main', url: 'https://github.com/ayan-create/mlops-kubeflow-assignment.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Compile Kubeflow Pipeline') {
            steps {
                // Optional: run a Python script that ensures pipeline.py compiles without errors
                sh '''
                    python -m py_compile pipeline.py
                    echo "Kubeflow pipeline compiled successfully."
                '''
            }
        }

        stage('Run MLflow Pipeline') {
            steps {
                sh '''
                    mkdir -p models
                    python scripts/mlflow_pipeline.py
                '''
            }
        }
    }

    post {
        always {
            echo "Pipeline finished. Check console output for logs."
        }
        success {
            echo "Pipeline completed successfully!"
        }
        failure {
            echo "Pipeline failed. Check logs for details."
        }
    }
}
