pipeline {
    agent any

    environment {
        PYTHON = "python" // On Windows, likely just 'python' or 'python3.11' if installed
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
                bat """
                    %PYTHON% -m pip install --upgrade pip
                    %PYTHON% -m pip install -r requirements.txt
                """
            }
        }

        stage('Compile Kubeflow Pipeline') {
            steps {
                bat """
                    %PYTHON% -m py_compile pipeline.py
                    echo Kubeflow pipeline compiled successfully.
                """
            }
        }

        stage('Run MLflow Pipeline') {
            steps {
                bat """
                    if not exist models mkdir models
                    %PYTHON% scripts\\mlflow_pipeline.py
                """
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
