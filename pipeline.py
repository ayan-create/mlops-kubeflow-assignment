from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import Input, Output, Dataset, Model, Metrics

# Import components
from src.pipeline_components import extract_data, preprocess_data, train_model, evaluate_model

@dsl.pipeline(
    name="boston-housing-pipeline",
    description="A simple ML pipeline for Boston Housing using Kubeflow components."
)
def pipeline(raw_data_path: str):
    # Step 1: Extract data
    extract_step = extract_data(raw_data_path=raw_data_path)

    # Step 2: Preprocess data
    preprocess_step = preprocess_data(extracted_input=extract_step.outputs["extracted_output"])

    # Step 3: Train model
    train_step = train_model(processed_data=preprocess_step.outputs["processed_data"])

    # Step 4: Evaluate model
    evaluate_step = evaluate_model(
        model_input=train_step.outputs["model_output"],
        extracted_input=extract_step.outputs["extracted_output"]
    )

if __name__ == "__main__":
    # Compile pipeline to YAML
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml"
    )
