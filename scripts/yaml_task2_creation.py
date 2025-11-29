import sys
import os

# Add the 'src' folder to Python path so we can import pipeline_components
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from pipeline_components import (
    extract_data,
    preprocess_data,
    train_model,
    evaluate_model
)

if __name__ == "__main__":
    """
    Running this script triggers the YAML generation specified
    in each component's `output_component_file` decorator.
    """
    # Accessing component_spec triggers YAML creation
    extract_data.component_spec
    preprocess_data.component_spec
    train_model.component_spec
    evaluate_model.component_spec

    print("✅ All component YAMLs are generated in the 'components/' folder.")
