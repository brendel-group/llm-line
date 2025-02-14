import yaml
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class PretrainingData:
    """Information about model's pretraining data."""
    dataset: str
    dataset_version: str
    dataset_size: str

@dataclass
class ModelConfig:
    """Configuration for model evaluation."""
    name: str
    cls: str
    batch_size: int = 8
    device: str = "cuda"
    architecture: str = "Unknown"
    dataset: str = "Unknown"
    dataset_version: str = "Unknown"
    dataset_size: str = "Unknown"
    max_length: Optional[int] = None
    revision: Optional[str] = None
    tokenizer_name: Optional[str] = None  # if not specified, will default to the model name

    @classmethod
    def from_dict(cls, data: Dict):
        """Create ModelConfig from dictionary."""
        # Copy the data to avoid modifying the original
        config_data = data.copy()
        return cls(**config_data)

@dataclass
class ValidationConfig:
    """Configuration for validation dataset."""
    path: str
    batch_size: int = 16
    max_seq_len: Optional[int] = 1024
    max_steps: Optional[int] = None

@dataclass
class EvalConfig:
    """Configuration for evaluation settings."""
    tasks: Dict
    validation_configs: Dict[str, ValidationConfig]
    output_dir: str = "results"
    save_details: bool = True
    compute_loss: bool = False

def load_yaml(file_path: str):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_model_configs(yaml_path: str) -> List[ModelConfig]:
    data = load_yaml(yaml_path)
    return [ModelConfig.from_dict(model) for model in data['models']]

def get_eval_config(yaml_path: str, output_dir: str, compute_loss: bool = False) -> EvalConfig:
    data = load_yaml(yaml_path)
    
    # Convert validation configs to ValidationConfig objects
    validation_configs = {}
    if 'validation_configs' in data:
        for name, config in data['validation_configs'].items():
            validation_configs[name] = ValidationConfig(**config)
    
    return EvalConfig(
        tasks=data['task_configs'],
        validation_configs=validation_configs,
        output_dir=output_dir,
        compute_loss=compute_loss
    )

def format_model_info(model_config) -> str:
    """Format model info for filename and metadata."""
    dataset = model_config.dataset.replace(' ', '')
    tokens = model_config.dataset_size.split()[0]
    
    filename = (
        f"name={model_config.name.replace('/', '_')}"
        f"__arch={model_config.architecture}"
        f"__dataset={dataset}"
        f"__dataset_version={model_config.dataset_version}"
        f"__size={tokens}"
    )
    
    if model_config.revision:
        filename += f"__checkpoint={model_config.revision}"
    
    return filename

def get_model_metadata(model_config) -> dict:
    """Get model metadata for storing in results."""
    metadata = {
        "model_name": model_config.name,
        "architecture": model_config.architecture,
        "dataset": model_config.dataset,
        "dataset_version": model_config.dataset_version,
        "dataset_size": model_config.dataset_size,
        "batch_size": model_config.batch_size,
        "device": model_config.device
    }
    
    if model_config.revision:
        metadata["checkpoint"] = model_config.revision
        
    return metadata

def parse_args():
    parser = argparse.ArgumentParser(description='LLM Evaluation Framework')
    parser.add_argument('--models_yaml', type=str, default='pile_models.yaml',
                      help='Path to models configuration file')
    parser.add_argument('--tasks_yaml', type=str, default='tasks.yaml',
                      help='Path to tasks configuration file')
    parser.add_argument('--results_dir', type=str, 
                      default='/is/cluster/fast/pmayilvahanan/llm_line/results/',
                      help='Directory to store evaluation results')
    parser.add_argument('--shuffle', action='store_true',
                      help='shuffling the models to evaluate')
    
    # Add evaluation mode group
    eval_mode = parser.add_mutually_exclusive_group()
    eval_mode.add_argument('--accuracy-only', action='store_true',
                          help='Only compute accuracy metrics')
    eval_mode.add_argument('--loss-only', action='store_true',
                          help='Only compute cross-entropy loss')
    eval_mode.add_argument('--compute-both', action='store_true',
                          help='Compute both accuracy and loss metrics (default)')
    
    return parser.parse_args()