# LLMs on the Line: Data Determines Loss-To-Loss Scaling Laws
Official code to reproduce the results and data presented in the paper [LLMs on the Line: Data Determines Loss-To-Loss Scaling Laws](https://brendel-group.github.io/llm-line/).

<p align="center">
  <img src="https://brendel-group.github.io/llm-line/img/fig1.png"/>
</p>


## Requirements
- Python 3.10
- CUDA 11.7
- CUDNN 8.4.1

## Repository Structure
- `main.py`: Main evaluation script for huggingface models
- `config.py`: Configuration handling and argument parsing
- `utils.py`: Utility classes and functions
- `models.yaml`: Model configurations
- `tasks.yaml`: Task configurations with few-shot settings
- `config/`: Configuration files for different models
- `data/`: Contains a csv file with all evaluation results
- `lingua`: Contains the lingua-huggingface repository with all the changes and config files to train / evaluate models from scratch

## Setup
1. Load required CUDA modules:
```bash
module purge
module load cuda/11.7
```

2. Initialize and update submodules:
```bash
git submodule update --init --recursive
```

3. Create and activate virtual environment:
```bash
python3.10 -m venv ~/.llm_line
source ~/.llm_line/bin/activate
```

4. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration Files
### models.yaml
Specifies models to evaluate with their configurations:
```yaml
models:
  - name: "model/name"
    cls: "hf" # lm_eval argument
    batch_size: 8
    device: "cuda"
```

### tasks.yaml
Defines tasks and their few-shot settings:
```yaml
task_configs:
  commonsense_qa:
    shots: [0, 5, 10]  # Will evaluate with 0, 5, and 10 shots
```

## Usage
Basic usage:
```bash
python main.py
```

With custom paths:
```bash
python main.py \
    --models_yaml path/to/models.yaml \
    --tasks_yaml path/to/tasks.yaml \
    --results_dir custom_results_dir
```

## Arguments
- `--models_yaml`: Path to models configuration file (default: 'models.yaml')
- `--tasks_yaml`: Path to tasks configuration file (default: 'tasks.yaml')
- `--results_dir`: Directory to store evaluation results (default: 'results')

## Troubleshooting
If you encounter CUDA-related errors:
1. Ensure you have the correct CUDA modules loaded (cuda/11.7, cudnn/8.4.1-cu11.6)
2. Try cleaning the virtual environment and reinstalling:
```bash
rm -rf ~/.llm_line
python3.10 -m venv ~/.llm_line
source ~/.llm_line/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Adding New Model Arguments
When adding new arguments to model configurations, several files need to be updated to ensure proper handling:

1. **Model YAML Files** (`models.yaml`, etc.):
   ```yaml
   models:
     - name: "model/name"
       revision: "step-123"  # New argument
       cls: "hf"
       batch_size: 8
       device: "cuda"
   ```

2. **Configuration Class** (`config.py`):
   - Add the new field to `ModelConfig` class
   - Update `from_dict` method if special handling is needed
   ```python
   @dataclass
   class ModelConfig:
       name: str
       revision: Optional[str] = None  # New field
   ```

3. **Filename Generation** (`config.py`):
   - Update `format_model_info` to include new args in filenames
   ```python
   def format_model_info(model_config):
       filename = f"name={model_config.name}"
       if model_config.revision:
           filename += f"__checkpoint={model_config.revision}"
   ```

4. **Metadata Storage** (`config.py`):
   - Update `get_model_metadata` to include new args in results
   ```python
   def get_model_metadata(model_config):
       metadata = {
           "model_name": model_config.name,
           # Add new fields
           "checkpoint": model_config.revision if model_config.revision else None
       }
   ```

5. **Model Loading** (`main.py`):
   - Update model loading logic if the new argument affects how models are loaded
   ```python
   model_args = f"pretrained={model_config.name}"
   if model_config.revision:
       model_args += f",revision={model_config.revision}"
   ```

This ensures that:
- New arguments are properly parsed from YAML
- Arguments are included in result filenames for tracking
- Metadata in result files includes new information
- Model loading uses new arguments correctly