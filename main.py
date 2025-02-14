from typing import Dict, List, Tuple, Optional
from lm_eval import evaluator, models  # type: ignore
import torch  # type: ignore
import json
from pathlib import Path
from config import get_model_configs, get_eval_config, parse_args, format_model_info, get_model_metadata
import random
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cross_entropy
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from based.based.models.gpt import GPTLMHeadModel



datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def load_existing_results(results_path: Path) -> Dict:
    """Load existing results if they exist."""
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}

def compute_validation_loss(model_config, jsonl_path: str, batch_size: int, max_seq_len: Optional[int] = None, max_steps: Optional[int] = None) -> float:
    """Compute average negative log-likelihood per token on input text."""
    # Load model and tokenizer
    print(f"Loading model {model_config.name}...")
    tokenizer_name = model_config.tokenizer
    
    # Special handling for Mamba models
    if model_config.cls == "mamba_ssm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or "EleutherAI/gpt-neox-20b",  # Use provided tokenizer or default
            trust_remote_code=True
        )
        # Import and use Mamba-specific model loading
        model = MambaLMHeadModel.from_pretrained(
            model_config.name).to(model_config.device)
    elif 'hazyresearch' in model_config.name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or "gpt2")
        model = GPTLMHeadModel.from_pretrained_hf(model_config.name).to(model_config.device)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name,
            revision=model_config.revision,
            trust_remote_code=True
        ).to(model_config.device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_config.name,
            revision=model_config.revision,
            trust_remote_code=True
        )
    
    print(f"tokenizer_name: {tokenizer_name}")
    print(f"tokenizer: {tokenizer}")
    # Get model's maximum sequence length
    model_max_length = getattr(model.config, 'max_position_embeddings', max_seq_len)
    if model_config.max_length:
        model_max_length = min(model_max_length, model_config.max_length)
    
    total_loss = 0
    total_tokens = 0
    steps = 0
    
    try:
        # Process file line by line
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Processing validation file"):
                if max_steps and steps >= max_steps:
                    break
                    
                if not line.strip():
                    continue
                
                # Parse JSON line and get text
                data = json.loads(line)
                text = data['text']
                
                # Tokenize with truncation
                tokens = tokenizer(text, truncation=True, 
                                 max_length=model_max_length, return_tensors="pt")
                # Get device from model parameters instead of model.device
                device = next(model.parameters()).device
                input_ids = tokens.input_ids.to(device)
                
                # Process in batches
                for i in range(0, input_ids.size(1) - 1, batch_size):
                    batch_input = input_ids[:, i:i + batch_size]
                    if batch_input.size(1) < 2:
                        continue
                        
                    with torch.no_grad():
                        outputs = model(batch_input)
                        logits = outputs.logits
                        
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch_input[:, 1:].contiguous()
                    
                    loss = cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_tokens += shift_labels.numel()
                
                steps += 1
    finally:
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

def get_pending_evaluations(task_configs: Dict, 
                          validation_configs: Dict,
                          existing_results: Dict,
                          eval_mode: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Determine which evaluations need to be run."""
    pending_task_evals = []
    pending_validation_evals = []
    
    # Check regular tasks
    for task_name, task_config in task_configs.items():
        for num_shot in task_config['shots']:
            eval_key = f"{task_name}_{num_shot}_shot"
            if eval_key not in existing_results or \
               (eval_mode in ['accuracy-only', 'compute-both'] and 'accuracy' not in existing_results[eval_key]) or \
               (eval_mode in ['loss-only', 'compute-both'] and 'loss' not in existing_results[eval_key]):
                pending_task_evals.append((task_name, num_shot))
    
    # Check validation datasets
    if validation_configs:
        for val_name in validation_configs:
            eval_key = f"validation_{val_name}"
            if eval_key not in existing_results or 'loss' not in existing_results[eval_key]:
                pending_validation_evals.append(val_name)
    
    return pending_task_evals, pending_validation_evals

def evaluate_model(model_config, task_name: str, num_shot: int, eval_mode: str, tokenizer_name: Optional[str] = None) -> dict:
    """Evaluate a model using specified metrics."""
    results = {}
    
    # Adjust batch size based on number of shots
    adjusted_batch_size = model_config.batch_size // max(1, (num_shot // 5) * 5)
    adjusted_batch_size = max(4, adjusted_batch_size)
    
    model_args = f"pretrained={model_config.name}"
    if model_config.revision:
        model_args += f",revision={model_config.revision}"
    
    model_args += ",trust_remote_code=True"
    
    # Add tokenizer configuration for Mamba models
    if model_config.cls == "mamba_ssm":
        model_args += f",tokenizer={tokenizer_name or 'EleutherAI/gpt-neox-20b'}"
    elif 'hazyresearch' in model_config.name:
        model_args += f",tokenizer={tokenizer_name or 'gpt2'}"
    else:
        if tokenizer_name:
            model_args += f",tokenizer={tokenizer_name}"

    
    print(f"model_args: {model_args}")
    # Determine if we need to log samples (for loss computation)
    need_samples = eval_mode in ['loss-only', 'compute-both']
    
    # Single evaluation call
    eval_results = evaluator.simple_evaluate(
        model=model_config.cls,
        model_args=model_args,
        tasks=[task_name],
        num_fewshot=num_shot,
        batch_size=adjusted_batch_size,
        log_samples=need_samples)    
    # Extract accuracy if needed
    if eval_mode in ['accuracy-only', 'compute-both']:
        # results['accuracy'] = eval_results['results'][task_name]
        results['accuracy'] = eval_results['results'] # new addition
    
    # Compute loss if needed
    if need_samples:
        # Handle MMLU subtasks separately
        if 'mmlu' in task_name:
            results['loss'] = {}
            total_weighted_loss = 0.0
            total_samples = 0
            
            for subtask, samples in eval_results['samples'].items():
                total_loss = 0.0
                num_samples = 0
                
                for sample in samples:
                    if 'resps' in sample and 'doc' in sample:
                        target_idx = int(sample['doc']['answer'])
                        log_probs = [-resp[0][0] for resp in sample['resps']]
                        
                        # if num_samples == 3:
                        #     break
                            
                        total_loss += log_probs[target_idx]
                        num_samples += 1
                
                subtask_loss = total_loss / num_samples if num_samples > 0 else float('inf')
                results['loss'][subtask] = subtask_loss
                
                if subtask_loss != float('inf'):
                    total_weighted_loss += subtask_loss * num_samples
                    total_samples += num_samples
            
            # Add weighted average loss
            results['loss']['mmlu'] = total_weighted_loss / total_samples if total_samples > 0 else float('inf')
        else:
            # Original code for non-MMLU tasks
            task_dict = eval_results['samples'][task_name]
            total_loss = 0.0
            num_samples = 0
            
            # Get tokenizer for computing response lengths
            default_tokenizer = "EleutherAI/gpt-neox-20b" if model_config.cls == "mamba_ssm" else model_config.name
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name or default_tokenizer,
                revision=model_config.revision if model_config.cls != "mamba_ssm" else None,
                trust_remote_code=True
            )
            
            for sample in task_dict:
                if 'resps' in sample and 'doc' in sample:
                    if task_name in ['piqa', 'copa']:
                        target_idx = int(sample['doc']['label'])
                    elif task_name == 'winogrande':
                        target_idx = int(sample['doc']['answer'])-1
                    elif task_name == 'hellaswag':
                        target_idx = int(sample['doc']['gold'])
                    elif task_name == 'social_iqa':
                        target_idx = int(sample['target'])
                    else :
                        # tasks: arc_easy, arc_challenge, openbookqa, commonsense_qa
                        labels = sample['doc']['choices']['label']
                        answer_key = sample['doc']['answerKey']
                        # Get the target index by finding the position of the answer key in labels
                        target_idx = labels.index(answer_key)
                    # Get log probabilities and response texts for each choice

                    log_probs = []
                    for resp, qa in zip(sample['resps'], sample['arguments']):
                        # resp[0][0] is the negative log likelihood, resp[0][1] is the response text
                        answer = qa[1]
                        response_text = answer
                        nll = -resp[0][0]
                        # Count tokens in response (excluding any special tokens)
                        n_tokens = len(tokenizer.encode(response_text)) - 1  # -1 for the initial token
                        # Compute per-token negative log likelihood
                        log_probs.append(nll / max(1, n_tokens))  # avoid division by zero
                    
                    total_loss += log_probs[target_idx]
                    num_samples += 1
            
            results['loss'] = total_loss / num_samples if num_samples > 0 else float('inf')
    print(results)
    
    return results

def main():
    args = parse_args()
    # Determine evaluation mode
    if args.accuracy_only:
        eval_mode = 'accuracy-only'
    elif args.loss_only:
        eval_mode = 'loss-only'
    else:
        eval_mode = 'compute-both'  # default
    
    # Load configurations
    model_configs = get_model_configs(args.models_yaml)
    eval_config = get_eval_config(args.tasks_yaml, output_dir=args.results_dir)
    
    # Create output directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate each model
    shuffled_configs = list(model_configs)
    if args.shuffle:
        random.shuffle(shuffled_configs)
        
    for model_config in shuffled_configs:        
        print(f"Evaluating {model_config.name}...")
        try:
            filename = format_model_info(model_config)
            results_path = results_dir / f"{filename}.json"
            existing_results = load_existing_results(results_path)
            
            # Add model metadata if it doesn't exist
            if "model_metadata" not in existing_results:
                existing_results["model_metadata"] = get_model_metadata(model_config)
            
            # Get pending evaluations
            pending_task_evals, pending_validation_evals = get_pending_evaluations(
                eval_config.tasks, 
                eval_config.validation_configs,
                existing_results,
                eval_mode
            )
            
            # Run pending evaluations
            for task_name, num_shot in pending_task_evals:
                try:
                    print(f"  Task: {task_name}, {num_shot}-shot")
                    results = evaluate_model(model_config, task_name, num_shot, eval_mode, tokenizer_name=model_config.tokenizer)
                    
                    # Update results with new format
                    eval_key = f"{task_name}_{num_shot}_shot"
                    if eval_key not in existing_results:
                        existing_results[eval_key] = {}
                    
                    # Update metrics based on evaluation mode
                    if 'accuracy' in results:
                        existing_results[eval_key]['accuracy'] = results['accuracy']
                    if 'loss' in results:
                        existing_results[eval_key]['loss'] = results['loss']
                    # Atomic write of results
                    temp_path = results_path.with_suffix('.tmp')
                    with open(temp_path, 'w') as f:
                        json.dump(existing_results, f, indent=2)
                    temp_path.replace(results_path)
                    
                except Exception as e:
                    print(f"Error evaluating task {task_name} with {num_shot} shots: {str(e)}")
                    continue
                    
            # Compute validation losses
            for val_name in pending_validation_evals:
                try:
                    print(f"  Validation: {val_name}")
                    val_loss = compute_validation_loss(
                        model_config,
                        eval_config.validation_configs[val_name].path,
                        eval_config.validation_configs[val_name].batch_size,
                        eval_config.validation_configs[val_name].max_seq_len,
                        eval_config.validation_configs[val_name].max_steps                    )
                    eval_key = f"validation_{val_name}"
                    existing_results[eval_key] = {'loss': val_loss}
                    # Atomic write of results
                    temp_path = results_path.with_suffix('.tmp')
                    with open(temp_path, 'w') as f:
                        json.dump(existing_results, f, indent=2)
                    temp_path.replace(results_path)
                    
                except Exception as e:
                    print(f"Error computing validation loss for {val_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing model {model_config.name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()