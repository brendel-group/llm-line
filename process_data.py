"""
Create DataFrame Script

This script processes and combines training logs and evaluation results from different sources:
1. Training logs from local experiments (Llama and Mamba models)
2. Evaluation results from HuggingFace models

The script:
1. Reads training metrics from log files (metrics.jsonl, metrics.validation.jsonl, metrics.eval.jsonl)
2. Processes HuggingFace model evaluation results
3. Combines and standardizes the data
4. Adds metadata like model architecture, size, and training interventions
5. Outputs a CSV file with the combined dataset

Usage:
    python create_df.py [--log-dir LOG_DIR] [--hf-dir HF_DIR]

Arguments:
    --log-dir: Directory containing training logs (default: /fast/pmayilvahanan/lm_logs/lingua/)
    --hf-dir: Directory containing HuggingFace evaluation results (default: /fast/pmayilvahanan/llm_line/results_26012025)

Output:
    - data/model-data_<timestamp>.csv: Timestamped version of the combined dataset
    - model-data.csv: Latest version of the combined dataset
"""

import argparse
import datetime
import glob
import json
import os
import re
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

MMLU_SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


MMLU_TASK_SAMPLE_COUNTS = {
    "test": {
        "mmlu_abstract_algebra": 100,
        "mmlu_anatomy": 135,
        "mmlu_astronomy": 152,
        "mmlu_business_ethics": 100,
        "mmlu_clinical_knowledge": 265,
        "mmlu_college_biology": 144,
        "mmlu_college_chemistry": 100,
        "mmlu_college_computer_science": 100,
        "mmlu_college_mathematics": 100,
        "mmlu_college_medicine": 173,
        "mmlu_college_physics": 102,
        "mmlu_computer_security": 100,
        "mmlu_conceptual_physics": 235,
        "mmlu_econometrics": 114,
        "mmlu_electrical_engineering": 145,
        "mmlu_elementary_mathematics": 378,
        "mmlu_formal_logic": 126,
        "mmlu_global_facts": 100,
        "mmlu_high_school_biology": 310,
        "mmlu_high_school_chemistry": 203,
        "mmlu_high_school_computer_science": 100,
        "mmlu_high_school_european_history": 165,
        "mmlu_high_school_geography": 198,
        "mmlu_high_school_government_and_politics": 193,
        "mmlu_high_school_macroeconomics": 390,
        "mmlu_high_school_mathematics": 270,
        "mmlu_high_school_microeconomics": 238,
        "mmlu_high_school_physics": 151,
        "mmlu_high_school_psychology": 545,
        "mmlu_high_school_statistics": 216,
        "mmlu_high_school_us_history": 204,
        "mmlu_high_school_world_history": 237,
        "mmlu_human_aging": 223,
        "mmlu_human_sexuality": 131,
        "mmlu_international_law": 121,
        "mmlu_jurisprudence": 108,
        "mmlu_logical_fallacies": 163,
        "mmlu_machine_learning": 112,
        "mmlu_management": 103,
        "mmlu_marketing": 234,
        "mmlu_medical_genetics": 100,
        "mmlu_miscellaneous": 783,
        "mmlu_moral_disputes": 346,
        "mmlu_moral_scenarios": 895,
        "mmlu_nutrition": 306,
        "mmlu_philosophy": 311,
        "mmlu_prehistory": 324,
        "mmlu_professional_accounting": 282,
        "mmlu_professional_law": 1534,
        "mmlu_professional_medicine": 272,
        "mmlu_professional_psychology": 612,
        "mmlu_public_relations": 110,
        "mmlu_security_studies": 245,
        "mmlu_sociology": 201,
        "mmlu_us_foreign_policy": 100,
        "mmlu_virology": 166,
        "mmlu_world_religions": 171,
    },
    "validation": {
        "mmlu_abstract_algebra": 11,
        "mmlu_anatomy": 14,
        "mmlu_astronomy": 16,
        "mmlu_business_ethics": 11,
        "mmlu_clinical_knowledge": 29,
        "mmlu_college_biology": 16,
        "mmlu_college_chemistry": 8,
        "mmlu_college_computer_science": 11,
        "mmlu_college_mathematics": 11,
        "mmlu_college_medicine": 22,
        "mmlu_college_physics": 11,
        "mmlu_computer_security": 11,
        "mmlu_conceptual_physics": 26,
        "mmlu_econometrics": 12,
        "mmlu_electrical_engineering": 16,
        "mmlu_elementary_mathematics": 41,
        "mmlu_formal_logic": 14,
        "mmlu_global_facts": 10,
        "mmlu_high_school_biology": 32,
        "mmlu_high_school_chemistry": 22,
        "mmlu_high_school_computer_science": 9,
        "mmlu_high_school_european_history": 18,
        "mmlu_high_school_geography": 22,
        "mmlu_high_school_government_and_politics": 21,
        "mmlu_high_school_macroeconomics": 43,
        "mmlu_high_school_mathematics": 29,
        "mmlu_high_school_microeconomics": 26,
        "mmlu_high_school_physics": 17,
        "mmlu_high_school_psychology": 60,
        "mmlu_high_school_statistics": 23,
        "mmlu_high_school_us_history": 22,
        "mmlu_high_school_world_history": 26,
        "mmlu_human_aging": 23,
        "mmlu_human_sexuality": 12,
        "mmlu_international_law": 13,
        "mmlu_jurisprudence": 11,
        "mmlu_logical_fallacies": 18,
        "mmlu_machine_learning": 11,
        "mmlu_management": 11,
        "mmlu_marketing": 25,
        "mmlu_medical_genetics": 11,
        "mmlu_miscellaneous": 86,
        "mmlu_moral_disputes": 38,
        "mmlu_moral_scenarios": 100,
        "mmlu_nutrition": 33,
        "mmlu_philosophy": 34,
        "mmlu_prehistory": 35,
        "mmlu_professional_accounting": 31,
        "mmlu_professional_law": 170,
        "mmlu_professional_medicine": 31,
        "mmlu_professional_psychology": 69,
        "mmlu_public_relations": 12,
        "mmlu_security_studies": 27,
        "mmlu_sociology": 22,
        "mmlu_us_foreign_policy": 11,
        "mmlu_virology": 18,
        "mmlu_world_religions": 19,
    },
}


def compute_weighted_mmlu_metrics(
    row: pd.Series, task_sample_counts: dict, subjects_map: dict, split: str = "test"
):
    """
    Compute overall MMLU weighted accuracy and loss, as well as subgroup
    weighted metrics, for a single row of results in a DataFrame.

    :param row: A single row (pd.Series) containing columns like
                'mmlu_abstract_algebra/acc' and 'mmlu_abstract_algebra/loss'
    :param task_sample_counts: Dictionary mapping "split" -> { task_name -> sample_count }
    :param subjects_map: Dictionary mapping subject (e.g. "abstract_algebra") to subgroup
                        (e.g. "stem", "social_sciences", etc.)
    :param split: Which set to use from the sample_counts dictionary, typically "test" or "validation"
    :return: A dictionary containing overall weighted accuracy, overall weighted loss,
             and subgroup weighted accuracies and losses.
    """

    # Initialize accumulators
    total_acc_sum = 0.0
    total_loss_sum = 0.0
    total_samples = 0

    # Subgroup accumulators
    subgroup_acc_sums = defaultdict(float)
    subgroup_loss_sums = defaultdict(float)
    subgroup_samples = defaultdict(int)

    for subject, subgroup in subjects_map.items():
        # Construct column names, e.g. "mmlu_abstract_algebra/acc"
        acc_col = f"mmlu_{subject} Accuracy"
        loss_col = f"mmlu_{subject} Loss"
        task_key = f"mmlu_{subject}"  # used in task_sample_counts

        # Skip if this specific subject is not present in task_sample_counts for the split
        if task_key not in task_sample_counts[split]:
            continue

        # Sample count for this subject
        subject_count = task_sample_counts[split][task_key]
        if subject_count <= 0:
            continue

        # Fetch accuracy and loss from the row (if not present, skip)
        if acc_col not in row or loss_col not in row:
            continue

        subject_acc = row[acc_col]
        subject_loss = row[loss_col]

        # Aggregate overall sums
        total_acc_sum += subject_acc * subject_count
        total_loss_sum += subject_loss * subject_count
        total_samples += subject_count

        # Aggregate subgroup sums
        subgroup_acc_sums[subgroup] += subject_acc * subject_count
        subgroup_loss_sums[subgroup] += subject_loss * subject_count
        subgroup_samples[subgroup] += subject_count

    # Compute overall weighted accuracy/loss
    if total_samples > 0:
        overall_acc = total_acc_sum / total_samples
        overall_loss = total_loss_sum / total_samples
    else:
        overall_acc = float("nan")
        overall_loss = float("nan")

    # Compute subgroup weighted accuracies/losses
    subgroup_results = {}
    for sg in subgroup_acc_sums:
        if subgroup_samples[sg] > 0:
            sg_acc = subgroup_acc_sums[sg] / subgroup_samples[sg]
            sg_loss = subgroup_loss_sums[sg] / subgroup_samples[sg]
        else:
            sg_acc = float("nan")
            sg_loss = float("nan")
        subgroup_results[sg] = {
            "acc": sg_acc,
            "loss": sg_loss,
            "sample_count": subgroup_samples[sg],
        }

    return {
        "overall_acc": overall_acc,
        "overall_loss": overall_loss,
        "subgroups": subgroup_results,
        "total_samples": total_samples,
    }


def extract_mmlu_losses(row):
    # Compute weighted MMLU metrics for this row
    metrics = compute_weighted_mmlu_metrics(
        row, MMLU_TASK_SAMPLE_COUNTS, MMLU_SUBJECTS, split="test"
    )

    # Safely fetch subgroup losses (handle missing keys gracefully)
    stem_loss = metrics["subgroups"].get("stem", {}).get("loss", float("nan"))
    humanities_loss = (
        metrics["subgroups"].get("humanities", {}).get("loss", float("nan"))
    )
    social_sciences_loss = (
        metrics["subgroups"].get("social_sciences", {}).get("loss", float("nan"))
    )
    other_loss = metrics["subgroups"].get("other", {}).get("loss", float("nan"))

    # Return a pandas Series for assignment
    return pd.Series(
        {
            "mmlu Loss": metrics["overall_loss"],
            "mmlu_stem Loss": stem_loss,
            "mmlu_humanities Loss": humanities_loss,
            "mmlu_social_sciences Loss": social_sciences_loss,
            "mmlu_other Loss": other_loss,
        }
    )


def compute_mmlu_losses(df: pd.DataFrame) -> pd.DataFrame:
    df[
        [
            "mmlu Loss",
            "mmlu_stem Loss",
            "mmlu_other Loss",
            "mmlu_humanities Loss",
            "mmlu_social_sciences Loss",
        ]
    ] = df.apply(extract_mmlu_losses, axis=1)

    return df


def gather_experiment_data(base_dir, exclude_dirs=[]):
    """
    Gathers experiment information from each training run/folder in base_dir.

    1. Scans each folder (model run).
    2. Reads:
       - config.yaml
       - metrics.jsonl (training logs)
       - metrics.validation.jsonl (val logs)
       - metrics.eval.jsonl (test logs; if multi_eval subfolder exists, read from there)
    3. Aggregates data for each folder and merges per 'global_step'.
    4. Returns a single DataFrame with columns:
        'name': folder name
        'arch': first token in name (e.g., 'llama' or 'mamba')
        'size': second token in name
        'dim': from config.yaml->model->dim
        'n_layers': from config.yaml->model->n_layers
        'pretraining_data': from config.yaml->data->sources->(something_shuffled) => the something part
        'directory': from config.yaml->dump_dir
        'steps': current iteration (global_step)
        'train_loss': from metrics.jsonl->"loss/out"
        'tokens': from metrics.jsonl->"optim/total_tokens"
        'test_loss': from metrics.validation.jsonl->(pretraining_data)_shuffled->"nll_per_token"
        '{dataset}_val_loss': for each dataset in metrics.validation.jsonl->(dataset)_shuffled->"nll_per_token"
        '{task}/acc' and '{task}/loss': from metrics.eval.jsonl for all tasks (arc_challenge, arc_easy, etc.)

    Args:
        base_dir (str): Path containing many subfolders of model runs.

    Returns:
        pd.DataFrame: Combined DataFrame with row per (folder, global_step).
    """
    all_rows = []

    # -------------------------------------------------------------------------
    # Helper function to parse config.yaml and extract relevant info
    # -------------------------------------------------------------------------
    def parse_config(config_path):
        """
        Parse config.yaml and extract:
            dim, n_layers, pretraining_data (e.g., 'c4'), and dump_dir.
        """
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        parsed = {}

        parsed["dim"] = config_data.get("model", {}).get("dim", None)
        parsed["n_layers"] = config_data.get("model", {}).get("n_layers", None)
        parsed["tokenizer"] = (
            config_data.get("data", {}).get("tokenizer", {}).get("name", None)
        )

        # data->sources typically has a key like 'c4_shuffled' => 1.0
        # we only grab the portion before "_shuffled" from the single key
        data_sources = config_data.get("data", {}).get("sources", {})
        p_data = None
        if isinstance(data_sources, dict) and len(data_sources) > 0:
            source_keys = list(data_sources.keys())
            # Usually there's just one or so, but let's take the first
            key = source_keys[0]
            if key.endswith("_shuffled"):
                p_data = key.replace("_shuffled", "")
            else:
                p_data = key  # fallback if something else
        parsed["pretraining_data"] = p_data

        # new config args from gpt2 module
        parsed["norm_type"] = config_data.get("model", {}).get("norm_type", "rms_norm")
        parsed["use_rope"] = config_data.get("model", {}).get("use_rope", None)

        # new config args from llama ablation code edits
        parsed["attn_type"] = config_data.get("model", {}).get(
            "attn_type", None
        )  # gpt, llama
        parsed["position_embedding"] = config_data.get("model", {}).get(
            "pos_embed_type", None
        )  # absolute, rope
        parsed["ffn_activation"] = config_data.get("model", {}).get(
            "ffn_activation", None
        )  # gelu, relu
        parsed["dropout"] = config_data.get("model", {}).get("dropout", None)
        parsed["use_gpt_init"] = config_data.get("model", {}).get("use_gpt_init", None)
        parsed["bias"] = config_data.get("model", {}).get("bias", None)
        parsed["weight_tying"] = config_data.get("model", {}).get("weight_tying", None)
        parsed["llama_linear"] = config_data.get("model", {}).get("llama_linear", None)
        parsed["context_length"] = config_data.get("data", {}).get("seq_len", None)
        parsed["learning_rate"] = config_data.get("optim", {}).get("lr", None)
        parsed["weight_decay"] = config_data.get("optim", {}).get("weight_decay", None)
        parsed["optimizer"] = config_data.get("optim", {}).get(
            "optimizer_type", "adamw"
        )
        optim_names = {"adam": "Adam", "adamw": "AdamW"}
        parsed["optimizer"] = optim_names.get(parsed["optimizer"], parsed["optimizer"])
        parsed["scheduler"] = config_data.get("optim", {}).get("scheduler", "cosine")
        sched_names = {"cosine": "Cosine", "wsd": "WSD"}
        parsed["scheduler"] = sched_names.get(parsed["scheduler"], parsed["scheduler"])
        parsed["directory"] = config_data.get("dump_dir", None)

        return parsed

    # -------------------------------------------------------------------------
    # Helper to read metrics.validation.jsonl => returns { step: { ... } }
    # Also sets 'test_loss' based on the pretraining_data key
    # -------------------------------------------------------------------------
    def parse_metrics_validation(val_path, pretraining_data):
        """
        Parse metrics.validation.jsonl and store columns:
            {dataset}_val_loss for each dataset in the line
            test_loss = <(pretraining_data)_shuffled nll_per_token>
            keyed by global_step
        """
        step_dict = {}
        with open(val_path, "r") as f:
            for line in f:
                line_data = json.loads(line)
                gstep = line_data.get("global_step", None)
                if gstep is None:
                    continue

                # Create a dictionary for the current step if not present
                if gstep not in step_dict:
                    step_dict[gstep] = {}

                # For each dataset we find "nll_per_token"
                for dataset_key, dataset_vals in line_data.items():
                    # The keys with dataset info typically end with "_shuffled"
                    if (
                        isinstance(dataset_vals, dict)
                        and "nll_per_token" in dataset_vals
                    ):
                        # e.g. c4_shuffled => "c4_val_loss"
                        if dataset_key.endswith("_shuffled"):
                            dataset_name = dataset_key.replace("_shuffled", "")
                            val_loss_col = f"{dataset_name} Validation Loss"
                            step_dict[gstep][val_loss_col] = dataset_vals[
                                "nll_per_token"
                            ]

                # test_loss: if pretraining_data = 'c4', we look at c4_shuffled => nll_per_token
                # and store it in step_dict[gstep]['test_loss']
                if pretraining_data is not None:
                    pd_shuffled = f"{pretraining_data}_shuffled"
                    if (
                        pd_shuffled in line_data
                        and "nll_per_token" in line_data[pd_shuffled]
                    ):
                        step_dict[gstep]["Test Loss"] = line_data[pd_shuffled][
                            "nll_per_token"
                        ]

        return step_dict

    # -------------------------------------------------------------------------
    # Helper to read metrics.jsonl => returns { step: { 'train_loss': ..., 'tokens': ... } }
    # -------------------------------------------------------------------------
    def parse_metrics_train(train_path):
        """
        Parse metrics.jsonl for training data:
            train_loss = "loss/out"
            tokens = "optim/total_tokens"
        keyed by global_step
        """
        step_dict = {}
        with open(train_path, "r") as f:
            for line in f:
                line_data = json.loads(line)
                gstep = line_data.get("global_step", None)
                if gstep is None:
                    continue

                if gstep not in step_dict:
                    step_dict[gstep] = {}

                # Store train loss and token count
                if "loss/out" in line_data:
                    step_dict[gstep]["Train Loss"] = line_data["loss/out"]

                if "optim/total_tokens" in line_data:
                    step_dict[gstep]["Tokens"] = line_data["optim/total_tokens"]

                if "speed/FLOPS" in line_data:
                    step_dict[gstep]["FLOP/s"] = line_data["speed/FLOPS"]

                if "speed/curr_iter_time" in line_data:
                    step_dict[gstep]["Iter Time"] = line_data["speed/curr_iter_time"]

                if "speed/data_load_time" in line_data:
                    step_dict[gstep]["Data Time"] = line_data["speed/data_load_time"]

        return step_dict

    # -------------------------------------------------------------------------
    # Helper to read metrics.eval.jsonl => returns { step: { 'arc_challenge/acc': ..., 'arc_challenge/loss': ... etc. } }
    # If multi_eval folder is present with a metrics.eval.jsonl, read from there instead.
    # -------------------------------------------------------------------------
    def parse_metrics_eval(folder_path):
        """
        Parse metrics.eval.jsonl for test tasks:
         - each task => 'acc,none' => (task)/acc
         - each task => 'loss' => (task)/loss
        Return: { step: { 'arc_challenge/acc': <val>, 'arc_challenge/loss': <val>, ... } }
        """
        # Check if there is a 'multi_eval' subfolder with a 'metrics.eval.jsonl' file
        eval_path = os.path.join(folder_path, "metrics.eval.jsonl")

        if not os.path.isfile(eval_path):
            return {}

        step_dict = {}
        with open(eval_path, "r") as f:
            for line in f:
                line_data = json.loads(line)
                gstep = line_data.get("global_step", None)
                if gstep is None:
                    continue

                if gstep not in step_dict:
                    step_dict[gstep] = {}

                # For each task in this line, if it's a dictionary
                # with 'acc,none' or 'loss', store them
                for k, v in line_data.items():
                    if isinstance(v, dict):
                        # e.g. arc_challenge => v might have 'acc,none', 'loss'
                        if "acc,none" in v:
                            step_dict[gstep][f"{k} Accuracy"] = v["acc,none"]
                        if "loss" in v:
                            step_dict[gstep][f"{k} Loss"] = v["loss"]

        return step_dict

    # Add helper function to parse train.log for model size
    def parse_train_log_size_flops(folder_path) -> Tuple[int, float]:
        """Extract model size from train.log file"""
        log_path = os.path.join(folder_path, "train.log")
        if not os.path.isfile(log_path):
            return None, None

        pattern = r"flops: ([\d.eE+-]+)"
        flops_list = []
        with open(log_path, "r") as f:
            for line in f:
                if "Model size:" in line:
                    # Extract number before "total parameters"
                    size_str = (
                        line.split("Model size:")[1]
                        .split("total parameters")[0]
                        .strip()
                    )

                match = re.search(pattern, line)
                if match:
                    flops_list.append(float(match.group(1)))
                if len(flops_list) >= 100:
                    break

        return int(size_str.replace(",", "")), np.mean(flops_list)

    # -------------------------------------------------------------------------
    # Main loop: each directory in base_dir is presumably a model run.
    # We parse the config.yaml, read the logs, and merge them by global_step.
    # -------------------------------------------------------------------------
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)

        if folder_name in exclude_dirs:
            continue

        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        config_path = os.path.join(folder_path, "config.yaml")
        metrics_train_path = os.path.join(folder_path, "metrics.jsonl")
        metrics_val_path = os.path.join(folder_path, "metrics.validation.jsonl")

        # We only consider "valid" runs that have a config and logs
        if not os.path.isfile(config_path):
            continue
        if not os.path.isfile(metrics_train_path):
            continue
        if not os.path.isfile(metrics_val_path):
            continue

        # Parse config
        config_info = parse_config(config_path)
        # e.g. config_info = {'dim': 1024, 'n_layers': 24, 'pretraining_data': 'c4', 'directory': '/path/to/folder'}

        # Parse logs
        val_dict = parse_metrics_validation(
            metrics_val_path, config_info["pretraining_data"]
        )
        train_dict = parse_metrics_train(metrics_train_path)
        eval_dict = parse_metrics_eval(folder_path)

        # Some columns are from the folder name
        # e.g. "llama_172M_d_512_l_12_c4_10BT"
        name_tokens = folder_name.split("_")
        arch = name_tokens[0] if len(name_tokens) > 0 else None
        size = name_tokens[1] if len(name_tokens) > 1 else None

        # Get model size from train.log
        model_size, flops = parse_train_log_size_flops(folder_path)

        # Gather all possible steps from the union of val, train, eval data
        all_steps = (
            set(val_dict.keys()).union(train_dict.keys()).union(eval_dict.keys())
        )

        flops = 0
        for step in sorted(all_steps):
            row = {
                "Name": folder_name,
                "Architecture": arch,
                "Size": model_size,
                "Dimensions": config_info["dim"],
                "# Layers": config_info["n_layers"],
                "Pretraining Data": config_info["pretraining_data"],
                "Directory": config_info["directory"],
                "Steps": step,
                "Tokenizer": config_info["tokenizer"],
                "Layer Norm": config_info["norm_type"],
                "Rope": config_info["use_rope"],
                "Attention Type": config_info["attn_type"],
                "Position Embedding": config_info["position_embedding"],
                "FFN Activation": config_info["ffn_activation"],
                "Dropout": config_info["dropout"],
                "GPT Init": config_info["use_gpt_init"],
                "Bias": config_info["bias"],
                "Weight Tying": config_info["weight_tying"],
                "Llama Linear": config_info["llama_linear"],
                "Context Length": config_info["context_length"],
                "Learning Rate": config_info["learning_rate"],
                "Weight Decay": config_info["weight_decay"],
                "Optimizer": config_info["optimizer"],
                "Scheduler": config_info["scheduler"],
            }

            # Insert validation info
            if step in val_dict:
                row.update(val_dict[step])

            # Insert training info
            if step in train_dict:
                flops += train_dict[step]["FLOP/s"] * (
                    train_dict[step]["Iter Time"] - train_dict[step]["Data Time"]
                )
                row.update(train_dict[step])
                row["FLOPs"] = flops

            # Insert eval info
            if step in eval_dict:
                row.update(eval_dict[step])

            all_rows.append(row)

    # Create a DataFrame from all rows and return
    df = pd.DataFrame(all_rows)
    return df


def process_json_results(results_dir: str) -> pd.DataFrame:
    """
    Process JSON result files to create a DataFrame compatible with the training logs format.

    The function:
    1. Reads all JSON files in results_dir
    2. Extracts model metadata and metrics
    3. Creates rows with columns matching the training logs DataFrame
    """
    rows = []

    # Find all JSON files
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract base metadata
        metadata = data.get("model_metadata", {})
        base_row = {
            "Name": metadata.get("model_name", ""),
            "Architecture": metadata.get("architecture", ""),
            "Size": None,  # Could parse from model name if needed
            "Pretraining Data": metadata.get("dataset", ""),
            # Add training-specific columns with None/NA
            "Dimensions": None,
            "# Layers": None,
            "Directory": None,
            "Steps": 0,  # Safely parse checkpoint step number
            "Tokenizer": None,
            "Layer Norm": None,
            "Rope": None,
            "Attention Type": None,
            "Position Embedding": None,
            "FFN Activation": None,
            "Dropout": None,
            "GPT Init": None,
            "Bias": None,
            "Weight Tying": None,
            "Llama Linear": None,
            "Context Length": None,
            "Train Loss": None,
            "Tokens": None,
            "Test Loss": None,
            "Optimizer": None,
            "Scheduler": None,
            "Learning Rate": None,
            "Weight Decay": None,
        }

        # Safely parse checkpoint step number
        checkpoint = metadata.get("checkpoint", "")
        try:
            step = int(checkpoint.split("-")[1]) if "-" in checkpoint else None
        except (IndexError, ValueError):
            step = None

        base_row["Steps"] = step

        # Process validation losses
        for key, value in data.items():
            if key.startswith("validation_"):
                dataset = key.replace("validation_", "")
                if dataset == "slimpajama_shuffled":
                    dataset = "slimpajama"
                if isinstance(value, dict) and "loss" in value:
                    base_row[f"{dataset} Validation Loss"] = value["loss"]

        # Process accuracy metrics - expanded mappings based on the JSON file
        task_mappings = {
            "arc_challenge_0_shot": "arc_challenge",
            "arc_easy_0_shot": "arc_easy",
            "hellaswag_0_shot": "hellaswag",
            "mmlu_0_shot": "mmlu",
            "winogrande_0_shot": "winogrande",
            "piqa_0_shot": "piqa",
            "social_iqa_0_shot": "social_iqa",
            "commonsense_qa_0_shot": "commonsense_qa",
            "copa_0_shot": "copa",
            "openbookqa_0_shot": "openbookqa",
        }

        for result_key, task_name in task_mappings.items():
            if result_key in data:
                result = data[result_key]

                # Get accuracy
                if "accuracy" in result and task_name in result["accuracy"]:
                    metrics = result["accuracy"][task_name]
                    if "acc,none" in metrics:
                        base_row[f"{task_name} Accuracy"] = metrics["acc,none"]

                # Get loss
                if "loss" in result:
                    if isinstance(result["loss"], dict):
                        # Handle MMLU case where loss is a dict
                        if task_name == "mmlu":
                            base_row[f"{task_name} Loss"] = result["loss"].get("mmlu")
                    else:
                        base_row[f"{task_name} Loss"] = result["loss"]

        rows.append(base_row)

    return pd.DataFrame(rows)


def get_intervention(row):
    interventions = []
    name = row["Name"]
    if any(pattern in name for pattern in ["_d_", "_l_"]):
        interventions.append("Size")
    if "_ctx_" in name:
        interventions.append("Context Length")
    if any(pattern in name for pattern in ["_lr_", "_wd_", "adam", "cosine"]):
        interventions.append("Optimizer")
    if row["Architecture"] != "Llama":
        interventions.append("Architecture")
    if row["Pretraining Data"] != "FineWeb-Edu":
        interventions.append("Pretraining Data")
    if row["Tokenizer"] != "tiktoken":
        interventions.append("Tokenizer")
    return tuple(interventions) if interventions else None


def main(log_dir: str, hf_dir: str):
    # Load training logs data
    print("Gathering Llama data...")
    df_llama = gather_experiment_data(log_dir + "/llama/")
    df_llama.dropna(subset=["Train Loss", "Test Loss"], how="any", inplace=True)

    print("Gathering Mamba data...")
    df_mamba = gather_experiment_data(log_dir + "/mamba/")
    df_mamba.dropna(subset=["Train Loss", "Test Loss"], how="any", inplace=True)

    print("Concatenating data...")
    df_train = pd.concat([df_llama, df_mamba], ignore_index=True)
    df_train.dropna(subset=["Train Loss", "Test Loss"], how="any", inplace=True)

    # Add HF column to training data
    df_train["Hugging Face"] = False

    # Load and process results data
    print("Gathering HF data...")
    df_results = process_json_results(hf_dir)
    df_results["Hugging Face"] = True

    # Combine both dataframes
    print("Concatenating data...")
    df_combined = pd.concat([df_train, df_results], ignore_index=True)

    # Post-process the combined dataframe
    df = compute_mmlu_losses(df_combined)

    # Make architecture explicit
    df.loc[df["Architecture"] == "mamba", "Architecture"] = "Mamba"
    df.loc[df["Architecture"] == "llama", "Architecture"] = "Llama"

    # Make tokenizer explicit
    df.loc[
        df["Hugging Face"] & (df["Name"].str.contains("neox|pythia|mamba|mamba2")),
        "Tokenizer",
    ] = "gpt-neox"
    df.loc[
        df["Hugging Face"] & (df["Name"].str.contains("gpt-j|neo-|ablation")),
        "Tokenizer",
    ] = "gpt2-HF"
    df.loc[df["Hugging Face"] & (df["Name"].str.contains("Qwen")), "Tokenizer"] = "qwen"

    # Make pretraining data explicit
    df.loc[df["Pretraining Data"] == "FineWeb-EDU", "Pretraining Data"] = "FineWeb-Edu"
    df.loc[df["Pretraining Data"] == "fineweb_edu_100bt", "Pretraining Data"] = (
        "FineWeb-Edu"
    )
    df.loc[df["Pretraining Data"] == "c4", "Pretraining Data"] = "C4"
    df.loc[df["Pretraining Data"] == "pile_uc", "Pretraining Data"] = "The Pile UC"
    # GPT models were trained on the Pile or the Pile deduped depending on the name
    df.loc[
        (df["Architecture"] == "GPT") & (df["Name"].str.contains("deduped")),
        "Pretraining Data",
    ] = "The Pile Deduped"
    df.loc[
        (df["Architecture"] == "GPT") & ~(df["Name"].str.contains("deduped")),
        "Pretraining Data",
    ] = "The Pile"
    # LLama models were trained on the Pile deduped
    df.loc[
        (df["Architecture"] == "Llama")
        & (df["Pretraining Data"] == "The Pile")
        & df["Hugging Face"],
        "Pretraining Data",
    ] = "The Pile Deduped"

    # Get interventions
    df["Intervention"] = df.apply(get_intervention, axis=1)

    # Cap size
    df["Overtraining"] = df["Tokens"] > 8.4e9

    # Rename columns
    legible_columns = {
        "c4": "C4",
        "pile_uc": "The Pile UC",
        "fineweb_edu_100bt": "FineWeb-Edu",
        "refineweb": "RefineWeb",
        "slimpajama": "Slimpajama",
        "arc_challenge": "ARC-Challenge",
        "arc_easy": "ARC-Easy",
        "openbookqa": "OpenBookQA",
        "piqa": "PIQA",
        "copa": "COPA",
        "winogrande": "Winogrande",
        "hellaswag": "HellaSwag",
        "mmlu": "MMLU",
        "social_iqa": "Social IQa",
        "commonsense_qa": "CommonSenseQA",
    }
    for i, _ in enumerate(df.columns):
        for old, new in legible_columns.items():
            col = df.columns[i]
            if old in col:
                df.rename(columns={col: col.replace(old, new)}, inplace=True)

    # Add details to HuggingFace models
    df_hf_details = pd.read_csv("../update_model_details.csv")
    # Clean all column names
    df_hf_details.columns = df_hf_details.columns.str.strip().str.lower()
    df_hf_details.columns = df_hf_details.columns.str.replace(" ", "_")

    for index, row in df_hf_details.iterrows():
        name = row["name"].strip()
        revision = row["revision"].strip()
        if "-" in revision:
            step = float(revision.split("-")[1])
        else:
            step = None

        size = int(row["parameter_size"])
        tokens = int(row["number_of_tokens_trained_on"])

        if step is not None:
            matching_rows = df[(df["Name"] == name) & (df["Steps"] == step)]
        else:
            matching_rows = df[(df["Name"] == name) & (df["Steps"].isna())]
        if len(matching_rows) < 1:
            print("Skipping", name, step, len(matching_rows))
            continue
        df.loc[matching_rows.index, "Size"] = size
        df.loc[matching_rows.index, "Tokens"] = tokens

    # Save the combined DataFrame
    print("Saving data...")
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    df_combined.to_csv(f"data/{timestamp}.csv", index=False)
    df_combined.to_csv("data/latest.csv", index=False)

    # Print sample of combined data
    print("\nData shape:", df.shape)
    print("\nData sample:")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/fast/pmayilvahanan/lm_logs/lingua/",
        help="Directory where the log files are stored",
    )
    parser.add_argument(
        "--hf-dir",
        type=str,
        default="/fast/pmayilvahanan/llm_line/results_26012025",
        help="Directory where the HuggingFace model results are stored",
    )
    args = parser.parse_args()

    main(args.log_dir, args.hf_dir)
