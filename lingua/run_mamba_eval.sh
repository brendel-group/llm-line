#!/bin/bash

# Get arguments
NPROC=$1
CONFIG=$2
API_KEY=$3

# Source virtual environment
source ~/.lingua_241216/bin/activate

# Setup tokenizer
python setup/download_tokenizer.py llama3 /tmp/tokenizers/ --api_key=$API_KEY

# Run evaluation
export NCCL_IB_DISABLE=1
torchrun --nproc_per_node=$NPROC -m apps.mamba.eval config=apps/mamba/configs/$CONFIG 