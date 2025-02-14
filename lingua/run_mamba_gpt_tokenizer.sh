# Get arguments
NPROC=$1
CONFIG=$2
API_KEY=$3
source ~/.lingua_241216/bin/activate
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not needed
export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues
torchrun --nproc-per-node $NPROC -m apps.mamba.train config=apps/mamba/configs/$CONFIG
