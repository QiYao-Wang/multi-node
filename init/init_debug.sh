#!/bin/bash
source /host/adapt.sh

export PROJECT_HOME="your path"

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=${PROJECT_HOME}/debug_log/log_rank_%h_%p.txt
export NCCL_ASYNC_ERROR_HANDLING=1

# Performance settings
export OMP_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Get environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NNODES=$NNODES
export NODE_RANK=$NODE_RANK
export GPUS_PER_NODE=4

# Calculate total world size
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Run Python script with torchrun
torchrun --nnodes=$NNODES \
         --nproc_per_node=$GPUS_PER_NODE \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         $PROJECT_HOME/init_debug.py
