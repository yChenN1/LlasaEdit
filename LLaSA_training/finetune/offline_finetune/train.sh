#!/bin/bash

# Set CUDA environment
# conda activate llasa
export CUDA_HOME=$HOME/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set GCC compiler paths from conda
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib:$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0:$LD_LIBRARY_PATH

# Allow unsupported compiler for NVCC
export NVCC_FLAGS="--allow-unsupported-compiler"

# Launch training with torchrun
# torchrun --nproc_per_node=1 --master-port=10203 finetune_offline.py
deepspeed --num_gpus=4 finetune_offline_w_rl.py
