#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=apertus-infer
#SBATCH --job-name=apertus-prefix
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/src/apertus/experiments/logs/apertus-prefix-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/src/apertus/experiments/logs/apertus-prefix-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --array=1-4

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure correct transformers version
pip install --upgrade transformers==4.56.0

# Model configuration - using HuggingFace models
# For 8B model (uses 2 GPUs)
MODEL_PATH="swiss-ai/Apertus-8B-2509"
TENSOR_PARALLEL=2

# For 70B model (uses 4 GPUs)
# MODEL_PATH="swiss-ai/Apertus-70B-2509"
# TENSOR_PARALLEL=4

# Configure based on array task ID
# Task 1: Gutenberg V1 + Greedy decoding
# Task 2: Gutenberg V2 + Greedy decoding  
# Task 3: Gutenberg V1 + Nucleus sampling
# Task 4: Gutenberg V2 + Nucleus sampling

case $SLURM_ARRAY_TASK_ID in
    1)
        echo "Task 1: Gutenberg V1 + Greedy decoding"
        VERSION="v1"
        OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/PDM/src/apertus/assets/inferences/Stage_1"
        STRATEGY="greedy"
        STRATEGY_ARGS=""
        ;;
    2)
        echo "Task 2: Gutenberg V2 + Greedy decoding"
        VERSION="v2"
        OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/PDM/src/apertus/assets/inferences/Stage_2"
        STRATEGY="greedy"
        STRATEGY_ARGS=""
        ;;
    3)
        echo "Task 3: Gutenberg V1 + Nucleus sampling (temp=1.0, p=0.9)"
        VERSION="v1"
        OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/PDM/src/apertus/assets/inferences/Stage_1"
        STRATEGY="nucleus"
        STRATEGY_ARGS="--temperature 1.0 --top-p 0.9"
        ;;
    4)
        echo "Task 4: Gutenberg V2 + Nucleus sampling (temp=1.0, p=0.9)"
        VERSION="v2"
        OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/PDM/src/apertus/assets/inferences/Stage_2"
        STRATEGY="nucleus"
        STRATEGY_ARGS="--temperature 1.0 --top-p 0.9"
        ;;
    *)
        echo "Error: Invalid array task ID $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Dataset: Gutenberg $VERSION ($([ "$VERSION" = "v1" ] && echo "0-9T tokens, 500 samples" || echo "9-12T tokens, 167 samples"))"
echo "Strategy: $STRATEGY"

# Script path - now in parent directory
SCRIPT_PATH="/iopsstor/scratch/cscs/xyixuan/PDM/src/apertus/main.py"

echo "=========================================="
echo "Apertus Prefix Length Experiment - Array Job"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: Gutenberg $VERSION"
echo "Strategy: $STRATEGY"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Run the full experiment with online data loading
python $SCRIPT_PATH \
    --model-path $MODEL_PATH \
    --version $VERSION \
    --output-dir $OUTPUT_DIR \
    --repetitions 8 24 \
    --prefix-length 50 100 250 500 1000 2000 3000 4000 5000 \
    --suffix-length 500 \
    --offset 0 \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --max-model-len 65536 \
    --show-top 10 \
    --strategy $STRATEGY \
    $STRATEGY_ARGS 

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="