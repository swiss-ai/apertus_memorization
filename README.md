# Apertus Memorization Analysis

Self-contained framework for analyzing verbatim memorization in Apertus long-context base model across different training stages.

## Overview

This repository provides tools to measure and analyze memorization in the Apertus family of models by comparing model outputs against known training data from Project Gutenberg. The analysis compares two training stages:
- **Stage 1 (V1)**: Gutenberg texts from 0-9T training tokens
- **Stage 2 (V2)**: Gutenberg texts from 9-12T training tokens

## Features

- **Online Data Loading**: Automatic loading of Gutenberg datasets from HuggingFace
- **HuggingFace Model Support**: Direct loading of Swiss AI Apertus models
- **Multiple Generation Strategies**: Greedy decoding and nucleus sampling
- **Comprehensive Metrics**: ROUGE-L, LCCS, and Type-Token Ratio (TTR)
<!-- - **Contamination Filtering**: Automatic exclusion of known problematic sequences -->
- **HTML Visualizations**: Token-level comparison visualizations
- **Batch Processing**: Efficient vLLM-based inference with tensor parallelism

## Installation

### Requirements
```bash
# Core dependencies
pip install vllm transformers==4.56.0 

# Note: transformers 4.56.0 is required for Apertus model support
# Note: vLLM requires CUDA 11.8+ and a compatible PyTorch version
# Important: Native vLLM support for Apertus models is pending official merge.
```

## Quick Start

### 1. Run Memorization Analysis

```bash
# Single configuration with greedy decoding
python main.py \
    --model-path swiss-ai/Apertus-8B-2509 \
    --version v1 \
    --output-dir ./results \
    --repetitions 1 8 32 128 \
    --prefix-length 500 \
    --suffix-length 500 \
    --offset 0 \
    --tensor-parallel-size 2 \
    --strategy greedy \
    --show-top 10

# With nucleus sampling
python main.py \
    --model-path swiss-ai/Apertus-8B-2509 \
    --version v2 \
    --output-dir ./results \
    --repetitions 0a 0b 1 2 4 8 16 32 64 128 \
    --prefix-length 500 \
    --suffix-length 500 \
    --offset 0 \
    --tensor-parallel-size 2 \
    --strategy nucleus \
    --temperature 1.0 \
    --top-p 0.9 \
    --show-top 10
```

### 2. Submit Batch Jobs (SLURM)

```bash
# Submit all experiments (greedy + nucleus for both V1 and V2)
cd scripts
sbatch submit_offset_experiment.sh
sbatch submit_prefix_length_experiment.sh

# Submit only greedy decoding experiments
sbatch --array=1-2 submit_offset_experiment.sh

# Submit only nucleus sampling experiments  
sbatch --array=3-4 submit_offset_experiment.sh
```

## Available Models

| Model | HuggingFace ID | Parameters | Recommended TP | GPU Memory |
|-------|----------------|------------|----------------|------------|
| Apertus-8B | swiss-ai/Apertus-8B-2509 | 8B | 2 | 2x40GB |
| Apertus-70B | swiss-ai/Apertus-70B-2509 | 70B | 4-8 | 4-8x80GB |

**Important**: Use only base models for memorization analysis, not instruction-tuned versions.

## Dataset Versions

| Version | Training Stage | Token Range | Samples/Rep | Repetitions |
|---------|---------------|-------------|-------------|-------------|
| v1 | Stage 1 | 0-9T | 500 | 0a-0d, 1-128 |
| v2 | Stage 2 | 9-12T | 167 | 0a-0d, 1-128 |

### Repetition Naming
- `0a, 0b, 0c, 0d`: Unique occurrences (seen exactly once)
- `1, 2, 3, 4, ...`: Number of times sequence appears in training

## Output Structure

```
output_dir/
└── swiss-ai_Apertus-8B-2509/
    └── offset_0_prefix_500_suffix_500/
        ├── inferences/
        │   ├── greedy/
        │   │   └── rep_*.jsonl         # Raw inference results
        │   └── nucleus_temp1.0_p0.9/
        │       └── rep_*.jsonl
        └── visualizations/
            ├── greedy/
            │   └── rep_*/
            │       └── sample_*.html   # Top memorized samples
            └── nucleus_temp1.0_p0.9/
                └── rep_*/
                    └── sample_*.html
```

## Command-Line Arguments

### Model Configuration
- `--model-path`: HuggingFace model ID or local path
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--gpu-memory-utilization`: GPU memory fraction to use (default: 0.9)
- `--max-model-len`: Maximum sequence length (default: auto)

### Data Configuration
- `--version`: Dataset version (`v1` or `v2`)
- `--repetitions`: Space-separated list of repetitions to process
- `--cache-dir`: HuggingFace cache directory (optional)

### Generation Configuration  
- `--prefix-length`: Number of tokens for context (can specify multiple)
- `--suffix-length`: Number of tokens to generate (can specify multiple)
- `--offset`: Starting position offset in sequence (can specify multiple)
- `--strategy`: Generation strategy (`greedy` or `nucleus`)
- `--temperature`: Temperature for nucleus sampling (default: 1.0)
- `--top-p`: Top-p value for nucleus sampling (default: 0.9)
- `--batch-size`: Batch size for processing (None for auto)

### Output Configuration
- `--output-dir`: Base directory for results
- `--show-top`: Number of top memorized samples to visualize (0 for none)
- `--force`: Reprocess even if output exists
- `--seed`: Random seed for reproducibility (default: 42)

## Metrics

### ROUGE-L
Measures longest common subsequence between generated and reference text, with F1 score normalization.

### LCCS (Normalized)
Normalized longest common consecutive subsequence length, providing a value between 0 and 1.

### Type-Token Ratio (TTR)
Measures lexical diversity as the ratio of unique tokens to total tokens.