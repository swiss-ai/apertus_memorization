# Apertus Experiment Scripts

## Overview
These SLURM scripts run memorization experiments with the Apertus models using different generation strategies and dataset versions.

## Job Array Configuration

Each script uses array jobs to run multiple configurations:
- **Tasks 1-2**: Greedy decoding (deterministic)
- **Tasks 3-4**: Nucleus sampling (temperature=1.0, top-p=0.9)
- **Tasks 1,3**: Gutenberg V1 (0-9T tokens)
- **Tasks 2,4**: Gutenberg V2 (9-12T tokens)

### Array Task Mapping
```
Task 1: Gutenberg V1 + Greedy
Task 2: Gutenberg V2 + Greedy
Task 3: Gutenberg V1 + Nucleus
Task 4: Gutenberg V2 + Nucleus
```

## Usage

### Submit all configurations (greedy + nucleus for both datasets):
```bash
sbatch submit_offset_experiment.sh
sbatch submit_prefix_length_experiment.sh
```

### Submit only greedy decoding:
```bash
sbatch --array=1-2 submit_offset_experiment.sh
sbatch --array=1-2 submit_prefix_length_experiment.sh
```

### Submit only nucleus sampling:
```bash
sbatch --array=3-4 submit_offset_experiment.sh
sbatch --array=3-4 submit_prefix_length_experiment.sh
```

### Submit specific combinations:
```bash
# Only V1 + Greedy
sbatch --array=1 submit_offset_experiment.sh

# Only V2 + Nucleus
sbatch --array=4 submit_offset_experiment.sh

# V1 with both strategies
sbatch --array=1,3 submit_offset_experiment.sh
```

## Output Structure

Results are organized by model, configuration, and strategy:
```
assets/inferences/
├── Stage_1/                           # Gutenberg V1
│   └── swiss-ai_Apertus-8B-2509/
│       └── offset_X_prefix_Y_suffix_Z/
│           ├── inferences/
│           │   ├── greedy/           # Greedy results
│           │   │   └── rep_*.jsonl
│           │   └── nucleus_temp1.0_p0.9/  # Nucleus results
│           │       └── rep_*.jsonl
│           └── visualizations/
│               ├── greedy/
│               └── nucleus_temp1.0_p0.9/
└── Stage_2/                           # Gutenberg V2
    └── [same structure]
```

## Customizing Generation Parameters

To use different nucleus sampling parameters, modify the `STRATEGY_ARGS` in the scripts:
```bash
# For more diverse generation:
STRATEGY_ARGS="--temperature 1.5 --top-p 0.95"

# For more focused generation:
STRATEGY_ARGS="--temperature 0.7 --top-p 0.8"
```

## Model Selection

To switch between 8B and 70B models, uncomment the appropriate lines in the scripts:
```bash
# For 8B model (default):
MODEL_PATH="swiss-ai/Apertus-8B-2509"
TENSOR_PARALLEL=2

# For 70B model:
MODEL_PATH="swiss-ai/Apertus-70B-2509"
TENSOR_PARALLEL=4  # or 8 for faster inference
```

## Notes
- The scripts automatically upgrade transformers to version 4.56.0 as required by Apertus models
- Results are automatically skipped if they already exist (use `--force` in main.py to reprocess)
- Each array task runs independently and can be restarted if needed