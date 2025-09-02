#!/usr/bin/env python
"""
Main script for Apertus memorization analysis using HuggingFace datasets.

This script runs vLLM-based inference on Gutenberg datasets loaded directly from HuggingFace.
It can process multiple repetitions in a single run with automatic data loading from the cloud.

Example usage:
    # Greedy (deterministic) generation with Apertus 8B and Gutenberg V1:
    python main.py \
        --model-path swiss-ai/Apertus-8B-2509 \
        --version v1 \
        --output-dir ./inferences \
        --repetitions 1 8 32 128 \
        --prefix-length 500 \
        --suffix-length 500 \
        --offset 0 \
        --tensor-parallel-size 2 \
        --strategy greedy \
        --show-top 5
    
    # Nucleus sampling with Apertus 70B and Gutenberg V2:
    python main.py \
        --model-path swiss-ai/Apertus-70B-2509 \
        --version v2 \
        --output-dir ./inferences \
        --repetitions 0a 0b 1 2 4 8 16 32 64 128 \
        --prefix-length 50 100 250 500 1000 2000 3000 4000 5000 \
        --suffix-length 500 \
        --offset 0 \
        --tensor-parallel-size 4 \
        --strategy nucleus \
        --temperature 1.0 \
        --top-p 0.9 \
        --show-top 5

Available models for memorization analysis (HuggingFace):
    - swiss-ai/Apertus-8B-2509: 8B parameter base model (TP=2 recommended)
    - swiss-ai/Apertus-70B-2509: 70B parameter base model (TP=4-8 recommended)
    
    Note: Use base models only for memorization analysis, not instruction-tuned versions.

Dataset versions:
    - v1: Gutenberg texts from 0-9T training tokens (500 sequences/rep)
    - v2: Gutenberg texts from 9-12T training tokens (167 sequences/rep)
"""

import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from itertools import product


# Fix import paths for both direct execution and module import
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.inference import ApertusInference
from core.gutenberg_loader import load_gutenberg_tokens
from utils.comparison_html import create_comparison_html

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_repetition(
    inference_engine: ApertusInference,
    output_dir: Path,
    repetition: str,
    prefix_length: int,
    suffix_length: int,
    offset: int,
    batch_size: int,
    strategy: str,
    version: str = "v1",  # Dataset version for online loading
    cache_dir: str = None,
    show_top: int = 0,
    **generation_kwargs
) -> str:
    """
    Process a single repetition using online data loading.
    
    Args:
        inference_engine: Initialized ApertusInference object
        output_dir: Output directory for inferences
        repetition: Repetition identifier (e.g., '1', '32', '0a', '0b')
        prefix_length: Prefix length in tokens
        suffix_length: Suffix length in tokens
        offset: Offset from sequence start
        batch_size: Batch size for processing
        strategy: Generation strategy
        version: Dataset version ('v1' or 'v2') for online loading
        cache_dir: Cache directory for datasets
        show_top: Number of top samples to visualize
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Path to output file
    """
    logger.info(f"\nProcessing repetition {repetition} from Gutenberg {version.upper()}")
    logger.info(f"Loading from HuggingFace (online mode)")
    
    # Load data using online mode
    try:
        samples = load_gutenberg_tokens(
            version=version,
            repetition=repetition,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            offset=offset,
            cache_dir=cache_dir
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None
    
    if not samples:
        logger.warning(f"No valid samples found for rep_{repetition}")
        return None
    
    logger.info(f"Loaded {len(samples)} sequences from HuggingFace")
    
    # Create strategy-specific inferences subdirectory and set output path
    if strategy == "greedy":
        strategy_folder = "greedy"
    elif strategy == "nucleus":
        # Include temperature and top_p in folder name for nucleus sampling
        temp = generation_kwargs.get('temperature')
        top_p = generation_kwargs.get('top_p')
        assert temp is not None and top_p is not None, "Temperature and top_p must be specified for nucleus sampling"
        strategy_folder = f"nucleus_temp{temp}_p{top_p}"
    else:
        strategy_folder = strategy
    
    inferences_dir = output_dir / "inferences" / strategy_folder
    inferences_dir.mkdir(parents=True, exist_ok=True)
    output_file = inferences_dir / f"rep_{repetition}_{strategy}.jsonl"
    
    # Check if already processed
    if output_file.exists():
        logger.info(f"Output already exists: {output_file}")
        logger.info("Skipping this repetition (use --force to reprocess)")
        return str(output_file)
    
    # Process dataset
    results = inference_engine.process_dataset(
        samples=samples,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        output_jsonl=str(output_file),
        batch_size=batch_size,
        strategy=strategy,
        show_progress=True,
        **generation_kwargs
    )
    
    # Log statistics
    logger.info(f"Completed repetition {repetition}")
    logger.info(f"Output: {output_file}")
    
    # Calculate average metrics
    avg_rouge = sum(r["Rouge-L"] for r in results) / len(results)
    avg_lcs = sum(r["lcs_norm"] for r in results) / len(results)
    avg_ttr_ref = sum(r["TTR_ref"] for r in results) / len(results)
    avg_ttr_gen = sum(r["TTR_gen"] for r in results) / len(results)
    logger.info(f"Average ROUGE-L: {avg_rouge:.4f}")
    logger.info(f"Average LCS norm: {avg_lcs:.4f}")
    logger.info(f"Average TTR (ref): {avg_ttr_ref:.4f}")
    logger.info(f"Average TTR (gen): {avg_ttr_gen:.4f}")
    
    # Generate HTML visualizations if requested
    if show_top > 0:
        logger.info(f"\nGenerating HTML visualizations for top {show_top} samples...")
        # Create strategy and repetition-specific subdirectory
        vis_dir = output_dir / "visualizations" / strategy_folder / f"rep_{repetition}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Add index to each result before sorting
        indexed_results = [(i, result) for i, result in enumerate(results)]
        
        # Sort by ROUGE-L score
        sorted_indexed_results = sorted(indexed_results, key=lambda x: x[1]["Rouge-L"], reverse=True)
        
        # Generate visualizations for top samples
        for rank, (original_idx, result) in enumerate(sorted_indexed_results[:show_top], 1):
            filename = f"sample_{original_idx:03d}_rep{repetition}_{strategy}_rouge{result['Rouge-L']:.2f}_lcs{result['lcs_norm']:.3f}.html"
            output_path = vis_dir / filename
            
            create_comparison_html(
                prefix_tokens=result["prefix"],
                true_suffix_tokens=result["true_suffix"],
                generated_suffix_tokens=result["generated_suffix"],
                tokenizer=inference_engine.tokenizer,
                repetition=repetition,
                strategy=strategy,
                sample_idx=original_idx,  # Use original sample index
                rouge_score=result["Rouge-L"],
                lcs_score=result["lcs_norm"],
                output_path=str(output_path),
                show_token_ids=False
            )
            logger.info(f"  [Rank {rank}] Sample {original_idx:03d}: ROUGE-L={result['Rouge-L']:.3f}, LCS={result['lcs_norm']:.3f} -> {filename}")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Apertus memorization analysis with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Model path or HuggingFace ID (e.g., swiss-ai/Apertus-8B-2509, swiss-ai/Apertus-70B-2509)"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="GPU memory utilization fraction"
    )
    
    # Data arguments
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Gutenberg dataset version (v1: 0-9T, v2: 9-12T)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Base output directory for inferences"
    )
    parser.add_argument(
        "--repetitions", 
        type=str, 
        nargs='+',
        required=True,
        help="Space-separated list of repetitions (e.g., 1 8 32 128 or 0a 0b 0c 0d)"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default=None,
        help="Cache directory for HuggingFace datasets (default: auto)"
    )
    
    # Generation arguments - support multiple values with space separation
    parser.add_argument(
        "--prefix-length", 
        type=int, 
        nargs='+',
        required=True,
        help="Space-separated prefix lengths (e.g., 50 100 500)"
    )
    parser.add_argument(
        "--suffix-length", 
        type=int, 
        nargs='+',
        required=True,
        help="Space-separated suffix lengths (e.g., 50 100 500)"
    )
    parser.add_argument(
        "--offset", 
        type=int, 
        nargs='+',
        required=True,
        help="Space-separated offsets (e.g., 0 100 500)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size for processing (None for automatic vLLM batching)"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="greedy",
        choices=["greedy", "nucleus", "beam"],
        help="Generation strategy"
    )
    
    # Strategy-specific arguments
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Temperature for nucleus sampling"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--num-beams", 
        type=int, 
        default=5,
        help="Number of beams for beam search"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reprocessing even if output exists"
    )
    parser.add_argument(
        "--max-model-len", 
        type=int, 
        default=None,
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=0,
        help="Number of top memorized samples to show as HTML visualizations (0 = none)"
    )
    
    args = parser.parse_args()
    
    # Validate nucleus sampling parameters
    if args.strategy == "nucleus" and (args.temperature == 0.0 or args.top_p == 0.0):
        parser.error("Nucleus sampling requires non-zero temperature and top_p values. "
                    "Typical values: temperature=0.7-1.5, top_p=0.9-0.95")
    
    # repetitions is now already a list of ints
    repetitions = args.repetitions
    prefix_lengths = args.prefix_length
    suffix_lengths = args.suffix_length
    offsets = args.offset
    
    logger.info(f"Will process:")
    logger.info(f"  Dataset version: Gutenberg {args.version.upper()}")
    logger.info(f"  Repetitions: {repetitions}")
    logger.info(f"  Prefix lengths: {prefix_lengths}")
    logger.info(f"  Suffix lengths: {suffix_lengths}")
    logger.info(f"  Offsets: {offsets}")
    
    # Create output directory structure with model name
    base_output_dir = Path(args.output_dir)
    
    # Determine model name for output directory
    if "/" in args.model_path and not args.model_path.startswith("/"):
        # HuggingFace model ID (e.g., "swiss-ai/Apertus-8B-2509")
        model_name = args.model_path.replace("/", "_")
        logger.info(f"Model: {args.model_path} (HuggingFace)")
    else:
        # Local model path
        model_name = Path(args.model_path).name
        logger.info(f"Model: {model_name} (local)")
    
    base_output_dir = base_output_dir / model_name
    
    logger.info(f"Output directory: {base_output_dir}")
    logger.info(f"Data source: HuggingFace Gutenberg {args.version.upper()} (online)")
    
    # Initialize inference engine ONCE
    logger.info("\nInitializing inference engine (loading model once)...")
    inference_engine = ApertusInference(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        max_model_len=args.max_model_len
    )
    logger.info("Model loaded successfully - will reuse for all configurations")
    
    # Prepare generation kwargs
    generation_kwargs = {}
    if args.strategy == "nucleus":
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
        logger.info(f"Using nucleus sampling: temperature={args.temperature}, top_p={args.top_p}")
    elif args.strategy == "beam":
        generation_kwargs["num_beams"] = args.num_beams
        logger.info(f"Using beam search: num_beams={args.num_beams}")
    else:  # greedy
        logger.info("Using greedy decoding (deterministic)")
    
    # Process all combinations with the same model
    processed_files = []
    
    # Create all configurations using itertools.product
    configurations = list(product(offsets, prefix_lengths, suffix_lengths, repetitions))
    
    # Process with progress bar showing detailed information
    pbar = tqdm(configurations, unit="config")
    for offset, prefix_length, suffix_length, rep in pbar:
        # Create config-specific output directory with subdirectories
        config_name = f"offset_{offset}_prefix_{prefix_length}_suffix_{suffix_length}"
        output_dir = base_output_dir / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create inferences and visualizations subdirectories
        (output_dir / "inferences").mkdir(parents=True, exist_ok=True)
        (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        
        # Update progress bar description with current configuration
        pbar.set_description(f"Processing {args.strategy} | off:{offset} pre:{prefix_length} suf:{suffix_length} rep:{rep}")
        
        logger.info(f"Processing: {config_name}, rep_{rep}")
        
        output_file = process_repetition(
            inference_engine,
            output_dir,
            rep,
            prefix_length,
            suffix_length,
            offset,
            args.batch_size,
            args.strategy,
            version=args.version,
            cache_dir=args.cache_dir,
            show_top=args.show_top,
            **generation_kwargs
        )
        
        if output_file:
            processed_files.append(output_file)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Processed {len(processed_files)} repetition(s)")
    logger.info(f"Inferences directory: {output_dir}")
    
    if processed_files:
        logger.info("\nGenerated files:")
        for f in processed_files:
            logger.info(f"  - {Path(f).name}")
    
    logger.info("\nNext steps:")
    logger.info("1. Run perplexity calculation with HuggingFace (if needed)")
    logger.info("2. Use controlled_expr.py for detailed analysis")
    logger.info("3. View HTML files in visualizations/ subdirectory")
    logger.info("=" * 60)
    
    # Cleanup
    inference_engine.cleanup()
    logger.info("\nInference engine cleaned up successfully")


if __name__ == "__main__":
    main()