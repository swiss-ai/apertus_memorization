"""
Apertus vLLM-based inference module for memorization analysis.

This module provides efficient batch inference using vLLM for text generation
and metrics calculation for memorization studies. It outputs JSON files
compatible with the existing analysis pipeline.
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Fix imports for metrics module
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import metrics at module level
from metrics import calculate_ttr, calculate_rouge_l, calculate_lcs_norm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ApertusInference:
    """
    vLLM-based inference engine for memorization analysis.
    
    This class handles:
    - Model loading with vLLM
    - Batch generation with configurable strategies
    - Metrics calculation (TTR, ROUGE-L, LCS)
    - JSON output in standard format
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float = 0.9,
        seed: int = 42,
        max_model_len: Optional[int] = None
    ):
        """
        Initialize the Apertus inference engine.
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
                Local: /path/to/model
                HuggingFace: swiss-ai/Apertus-8B-2509, swiss-ai/Apertus-70B-2509
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            seed: Random seed for reproducibility
            max_model_len: Maximum sequence length (None for auto)
        """
        # Determine if model is from HuggingFace or local
        if "/" in model_path and not model_path.startswith("/") and not model_path.startswith("."):
            logger.info(f"Loading model from HuggingFace: {model_path}")
        else:
            logger.info(f"Loading model from local path: {model_path}")
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Initialize vLLM
        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            "enforce_eager": True,  # Disable CUDA graphs for determinism
        }
        
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
            
        self.llm = LLM(**llm_kwargs)
        
        # Ensure BOS token is available
        self.bos_id = self.tokenizer.bos_token_id
        assert self.bos_id is not None, (
            f"BOS token not found in tokenizer for model {model_path}. "
            "The model tokenizer must have a BOS token defined."
        )
        
        logger.info("Model loaded successfully")
        self._log_config()
    
    def _log_config(self):
        """Log vLLM configuration details."""
        engine = self.llm.llm_engine
        cfg = engine.vllm_config
        
        logger.info("vLLM Configuration:")
        logger.info(f"  Model: {cfg.model_config.model}")
        logger.info(f"  Data type: {cfg.model_config.dtype}")
        logger.info(f"  Max model length: {cfg.model_config.max_model_len}")
        logger.info(f"  Tensor parallel size: {cfg.parallel_config.tensor_parallel_size}")
        logger.info(f"  GPU memory utilization: {cfg.cache_config.gpu_memory_utilization:.1%}")
        logger.info(f"  Block size: {cfg.cache_config.block_size}")
        logger.info(f"  Max num seqs: {cfg.scheduler_config.max_num_seqs}")
        logger.info(f"  Max num batched tokens: {cfg.scheduler_config.max_num_batched_tokens}")
    
    def generate_batch(
        self,
        prefixes: List[List[int]],
        suffix_length: int,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_beams: int = 5
    ) -> Tuple[List[List[int]], List[str]]:
        """
        Generate completions for a batch of prefixes.
        
        Args:
            prefixes: List of token ID lists (without BOS)
            suffix_length: Number of tokens to generate
            strategy: Generation strategy ('greedy', 'nucleus', 'beam')
            temperature: Temperature for sampling
            top_p: Top-p value for nucleus sampling
            num_beams: Number of beams for beam search (not yet implemented)
            
        Returns:
            Tuple of (generated_token_ids, generated_texts)
        """
        # Prepare inputs with BOS token
        inputs_with_bos = []
        for prefix in prefixes:
            inputs_with_bos.append([self.bos_id] + prefix)
        
        # Configure sampling parameters
        if strategy == "greedy":
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1.0,
                max_tokens=suffix_length,
                min_tokens=suffix_length
            )
        elif strategy == "nucleus":
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=suffix_length,
                min_tokens=suffix_length,
                seed=42  # For reproducibility
            )
        elif strategy == "beam":
            raise NotImplementedError(
                "Beam search is not yet implemented in vLLM for this use case. "
                "Please use 'greedy' or 'nucleus' strategy instead."
            )
        else:
            raise ValueError(f"Unknown generation strategy: {strategy}. Supported: 'greedy', 'nucleus', 'beam' (not yet implemented)")
        
        # Generate with optional progress bar
        logger.info(f"Starting generation for {len(inputs_with_bos)} prompts...")
        outputs = self.llm.generate(
            prompt_token_ids=inputs_with_bos,
            sampling_params=sampling_params,
            use_tqdm=True  # Show progress bar for generation
        )
        
        # Extract generated tokens and texts
        generated_tokens = []
        generated_texts = []
        
        for output in outputs:
            tokens = output.outputs[0].token_ids
            text = output.outputs[0].text
            generated_tokens.append(tokens)
            generated_texts.append(text)
        
        return generated_tokens, generated_texts
    
    def process_dataset(
        self,
        samples: List[List[int]],
        prefix_length: int,
        suffix_length: int,
        output_jsonl: str,
        batch_size: int = None,
        strategy: str = "greedy",
        show_progress: bool = True,
        **generation_kwargs
    ) -> List[Dict]:
        """
        Process a dataset of token sequences.
        
        Args:
            samples: List of token sequences, each already offset-adjusted and 
                     containing exactly prefix_length + suffix_length tokens
            prefix_length: Number of tokens to use as prefix (from start of sample)
            suffix_length: Number of tokens to use as suffix (following the prefix)
            output_jsonl: Path to save results JSONL file
            batch_size: Batch size for processing (None for automatic vLLM batching)
            strategy: Generation strategy ('greedy' or 'nucleus')
            show_progress: Show progress bar
            **generation_kwargs: Additional generation parameters (temperature, top_p)
            
        Returns:
            List of result dictionaries with keys: prefix, true_suffix, 
            generated_suffix, lcs_norm, TTR_ref, TTR_gen, Rouge-L
        """
        results = []
        
        # Create output file
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(output_path, 'w')
        logger.info(f"Writing results to {output_jsonl}")
        
        # Process in batches
        if batch_size is None:
            # Let vLLM handle batching automatically - process all at once
            batch_samples_list = [samples]  # Single batch with all samples
            desc = "Processing with vLLM auto-batching"
        else:
            # Manual batching with specified batch size
            num_batches = (len(samples) + batch_size - 1) // batch_size
            batch_samples_list = []
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batch_samples_list.append(samples[batch_start:batch_end])
            desc = "Processing batches"
        
        if show_progress:
            iterator = tqdm(batch_samples_list, total=len(batch_samples_list), desc=desc)
        else:
            iterator = batch_samples_list
        
        for batch_samples in iterator:
            logger.info(f"Processing batch with {len(batch_samples)} samples")
            
            # Extract prefixes and true suffixes using list comprehensions
            batch_prefixes = [sample[:prefix_length] for sample in batch_samples]
            batch_true_suffixes = [sample[prefix_length:prefix_length + suffix_length] for sample in batch_samples]
            
            logger.info(f"Batch has {len(batch_prefixes)} prefixes, {len(batch_true_suffixes)} suffixes")
            
            # Generate completions with timing
            start_time = time.time()
            generated_tokens, generated_texts = self.generate_batch(
                batch_prefixes,
                suffix_length,
                strategy,
                **generation_kwargs
            )
            inference_time = time.time() - start_time
            
            logger.info(f"Generated {len(generated_tokens)} completions in {inference_time:.2f} seconds")
            logger.info(f"Speed: {len(generated_tokens) / inference_time:.2f} samples/second")
            
            # Calculate metrics and save results
            for i, (prefix, true_suffix, gen_tokens) in enumerate(
                zip(batch_prefixes, batch_true_suffixes, generated_tokens)
            ):
                # Calculate metrics
                metrics = self._calculate_metrics(true_suffix, gen_tokens)
                
                # Create result dictionary
                result = {
                    "prefix": prefix,
                    "true_suffix": true_suffix,
                    "generated_suffix": gen_tokens,
                    "lcs_norm": metrics["lcs_norm"],
                    "TTR_ref": metrics["TTR_ref"],
                    "TTR_gen": metrics["TTR_gen"],
                    "Rouge-L": metrics["Rouge-L"]
                }
                
                results.append(result)
                
                # Write to file immediately
                json.dump(result, jsonl_file)
                jsonl_file.write('\n')
                jsonl_file.flush()
                
            logger.info(f"Wrote {len(batch_prefixes)} results to file")
        
        # Close output file
        jsonl_file.close()
        logger.info(f"Saved {len(results)} results to {output_jsonl}")
        
        return results
    
    def _calculate_metrics(
        self,
        true_suffix: List[int],
        generated_suffix: List[int]
    ) -> Dict[str, float]:
        """
        Calculate TTR, ROUGE-L, and LCS metrics.
        
        Args:
            true_suffix: Ground truth token IDs
            generated_suffix: Generated token IDs
            
        Returns:
            Dictionary with metric values
        """
        # Type-Token Ratio
        ttr_ref = calculate_ttr(true_suffix)
        ttr_gen = calculate_ttr(generated_suffix)
        
        # ROUGE-L
        rouge_l = calculate_rouge_l(true_suffix, generated_suffix)
        
        # Longest Common Subsequence (normalized)
        lcs_norm = calculate_lcs_norm(true_suffix, generated_suffix)
        
        return {
            "TTR_ref": ttr_ref,
            "TTR_gen": ttr_gen,
            "Rouge-L": rouge_l,
            "lcs_norm": lcs_norm
        }
    
    def cleanup(self):
        """Clean up resources."""
        # vLLM handles cleanup automatically
        logger.info("Inference engine cleaned up")


def run_inference(
    model_path: str,
    output_dir: str,
    repetition: Union[str, int],
    prefix_length: int = 50,
    suffix_length: int = 50,
    offset: int = 0,
    batch_size: int = None,
    strategy: str = "greedy",
    tensor_parallel_size: int = 1,
    # Data source parameters
    data_path: Optional[str] = None,  # For offline mode
    version: Optional[str] = None,    # For online mode
    use_online: bool = False,         # Force online mode
    cache_dir: Optional[str] = None,  # Cache for HuggingFace datasets
    **kwargs
) -> str:
    """
    Run inference on a single repetition.
    
    Args:
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        repetition: Repetition identifier (e.g., '1', '128', '0a')
        prefix_length: Prefix length in tokens
        suffix_length: Suffix length in tokens
        offset: Offset from start of sequence
        batch_size: Batch size for processing (None for automatic vLLM batching)
        strategy: Generation strategy
        tensor_parallel_size: Number of GPUs
        
        Data source parameters:
            data_path: Path to local JSONL file (offline mode)
            version: Dataset version 'v1' or 'v2' (online mode)
            use_online: If True, use HuggingFace online loading
            cache_dir: Cache directory for datasets
            
        **kwargs: Additional generation parameters
        
    Returns:
        Path to output file
    """
    # Load data using unified interface
    from .gutenberg_loader import load_gutenberg_tokens
    
    # Determine data loading mode
    if use_online or version is not None:
        # Online mode
        if version is None:
            raise ValueError("Online mode requires 'version' parameter (v1 or v2)")
        logger.info(f"Loading Gutenberg {version.upper()} rep_{repetition} from HuggingFace")
        samples = load_gutenberg_tokens(
            version=version,
            repetition=repetition,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            offset=offset,
            cache_dir=cache_dir
        )
    else:
        # Offline mode
        if data_path is None:
            raise ValueError("Offline mode requires 'data_path' parameter")
        logger.info(f"Loading data from {data_path}")
        samples = load_gutenberg_tokens(
            data_path=data_path,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            offset=offset,
            cache_dir=cache_dir
        )
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Initialize inference engine
    engine = ApertusInference(
        model_path,
        tensor_parallel_size=tensor_parallel_size
    )
    
    # Set output path
    output_path = Path(output_dir) / f"rep_{repetition}_{strategy}.jsonl"
    
    # Process dataset
    results = engine.process_dataset(
        samples,
        prefix_length,
        suffix_length,
        batch_size,
        strategy,
        str(output_path),
        **kwargs
    )
    
    # Cleanup
    engine.cleanup()
    
    logger.info(f"Completed inference for repetition {repetition}")
    return str(output_path)