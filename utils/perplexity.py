"""
Perplexity and NLL calculation utilities for memorization analysis.

This module provides HuggingFace-based implementations for calculating perplexity
because vLLM is optimized for generation, not for computing log probabilities of
existing sequences. HuggingFace models provide direct access to logits via forward()
method, making perplexity calculation efficient and accurate.

Why HuggingFace for perplexity calculation:
1. Direct access to full logit distribution (not just top-k)
2. Efficient batch processing with forward pass
3. No need for generation overhead
4. Accurate perplexity calculation for existing sequences

Typical workflow:
1. Use vLLM for fast generation (Pass 1)
2. Unload vLLM to free memory
3. Load HuggingFace model for perplexity calculation (Pass 2)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def calc_nll_and_perplexity_hf(model, input_tensor, suffix_length):
    """
    Calculate NLL and perplexity for sequences using HuggingFace model.
    
    This function can be used for both reference (ground truth) and generated sequences.
    
    Backend: HuggingFace
    
    Args:
        model: HuggingFace model (AutoModelForCausalLM)
        input_tensor: Tensor of shape [batch_size, sequence_length]
                     Contains: [BOS, prefix, suffix]
                     Example: [1, 23, 45, 67, ..., 89] where 1 is BOS token
        suffix_length: Number of tokens in the suffix to calculate NLL for
    
    Returns:
        Tuple of (nll_mean, perplexity) - both as tensors
    """
    # Shift for teacher forcing: inputs[:-1] predicts targets[1:]
    inputs = input_tensor[:, :-1]  # [BOS, prefix, suffix[:-1]]
    targets = input_tensor[:, 1:]   # [prefix, suffix]

    outputs = model(inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    criterion = nn.CrossEntropyLoss(reduction="none")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    token_nlls_flat = criterion(flat_logits, flat_targets)
    token_nlls = token_nlls_flat.reshape(targets.shape)

    # Extract NLL for suffix tokens only
    # The last suffix_length tokens in targets correspond to the suffix
    suffix_token_nlls = token_nlls[:, -suffix_length:]
    nll_mean = suffix_token_nlls.mean(dim=1)
    perplexity = torch.exp(nll_mean)

    del flat_logits, flat_targets, token_nlls_flat, logits, outputs
    torch.cuda.empty_cache()

    return nll_mean, perplexity


def process_jsonl_for_perplexity_hf(
    model_path: str,
    input_jsonl: str,
    output_jsonl: str,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    calc_generation_ppl: bool = True
) -> None:
    """
    Process a JSONL file to add reference (and optionally generation) perplexity using HuggingFace.
    
    Backend: HuggingFace
    Reason: vLLM doesn't provide efficient perplexity calculation for existing sequences.
            This function is designed to run as Pass 2 after vLLM generation (Pass 1).
    
    Workflow:
        Pass 1: Generate completions with vLLM, save tokens to JSONL
        Pass 2: Run this function with HF model to add perplexity scores
    
    Expected JSONL format:
        {
            "prefix": [token_ids],        # e.g., [23, 45, 67, ...]
            "true_suffix": [token_ids],   # e.g., [89, 101, 112, ...]
            "generated_suffix": [token_ids]  # e.g., [90, 102, 113, ...]
        }
    
    Args:
        model_path: Path to HuggingFace model
        input_jsonl: Path to input JSONL with tokens from vLLM
        output_jsonl: Path to output JSONL with added perplexity
        batch_size: Batch size for processing
        device: Device to use (cuda/cpu)
        dtype: Model dtype (bfloat16 recommended for large models)
        calc_generation_ppl: Whether to also calculate generation perplexity (default: True)
    """
    print(f"Loading HuggingFace model from {model_path}...")
    print("Note: Ensure vLLM has been unloaded to free GPU memory")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    # Read all data
    data = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Processing {len(data)} samples with HuggingFace backend...")
    
    # Process in batches
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc="Calculating perplexity (HF)"):
        batch_data = data[i:i+batch_size]
        
        # Prepare batch tensors
        ref_sequences = []
        gen_sequences = []
        
        for item in batch_data:
            prefix = item['prefix']
            true_suffix = item['true_suffix']
            
            # Add BOS token if needed
            bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
            
            # Reference sequence: [BOS, prefix, true_suffix]
            ref_seq = [bos_id] + prefix + true_suffix
            ref_sequences.append(ref_seq)
            
            # Generated sequence: [BOS, prefix, generated_suffix]
            if calc_generation_ppl and 'generated_suffix' in item:
                gen_suffix = item['generated_suffix']
                gen_seq = [bos_id] + prefix + gen_suffix
                gen_sequences.append(gen_seq)
        
        # Pad sequences to same length for batch processing
        max_ref_len = max(len(s) for s in ref_sequences)
        ref_tensor = torch.zeros((len(batch_data), max_ref_len), dtype=torch.long, device=device)
        
        for idx, ref_seq in enumerate(ref_sequences):
            ref_tensor[idx, :len(ref_seq)] = torch.tensor(ref_seq)
        
        suffix_length = len(batch_data[0]['true_suffix'])
        
        # Calculate reference NLL and perplexity
        ref_nll_mean, ref_perplexity = calc_nll_and_perplexity_hf(
            model, ref_tensor, suffix_length
        )
        
        # Calculate generation NLL and perplexity (if requested)
        if calc_generation_ppl and gen_sequences:
            max_gen_len = max(len(s) for s in gen_sequences)
            gen_tensor = torch.zeros((len(batch_data), max_gen_len), dtype=torch.long, device=device)
            
            for idx, gen_seq in enumerate(gen_sequences):
                gen_tensor[idx, :len(gen_seq)] = torch.tensor(gen_seq)
            
            gen_nll_mean, gen_perplexity = calc_nll_and_perplexity_hf(
                model, gen_tensor, suffix_length
            )
        
        # Update batch data with perplexity values
        for j, item in enumerate(batch_data):
            item['ref_nll_mean'] = ref_nll_mean[j].item()
            item['ref_perplexity'] = ref_perplexity[j].item()
            
            if calc_generation_ppl and gen_sequences:
                item['nll_mean'] = gen_nll_mean[j].item()
                item['perplexity'] = gen_perplexity[j].item()
            
            results.append(item)
    
    # Write results
    print(f"Writing results to {output_jsonl}...")
    with open(output_jsonl, 'w') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')
    
    print("Done! Perplexity calculation complete using HuggingFace backend.")


def calculate_perplexity_from_logprobs(
    logprobs: List[float]
) -> float:
    """
    Calculate perplexity from a list of log probabilities.
    
    This is a utility function that can work with logprobs from any source
    (vLLM, HuggingFace, OpenAI API, etc.)
    
    Formula: perplexity = exp(mean(-log_probs))
    
    Args:
        logprobs: List of log probabilities
    
    Returns:
        Perplexity value
    """
    if not logprobs:
        return float('inf')
    
    avg_nll = -np.mean(logprobs)
    return float(np.exp(avg_nll))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate perplexity for vLLM-generated sequences"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to HuggingFace model"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Input JSONL file with tokens from vLLM"
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Output JSONL file with added perplexity scores"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--no-generation-ppl",
        action="store_true",
        help="Skip generation perplexity calculation"
    )
    
    args = parser.parse_args()
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    
    # Run perplexity calculation
    print(f"Processing {args.input_jsonl}...")
    process_jsonl_for_perplexity_hf(
        model_path=args.model_path,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype_map[args.dtype],
        calc_generation_ppl=not args.no_generation_ppl
    )
    
    print(f"Results saved to {args.output_jsonl}")