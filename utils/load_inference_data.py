"""
Data loading utilities for Apertus inference results.
"""

from pathlib import Path
from datasets import load_dataset
from typing import Optional, Any


def load_inference_data(
    base_dir: str,
    rep: int,
    policy: str,
    offset: int,
    len_prefix: int,
    len_suffix: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None
) -> Optional[Any]:
    """
    Load inference data from JSONL files using HuggingFace datasets.
    
    HuggingFace datasets support direct index access (dataset[idx]) and are
    memory-efficient for large files.
    
    Parameters
    ----------
    base_dir : str
        Base directory path to the model (e.g., path/to/Apertus70B-tokens15T-it1155828)
    rep : int
        Repetition number
    policy : str
        Generation policy (e.g., 'greedy', 'nucleus')
    offset : int
        Offset value
    len_prefix : int
        Prefix length
    len_suffix : int
        Suffix length
    temperature : float, optional
        Temperature for nucleus sampling (required if policy='nucleus')
    top_p : float, optional
        Top-p value for nucleus sampling (required if policy='nucleus')
        
    Returns
    -------
    Dataset or None
        HuggingFace dataset if file exists, None otherwise
    """
    # Construct policy folder name
    if policy == 'nucleus':
        if temperature is None or top_p is None:
            raise ValueError("temperature and top_p are required for nucleus sampling")
        policy_folder = f"nucleus_temp{temperature}_p{top_p}"
    else:
        policy_folder = policy
    
    # Construct the file path
    # base_dir should already point to the model directory
    file_path = Path(base_dir) / f"offset_{offset}_prefix_{len_prefix}_suffix_{len_suffix}" / "inferences" / policy_folder / f"rep_{rep}_{policy}.jsonl"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Inference data not found at: {file_path}\n"
            f"Please check:\n"
            f"  - Model directory: {base_dir}\n"
            f"  - Offset: {offset}, Prefix: {len_prefix}, Suffix: {len_suffix}\n"
            f"  - Repetition: {rep}, Policy: {policy_folder}"
        )
    
    # Load the dataset using HuggingFace datasets for efficiency
    dataset = load_dataset(
        'json',
        data_files=str(file_path),
        split='train'
    )
    
    return dataset