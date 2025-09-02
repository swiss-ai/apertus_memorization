"""
Gutenberg dataset loader for Apertus memorization analysis.

This module provides a unified interface for loading Gutenberg dataset token sequences
used in memorization studies. It supports both online (HuggingFace) and offline (local file)
loading with automatic mode detection.

Main Interface:
    load_gutenberg_tokens: Unified loader with automatic mode detection
        - Automatically chooses between online/offline based on parameters
        - Consistent interface for all loading scenarios
        
Backend Functions:
    load_gutenberg_tokens_online: HuggingFace dataset loading
        - Loads from swiss-ai/apertus-pretrain-gutenberg repository
        - Supports V1 (0-9T tokens) and V2 (9-12T tokens) datasets
        
    load_gutenberg_tokens_offline: Local JSONL file loading
        - Loads from local filesystem
        - Expected format: JSONL with 'input_ids' field
        
Helper Functions:
    extract_sequences: Common sequence extraction with offset/length handling
    validate_repetition: Validate and normalize repetition identifiers
    validate_version: Validate dataset version identifiers
    get_cache_directory: Manage cache directory creation and selection

Dataset Information:
    Versions:
        - V1: Gutenberg texts from 0-9T training tokens (500 sequences per repetition)
        - V2: Gutenberg texts from 9-12T training tokens (167 sequences per repetition)
        
    Repetitions:
        - Unseen: 0a, 0b, 0c, 0d (control sequences not in training data)
        - Seen: 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128 (N times in training)

Usage Examples:
    # Unified interface (auto-detects mode)
    sequences = load_gutenberg_tokens(
        version="v1", repetition="128",  # Online mode
        prefix_length=500, suffix_length=500
    )
    
    sequences = load_gutenberg_tokens(
        data_path="/path/to/data.jsonl",  # Offline mode
        prefix_length=500, suffix_length=500
    )
    
    # Explicit backend functions
    sequences = load_gutenberg_tokens_online("v1", "128", 500, 500)
    sequences = load_gutenberg_tokens_offline("/path/to/data.jsonl", 500, 500)
"""

import logging
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset, Features, Sequence, Value, load_dataset

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# HuggingFace repository
HUGGINGFACE_REPO: str = "swiss-ai/apertus-pretrain-gutenberg"

# Dataset version configurations
DATASET_VERSIONS: Dict[str, Dict[str, str]] = {
    "v1": {
        "path": "gutenberg_v1_bucket500",
        "description": "Gutenberg V1 (0-9T tokens)",
        "expected_count": 500  # Expected sequences per repetition
    },
    "v2": {
        "path": "gutenberg_v2_bucket167", 
        "description": "Gutenberg V2 (9-12T tokens)",
        "expected_count": 167  # Expected sequences per repetition
    }
}

# Repetition configurations
UNSEEN_REPETITIONS: List[str] = ['0a', '0b', '0c', '0d']  # Control sequences
SEEN_REPETITIONS: List[str] = ['1', '2', '4', '8', '16', '24', '32', '48', '64', '96', '128']
VALID_REPETITIONS: List[str] = UNSEEN_REPETITIONS + SEEN_REPETITIONS

# Dataset schema
TOKEN_FEATURES: Features = Features({"input_ids": Sequence(Value("int64"))})

# Default parameters
DEFAULT_PREFIX_LENGTH: int = 500
DEFAULT_SUFFIX_LENGTH: int = 500
DEFAULT_NUM_PROC: int = 4


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_repetition(repetition: Union[str, int]) -> str:
    """
    Validate and normalize repetition identifier.
    
    Args:
        repetition: Repetition identifier (string or int)
        
    Returns:
        Normalized repetition string
        
    Raises:
        ValueError: If repetition is not valid
    """
    rep_str = str(repetition)
    if rep_str not in VALID_REPETITIONS:
        raise ValueError(
            f"Invalid repetition '{repetition}'. Valid options are:\n"
            f"  - Unseen: {', '.join(UNSEEN_REPETITIONS)}\n"
            f"  - Seen: {', '.join(SEEN_REPETITIONS)}"
        )
    return rep_str


def validate_version(version: str) -> str:
    """
    Validate dataset version.
    
    Args:
        version: Dataset version identifier
        
    Returns:
        Normalized version string
        
    Raises:
        ValueError: If version is not valid
    """
    version_lower = version.lower()
    if version_lower not in DATASET_VERSIONS:
        valid_versions = ', '.join(f"'{v}'" for v in DATASET_VERSIONS.keys())
        raise ValueError(f"Invalid version '{version}'. Valid options are: {valid_versions}")
    return version_lower


def get_cache_directory(cache_dir: Optional[str] = None) -> str:
    """
    Get or create cache directory for datasets.
    
    Args:
        cache_dir: Optional cache directory path
        
    Returns:
        Path to cache directory
    """
    if cache_dir is None:
        # Use user's home directory to avoid permission issues
        arch = platform.machine()
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", "apertus_memorization", arch)
    
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Using cache directory: {cache_dir}")
    return cache_dir


# ============================================================================
# SEQUENCE EXTRACTION
# ============================================================================

def extract_sequences(
    dataset: Dataset,
    offset: int,
    prefix_length: int,
    suffix_length: int,
    num_proc: int = DEFAULT_NUM_PROC
) -> List[List[int]]:
    """
    Extract sequences from a dataset with given offset and lengths.
    
    Args:
        dataset: HuggingFace dataset with 'input_ids' field
        offset: Offset from start of sequence
        prefix_length: Number of tokens for prefix
        suffix_length: Number of tokens for suffix
        num_proc: Number of processes for parallel processing
        
    Returns:
        List of token sequences (prefix + suffix)
    """
    required_length = offset + prefix_length + suffix_length
    
    def extract_sequence(batch):
        sequences = []
        for input_ids in batch["input_ids"]:
            # Validate sequence length
            if len(input_ids) < required_length:
                logger.warning(
                    f"Sequence too short: {len(input_ids)} < {required_length}"
                )
                continue
            
            # Extract prefix and suffix
            start_idx = offset
            end_idx = offset + prefix_length + suffix_length
            sequences.append(input_ids[start_idx:end_idx])
        
        return {"sequence": sequences}
    
    # Process dataset in parallel
    processed = dataset.map(
        extract_sequence,
        batched=True,
        num_proc=num_proc,
        desc="Extracting sequences"
    )
    
    # Extract and return sequences
    sequences = processed["sequence"]
    logger.info(
        f"Extracted {len(sequences)} sequences "
        f"(offset={offset}, prefix={prefix_length}, suffix={suffix_length})"
    )
    
    return sequences


# ============================================================================
# ONLINE LOADING (HUGGINGFACE)
# ============================================================================

def load_gutenberg_tokens_online(
    version: str = "v1",
    repetition: Union[str, int] = "1",
    prefix_length: int = DEFAULT_PREFIX_LENGTH,
    suffix_length: int = DEFAULT_SUFFIX_LENGTH,
    offset: int = 0,
    num_proc: int = DEFAULT_NUM_PROC,
    cache_dir: Optional[str] = None
) -> List[List[int]]:
    """
    Load Gutenberg token sequences from HuggingFace dataset (online).
    
    Args:
        version: Dataset version ("v1" for 0-9T training, "v2" for 9-12T training)
        repetition: Repetition identifier. Available options:
                   - '0a', '0b', '0c', '0d' (unseen sequences)
                   - 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128 (seen sequences)
        prefix_length: Number of tokens for prefix
        suffix_length: Number of tokens for suffix
        offset: Offset from start of sequence  
        num_proc: Number of processes for parallel loading
        cache_dir: Cache directory for datasets (None = auto-detect)
        
    Returns:
        List of token sequences (prefix + suffix)
    
    Raises:
        ValueError: If repetition or version is not valid
    """
    # Validate inputs
    rep_str = validate_repetition(repetition)
    version_norm = validate_version(version)
    
    # Get dataset configuration
    dataset_config = DATASET_VERSIONS[version_norm]
    data_files = f"{dataset_config['path']}/rep_{rep_str}_token.jsonl"
    logger.info(f"Loading {dataset_config['description']} rep_{rep_str}")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(
        HUGGINGFACE_REPO,
        data_files=data_files,
        split="train",
        cache_dir=cache_dir,
        features=TOKEN_FEATURES
    )
    
    logger.info(f"Loaded {len(dataset)} sequences from HuggingFace")
    
    # Extract and return sequences
    return extract_sequences(dataset, offset, prefix_length, suffix_length, num_proc)


# ============================================================================
# OFFLINE LOADING (LOCAL FILES)
# ============================================================================

def load_gutenberg_tokens_offline(
    data_path: str,
    prefix_length: int,
    suffix_length: int,
    offset: int = 0,
    num_proc: int = DEFAULT_NUM_PROC,
    cache_dir: Optional[str] = None
) -> List[List[int]]:
    """
    Load Gutenberg token sequences from local JSONL file (offline).
    
    Args:
        data_path: Path to JSONL file with token sequences
        prefix_length: Number of tokens for prefix
        suffix_length: Number of tokens for suffix
        offset: Offset from start of sequence
        num_proc: Number of processes for parallel loading
        cache_dir: Cache directory for datasets (None = auto-detect)
        
    Returns:
        List of token sequences (prefix + suffix)
        
    Raises:
        FileNotFoundError: If data_path does not exist
    """
    # Validate input file exists
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    
    # Get cache directory
    cache_dir = get_cache_directory(cache_dir)
    
    # Load dataset from local file
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
        cache_dir=cache_dir,
        features=TOKEN_FEATURES
    )
    
    logger.info(f"Loaded {len(dataset)} sequences from {data_file.name}")
    
    # Extract and return sequences
    return extract_sequences(dataset, offset, prefix_length, suffix_length, num_proc)


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def load_gutenberg_tokens(
    prefix_length: int = DEFAULT_PREFIX_LENGTH,
    suffix_length: int = DEFAULT_SUFFIX_LENGTH,
    offset: int = 0,
    num_proc: int = DEFAULT_NUM_PROC,
    cache_dir: Optional[str] = None,
    # Online parameters
    version: Optional[str] = None,
    repetition: Optional[Union[str, int]] = None,
    # Offline parameters
    data_path: Optional[str] = None,
    # Mode selection
    mode: str = "auto"
) -> List[List[int]]:
    """
    Unified interface for loading Gutenberg token sequences.
    
    This function automatically selects between online (HuggingFace) and offline (local file)
    loading based on the provided parameters.
    
    Args:
        prefix_length: Number of tokens for prefix
        suffix_length: Number of tokens for suffix
        offset: Offset from start of sequence
        num_proc: Number of processes for parallel loading
        cache_dir: Cache directory for datasets (None = auto-detect)
        
        Online mode parameters:
            version: Dataset version ("v1" or "v2") - required for online mode
            repetition: Repetition identifier (e.g., '1', '0a') - required for online mode
            
        Offline mode parameters:
            data_path: Path to local JSONL file - required for offline mode
            
        mode: Loading mode ("auto", "online", "offline")
            - "auto": Automatically determine based on provided parameters
            - "online": Force online loading from HuggingFace
            - "offline": Force offline loading from local file
    
    Returns:
        List of token sequences (prefix + suffix)
        
    Raises:
        ValueError: If required parameters are missing or invalid
        
    Examples:
        # Online loading
        sequences = load_gutenberg_tokens(
            version="v1", 
            repetition="1",
            prefix_length=500,
            suffix_length=500
        )
        
        # Offline loading
        sequences = load_gutenberg_tokens(
            data_path="/path/to/rep_1_token.jsonl",
            prefix_length=500,
            suffix_length=500
        )
    """
    # Determine mode
    if mode == "auto":
        if data_path is not None:
            mode = "offline"
        elif version is not None and repetition is not None:
            mode = "online"
        else:
            raise ValueError(
                "Cannot determine loading mode. Provide either:\n"
                "  - 'version' and 'repetition' for online loading, or\n"
                "  - 'data_path' for offline loading"
            )
    
    # Load based on mode
    if mode == "online":
        if version is None or repetition is None:
            raise ValueError("Online mode requires 'version' and 'repetition' parameters")
        return load_gutenberg_tokens_online(
            version=version,
            repetition=repetition,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            offset=offset,
            num_proc=num_proc,
            cache_dir=cache_dir
        )
    elif mode == "offline":
        if data_path is None:
            raise ValueError("Offline mode requires 'data_path' parameter")
        return load_gutenberg_tokens_offline(
            data_path=data_path,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            offset=offset,
            num_proc=num_proc,
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'auto', 'online', or 'offline'")


# ============================================================================
# TESTING
# ============================================================================

def _run_tests() -> List[Tuple[str, bool, str]]:
    """Run basic functionality tests."""
    test_results = []
    
    # Test 1: Online loading with V1
    try:
        sequences = load_gutenberg_tokens_online("v1", "1", 50, 50)
        test_results.append(("Online V1 loading", True, f"{len(sequences)} sequences"))
    except Exception as e:
        test_results.append(("Online V1 loading", False, str(e)))
    
    # Test 2: Online loading with V2 unseen
    try:
        sequences = load_gutenberg_tokens_online("v2", "0a", 100, 100, offset=10)
        test_results.append(("Online V2 unseen", True, f"{len(sequences)} sequences"))
    except Exception as e:
        test_results.append(("Online V2 unseen", False, str(e)))
    
    # Test 3: Validation error handling
    try:
        sequences = load_gutenberg_tokens_online("v1", "999", 50, 50)
        test_results.append(("Validation error", False, "Should have raised error"))
    except ValueError:
        test_results.append(("Validation error", True, "Correctly rejected invalid input"))
    
    # Test 4: Unified interface
    try:
        sequences = load_gutenberg_tokens(version="v1", repetition="2", 
                                         prefix_length=25, suffix_length=25)
        test_results.append(("Unified interface", True, f"{len(sequences)} sequences"))
    except Exception as e:
        test_results.append(("Unified interface", False, str(e)))
    
    return test_results


if __name__ == "__main__":
    """Test module functionality."""
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("\n" + "=" * 70)
    print(" Gutenberg Loader Module Tests ".center(70))
    print("=" * 70)
    
    # Run tests
    results = _run_tests()
    
    # Display results
    print("\nTest Results:")
    print("-" * 70)
    for test_name, passed, details in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:<8} | {test_name:<25} | {details}")
    
    # Summary
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print("-" * 70)
    print(f"Summary: {passed_count}/{total_count} tests passed")
    print("=" * 70 + "\n")


