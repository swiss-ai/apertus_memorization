"""
Metrics calculation for memorization analysis.

This module provides self-contained implementations of metrics used in
memorization studies, including TTR, ROUGE-L, and LCS.
"""

import numpy as np
from typing import List, Union, Tuple
from numba import jit


def calculate_ttr(tokens: List[int]) -> float:
    """
    Calculate Type-Token Ratio (TTR).
    
    TTR = (number of unique tokens) / (total number of tokens)
    
    Args:
        tokens: List of token IDs
        
    Returns:
        TTR value between 0 and 1
    """
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def calculate_rouge_l(
    reference: List[int],
    candidate: List[int]
) -> float:
    """
    Calculate ROUGE-L score between two token sequences.
    
    ROUGE-L is based on the Longest Common Subsequence (LCS).
    
    Args:
        reference: Reference token sequence
        candidate: Candidate token sequence
        
    Returns:
        ROUGE-L F1 score
    """
    if not reference or not candidate:
        return 0.0
    
    # Convert to numpy arrays for efficient computation
    ref_array = np.array(reference, dtype=np.int32)
    cand_array = np.array(candidate, dtype=np.int32)
    
    # Compute LCS using dynamic programming
    dp_matrix = _compute_lcs_dp_matrix(ref_array, cand_array)
    lcs_length = dp_matrix[-1, -1]
    
    # Calculate precision and recall
    precision = lcs_length / len(candidate) if len(candidate) > 0 else 0
    recall = lcs_length / len(reference) if len(reference) > 0 else 0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    
    f1_score = 2 * precision * recall / (precision + recall)
    return float(f1_score)


def calculate_lcs_norm(
    reference: List[int],
    candidate: List[int]
) -> float:
    """
    Calculate normalized Longest Common Substring length.
    
    This finds the longest exact match substring and normalizes by reference length.
    
    Args:
        reference: Reference token sequence
        candidate: Candidate token sequence
        
    Returns:
        Normalized LCS value between 0 and 1
    """
    if not reference or not candidate:
        return 0.0
    
    # Find longest common substring
    lcs_length = _find_longest_common_substring(reference, candidate)
    
    # Normalize by reference length
    return lcs_length / len(reference)


@jit(nopython=True)
def _compute_lcs_dp_matrix(
    seq1: np.ndarray,
    seq2: np.ndarray
) -> np.ndarray:
    """
    Compute dynamic programming matrix for Longest Common Subsequence.
    
    Args:
        seq1: First sequence as numpy array
        seq2: Second sequence as numpy array
        
    Returns:
        DP matrix where dp[i][j] = LCS length of seq1[:i] and seq2[:j]
    """
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    
    return dp


@jit(nopython=True)
def _backtrack_lcs(
    ref: np.ndarray,
    pred: np.ndarray, 
    dp: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backtrack through DP matrix to find contributing tokens.
    
    Args:
        ref: Reference sequence
        pred: Predicted sequence
        dp: DP matrix from _compute_lcs_dp_matrix
        
    Returns:
        Tuple of (lcs_length, contributing_tokens, contributing_pos1, contributing_pos2)
    """
    m, n = len(ref), len(pred)
    contributing_tokens = np.zeros(dp[m][n], dtype=np.int32)
    contributing_pos1 = np.zeros(dp[m][n], dtype=np.int32) 
    contributing_pos2 = np.zeros(dp[m][n], dtype=np.int32)

    i, j = m, n
    k = dp[m][n] - 1
    
    while i > 0 and j > 0:
        if ref[i-1] == pred[j-1]:
            contributing_tokens[k] = ref[i-1]
            contributing_pos1[k] = i-1
            contributing_pos2[k] = j-1
            i -= 1
            j -= 1
            k -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], contributing_tokens, contributing_pos1, contributing_pos2


def find_contributing_tokens(
    ref: np.ndarray,
    pred: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find tokens contributing to ROUGE-L score.
    
    Args:
        ref: Reference token sequence as numpy array
        pred: Predicted token sequence as numpy array
        
    Returns:
        Tuple of (lcs_length, contributing_tokens, ref_positions, pred_positions)
    """
    dp = _compute_lcs_dp_matrix(ref, pred)
    return _backtrack_lcs(ref, pred, dp)


@jit(nopython=True)
def _find_longest_common_substring_jit(
    seq1: np.ndarray,
    seq2: np.ndarray
) -> int:
    """
    JIT-compiled version of longest common substring finder.
    
    Args:
        seq1: First sequence as numpy array
        seq2: Second sequence as numpy array
        
    Returns:
        Length of longest common substring
    """
    if len(seq1) == 0 or len(seq2) == 0:
        return 0
    
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
                if dp[i, j] > max_length:
                    max_length = dp[i, j]
            else:
                dp[i, j] = 0
    
    return max_length


def _find_longest_common_substring(
    seq1: List[int],
    seq2: List[int]
) -> int:
    """
    Find the length of the longest common substring.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Length of longest common substring
    """
    if not seq1 or not seq2:
        return 0
    
    # Convert to numpy arrays and use JIT-compiled version
    seq1_array = np.array(seq1, dtype=np.int32)
    seq2_array = np.array(seq2, dtype=np.int32)
    
    return int(_find_longest_common_substring_jit(seq1_array, seq2_array))


def batch_calculate_metrics(
    true_suffixes: List[List[int]],
    generated_suffixes: List[List[int]]
) -> List[dict]:
    """
    Calculate metrics for multiple sequence pairs.
    
    Args:
        true_suffixes: List of reference sequences
        generated_suffixes: List of generated sequences
        
    Returns:
        List of metric dictionaries
    """
    results = []
    
    for true_seq, gen_seq in zip(true_suffixes, generated_suffixes):
        metrics = {
            "TTR_ref": calculate_ttr(true_seq),
            "TTR_gen": calculate_ttr(gen_seq),
            "Rouge-L": calculate_rouge_l(true_seq, gen_seq),
            "lcs_norm": calculate_lcs_norm(true_seq, gen_seq)
        }
        results.append(metrics)
    
    return results