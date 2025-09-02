#!/usr/bin/env python3
"""
Noise exclusion functions for Gutenberg memorization analysis.
"""

# Known contaminated sequences - canonical texts likely in web training data
KNOWN_CONTAMINATIONS = {
    # Gutenberg V1 (Stage 1)
    ('Stage_1', '0a', 354),    # Keats's poems
    ('Stage_1', '0c', 244),    # Anglican 39 Articles
    ('Stage_1', '0d', 161),    # Robert Burns's poem
    ('Stage_1', 2, 62),        # Thomas Gray's "Elegy"
    ('Stage_1', 3, 43),        # US Constitution
    ('Stage_1', 8, 87),        # Tolstoy story
    ('Stage_1', 8, 416),       # Bible - Zechariah
    ('Stage_1', 16, 259),      # Milton's "Comus"
    ('Stage_1', 32, 68),       # Lincoln's First Inaugural
    ('Stage_1', 32, 426),      # Shakespeare's "The Tempest"
    ('Stage_1', 48, 304),      # Shakespeare's "Much Ado"
    ('Stage_1', 64, 253),      # Bible - Daniel
    ('Stage_1', 96, 137),      # Longfellow's "Evangeline"
    ('Stage_1', 128, 26),      # Andersen's "The Fir Tree"
    ('Stage_1', 128, 55),      # Kipling's "The Jungle Book"
    ('Stage_1', 128, 170),     # Bible - Daniel
    
    # Gutenberg V2 (Stage 2)
    ('Stage_2', '0a', 58),     # Emerson's "Each and All"
    ('Stage_2', '0a', 66),     # Andersen's "Thumbelina"
    ('Stage_2', '0c', 137),    # Grimm's Fairy Tale
    ('Stage_2', '0d', 157),    # US Constitution
    ('Stage_2', 96, 4),        # Longfellow's "Wreck of the Hesperus"
}

def is_contaminated(stage: str, rep, idx: int) -> bool:
    """Check if a sequence is a known contamination."""
    rep_key = int(rep) if isinstance(rep, str) and rep.isdigit() else rep
    return (stage, rep_key, idx) in KNOWN_CONTAMINATIONS

def is_low_ttr(item: dict, ttr_threshold: float = 0.4) -> bool:
    """Check if a sequence has low Type-Token Ratio."""
    return 'TTR_ref' in item and item['TTR_ref'] < ttr_threshold