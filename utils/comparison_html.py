"""
HTML comparison visualization for prefix-suffix memorization analysis.

Creates side-by-side visual comparisons with:
- Prefix with green background on both sides
- Token-level diff between ground truth and generated text
"""

from datetime import datetime
from typing import List, Optional, Tuple
import os
import sys
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path

# Fix imports for metrics module
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.metrics import find_contributing_tokens, calculate_ttr


def create_sequencematcher_viz(
    true_tokens: List[int],
    generated_tokens: List[int],
    tokenizer,
    show_token_ids: bool = False
) -> Tuple[str, str]:
    """
    Create HTML visualization based on SequenceMatcher diff.
    
    Args:
        true_tokens: Ground truth token IDs
        generated_tokens: Generated token IDs
        tokenizer: Tokenizer for decoding
        show_token_ids: Whether to show token IDs as superscripts
        
    Returns:
        Tuple of (true_html, generated_html)
    """
    matcher = SequenceMatcher(None, true_tokens, generated_tokens)
    true_html = []
    gen_html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Decode and escape the matching segments
            true_span_tokens = true_tokens[i1:i2]
            gen_span_tokens = generated_tokens[j1:j2]
            
            for token_id in true_span_tokens:
                token_text = tokenizer.decode([token_id])
                escaped_text = escape_html(token_text)
                if show_token_ids:
                    token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
                else:
                    token_html = escaped_text
                true_html.append(f'<span class="seq-equal">{token_html}</span>')
                
            for token_id in gen_span_tokens:
                token_text = tokenizer.decode([token_id])
                escaped_text = escape_html(token_text)
                if show_token_ids:
                    token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
                else:
                    token_html = escaped_text
                gen_html.append(f'<span class="seq-equal">{token_html}</span>')
                
        elif tag == 'delete':
            # Only in true sequence
            true_span_tokens = true_tokens[i1:i2]
            for token_id in true_span_tokens:
                token_text = tokenizer.decode([token_id])
                escaped_text = escape_html(token_text)
                if show_token_ids:
                    token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
                else:
                    token_html = escaped_text
                true_html.append(f'<span class="seq-delete">{token_html}</span>')
                
        elif tag == 'insert':
            # Only in generated sequence
            gen_span_tokens = generated_tokens[j1:j2]
            for token_id in gen_span_tokens:
                token_text = tokenizer.decode([token_id])
                escaped_text = escape_html(token_text)
                if show_token_ids:
                    token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
                else:
                    token_html = escaped_text
                gen_html.append(f'<span class="seq-insert">{token_html}</span>')
                
        elif tag == 'replace':
            # Different in both sequences
            true_span_tokens = true_tokens[i1:i2]
            gen_span_tokens = generated_tokens[j1:j2]
            
            for token_id in true_span_tokens:
                token_text = tokenizer.decode([token_id])
                escaped_text = escape_html(token_text)
                if show_token_ids:
                    token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
                else:
                    token_html = escaped_text
                true_html.append(f'<span class="seq-delete">{token_html}</span>')
                
            for token_id in gen_span_tokens:
                token_text = tokenizer.decode([token_id])
                escaped_text = escape_html(token_text)
                if show_token_ids:
                    token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
                else:
                    token_html = escaped_text
                gen_html.append(f'<span class="seq-insert">{token_html}</span>')
    
    return ''.join(true_html), ''.join(gen_html)


def create_rouge_viz_with_ids(
    true_tokens: List[int],
    generated_tokens: List[int],
    contributing_pos1: np.ndarray,
    contributing_pos2: np.ndarray,
    tokenizer,
    show_token_ids: bool = False
) -> Tuple[str, str]:
    """
    Create HTML visualization based on ROUGE-L contributing tokens.
    
    Args:
        true_tokens: Ground truth token IDs
        generated_tokens: Generated token IDs  
        contributing_pos1: Positions in true_tokens that contribute to ROUGE-L
        contributing_pos2: Positions in generated_tokens that contribute to ROUGE-L
        tokenizer: Tokenizer for decoding
        show_token_ids: Whether to show token IDs as superscripts
        
    Returns:
        Tuple of (true_html, generated_html)
    """
    true_html = []
    gen_html = []
    
    # Convert contributing positions to set for O(1) lookup
    contrib_set1 = set(contributing_pos1)
    contrib_set2 = set(contributing_pos2)
    
    # Process true tokens
    for i, token_id in enumerate(true_tokens):
        token_text = tokenizer.decode([token_id])
        escaped_text = escape_html(token_text)
        
        if show_token_ids:
            token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
        else:
            token_html = escaped_text
            
        if i in contrib_set1:
            true_html.append(f'<span class="contributing">{token_html}</span>')
        else:
            true_html.append(f'<span class="non-contributing">{token_html}</span>')
    
    # Process generated tokens
    for i, token_id in enumerate(generated_tokens):
        token_text = tokenizer.decode([token_id])
        escaped_text = escape_html(token_text)
        
        if show_token_ids:
            token_html = f'{escaped_text}<sup class="token-id">{token_id}</sup>'
        else:
            token_html = escaped_text
            
        if i in contrib_set2:
            gen_html.append(f'<span class="contributing">{token_html}</span>')
        else:
            gen_html.append(f'<span class="non-contributing">{token_html}</span>')
    
    return ''.join(true_html), ''.join(gen_html)


def create_comparison_html(
    prefix_tokens: List[int],
    true_suffix_tokens: List[int],
    generated_suffix_tokens: List[int],
    tokenizer,
    repetition: int,
    strategy: str,
    sample_idx: int = 0,
    rouge_score: float = 0.0,
    lcs_score: float = 0.0,
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
    show_token_ids: bool = False
) -> str:
    """
    Create HTML comparison visualization of ground truth vs generated text.
    
    Args:
        prefix_tokens: Token IDs for the prefix (input to model)
        true_suffix_tokens: Token IDs for ground truth continuation
        generated_suffix_tokens: Token IDs for model-generated continuation
        tokenizer: Tokenizer for decoding tokens
        repetition: Repetition count (e.g., 1, 8, 32, 128)
        strategy: Generation strategy (e.g., 'greedy', 'nucleus')
        sample_idx: Sample index for display
        rouge_score: ROUGE-L score for this sample
        lcs_score: LCS score for this sample
        model_name: Optional model identifier for display
        output_path: Optional path to save HTML file
    
    Returns:
        HTML string
    """
    
    
    # Decode prefix with token IDs if requested
    if show_token_ids:
        prefix_html = tokens_to_html_with_ids(prefix_tokens, tokenizer, "prefix")
    else:
        prefix_text = tokenizer.decode(prefix_tokens)
        # Remove trailing newline from prefix if it exists to avoid line break before suffix
        prefix_escaped = escape_html(prefix_text)
        if prefix_escaped.endswith('<span class="newline-marker">\\n</span><br>'):
            prefix_escaped = prefix_escaped[:-len('<br>')]
        prefix_html = f'<span class="prefix">{prefix_escaped}</span>'
    
    # Find contributing tokens for ROUGE-L
    _, contributing_tokens, contrib_pos1, contrib_pos2 = find_contributing_tokens(
        np.array(true_suffix_tokens),
        np.array(generated_suffix_tokens)
    )
    
    # Create ROUGE-L based visualization for suffixes
    true_suffix_rouge_html, gen_suffix_rouge_html = create_rouge_viz_with_ids(
        true_suffix_tokens,
        generated_suffix_tokens,
        contrib_pos1,
        contrib_pos2,
        tokenizer,
        show_token_ids
    )
    
    # Create SequenceMatcher based visualization for suffixes
    true_suffix_seq_html, gen_suffix_seq_html = create_sequencematcher_viz(
        true_suffix_tokens,
        generated_suffix_tokens,
        tokenizer,
        show_token_ids
    )
    
    # Calculate number of contributing tokens
    num_contributing = len(contributing_tokens)
    
    # Calculate TTR (Type-Token Ratio) for both sequences
    ttr_ref = calculate_ttr(true_suffix_tokens)
    ttr_gen = calculate_ttr(generated_suffix_tokens)
    
    css = """
.container {
    display: grid;
    grid-template-columns: 50px 1fr 50px 1fr;
    font-family: monospace;
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 20px;
}
.header {
    padding: 8px;
    background: #4a4a4a;
    color: white;
    font-weight: bold;
    text-align: center;
}
.line-numbers {
    padding: 8px;
    text-align: right;
    color: #666;
    background: #f5f5f5;
    border-right: 1px solid #ddd;
    white-space: pre;
    align-self: stretch;
}
.content {
    padding: 8px;
    white-space: pre-wrap;
    word-break: break-word;
    display: block;
}

/* Prefix styling - green background */
.prefix {
    background: #c8e6c9;
    color: #1b5e20;
    font-weight: 600;
}

/* ROUGE-L highlighting */
.contributing {
    background: #fff3cd;
    color: #856404;
    font-weight: 600;
}
.non-contributing {
    color: #999;
    opacity: 0.7;
}

/* SequenceMatcher highlighting */
.seq-equal {
    background: #fff3cd;
    color: #856404;
}
.seq-delete {
    background: #f8d7da;
    color: #721c24;
    text-decoration: line-through;
}
.seq-insert {
    background: #cce5ff;
    color: #004085;
}

/* Token ID styling */
.token-id {
    font-size: 10px;
    color: #666;
    vertical-align: super;
    margin-left: 1px;
}

/* Tab controls */
.tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
}

.tab-button {
    padding: 10px 20px;
    background: #f8f9fa;
    border: 1px solid #ddd;
    border-bottom: none;
    cursor: pointer;
    font-family: Arial, sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #666;
    transition: all 0.3s;
}

.tab-button:hover {
    background: #e9ecef;
}

.tab-button.active {
    background: white;
    color: #333;
    border-top: 3px solid #007bff;
    margin-top: -2px;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.stats {
    background: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metrics {
    display: flex;
    gap: 30px;
    margin-bottom: 10px;
}

.metric {
    display: flex;
    align-items: center;
    gap: 10px;
}

.metric-label {
    font-weight: bold;
    color: #666;
}

.metric-value {
    font-size: 18px;
    color: #333;
}

body {
    font-family: 'Courier New', monospace;
    margin: 20px;
    background: #f5f5f5;
}

h1, h2 {
    font-family: Arial, sans-serif;
}

.legend {
    background: white;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.legend-item {
    display: inline-block;
    margin-right: 20px;
    padding: 4px 8px;
    border-radius: 4px;
}

/* Newline marker styling */
.newline-marker {
    color: #999;
    font-size: 10px;
    font-family: monospace;
    background: rgba(0, 0, 0, 0.05);
    padding: 0 2px;
    border-radius: 2px;
    margin: 0 1px;
}
"""
    
    # Decode text sections for line counting
    prefix_text = tokenizer.decode(prefix_tokens)
    true_suffix_text = tokenizer.decode(true_suffix_tokens)
    gen_suffix_text = tokenizer.decode(generated_suffix_tokens)
    
    # Count actual lines in the combined text
    # We need to count lines in the actual rendered text
    prefix_lines = prefix_text.count('\n') + 1
    true_suffix_lines = true_suffix_text.count('\n') + 1
    gen_suffix_lines = gen_suffix_text.count('\n') + 1
    
    # Calculate total lines for each side
    true_total_lines = prefix_lines + true_suffix_lines
    gen_total_lines = prefix_lines + gen_suffix_lines
    max_lines = max(true_total_lines, gen_total_lines)
    
    # Generate continuous line numbers for the entire text
    line_numbers = '\n'.join(str(i) for i in range(1, max_lines + 1))
    
    # Create title with model name if provided
    title_parts = [f"Sample {sample_idx}", f"Rep {repetition}", strategy.capitalize()]
    if model_name:
        title_parts.append(model_name)
    title = " - ".join(title_parts)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        {css}
    </style>
</head>
<body>
    <h1>Generation Comparison - {title}</h1>
    
    <div class="stats">
        <div class="metrics">
            <div class="metric">
                <span class="metric-label">Repetition:</span>
                <span class="metric-value">{repetition}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Strategy:</span>
                <span class="metric-value">{strategy}</span>
            </div>
            <div class="metric">
                <span class="metric-label">ROUGE-L:</span>
                <span class="metric-value">{rouge_score:.3f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">LCS:</span>
                <span class="metric-value">{lcs_score:.3f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Prefix Length:</span>
                <span class="metric-value">{len(prefix_tokens)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Suffix Length:</span>
                <span class="metric-value">{len(true_suffix_tokens)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Contributing Tokens:</span>
                <span class="metric-value">{num_contributing}</span>
            </div>
            <div class="metric">
                <span class="metric-label">TTR (Ref):</span>
                <span class="metric-value">{ttr_ref:.3f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">TTR (Gen):</span>
                <span class="metric-value">{ttr_gen:.3f}</span>
            </div>
        </div>
    </div>
    
    <h2>Side-by-Side Comparison</h2>
    
    <!-- Tab buttons -->
    <div class="tabs">
        <button class="tab-button active" onclick="showTab('rouge')" id="rouge-tab">ROUGE-L View</button>
        <button class="tab-button" onclick="showTab('sequence')" id="sequence-tab">SequenceMatcher View</button>
    </div>
    
    <!-- ROUGE-L view -->
    <div id="rouge-content" class="tab-content active">
        <div class="container">
        <!-- Headers -->
        <div class="header">Line</div>
        <div class="header">Ground Truth</div>
        <div class="header">Line</div>
        <div class="header">Generated</div>
        
            <!-- Combined content with continuous line numbers -->
            <div class="line-numbers">{line_numbers}</div>
            <div class="content">
                {prefix_html}{true_suffix_rouge_html}
            </div>
            <div class="line-numbers">{line_numbers}</div>
            <div class="content">
                {prefix_html}{gen_suffix_rouge_html}
            </div>
        </div>
    </div>
    
    <!-- SequenceMatcher view -->
    <div id="sequence-content" class="tab-content">
        <div class="container">
            <!-- Headers -->
            <div class="header">Line</div>
            <div class="header">Ground Truth</div>
            <div class="header">Line</div>
            <div class="header">Generated</div>
            
            <!-- Combined content with continuous line numbers -->
            <div class="line-numbers">{line_numbers}</div>
            <div class="content">
                {prefix_html}{true_suffix_seq_html}
            </div>
            <div class="line-numbers">{line_numbers}</div>
            <div class="content">
                {prefix_html}{gen_suffix_seq_html}
            </div>
        </div>
    </div>
    
    <div class="legend">
        <h3>Legend</h3>
        <div id="rouge-legend" style="display: inline;">
            <span class="legend-item prefix">Prefix (Input)</span>
            <span class="legend-item contributing">Contributing to ROUGE-L</span>
            <span class="legend-item non-contributing">Not contributing</span>
        </div>
        <div id="sequence-legend" style="display: none;">
            <span class="legend-item prefix">Prefix (Input)</span>
            <span class="legend-item seq-equal">Matching blocks</span>
            <span class="legend-item seq-delete">Missing in generated</span>
            <span class="legend-item seq-insert">Added in generated</span>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all buttons
            document.querySelectorAll('.tab-button').forEach(button => {{
                button.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabName + '-content').classList.add('active');
            
            // Mark button as active
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Update legend
            if (tabName === 'rouge') {{
                document.getElementById('rouge-legend').style.display = 'inline';
                document.getElementById('sequence-legend').style.display = 'none';
            }} else {{
                document.getElementById('rouge-legend').style.display = 'none';
                document.getElementById('sequence-legend').style.display = 'inline';
            }}
        }}
    </script>
    
    <div style="margin-top: 20px; color: #666; font-size: 12px;">
        Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>
"""
    
    if output_path:
        # If output_path doesn't have a filename, generate one
        if output_path.endswith('/'):
            filename = f"sample_{sample_idx:03d}_rep{repetition}_{strategy}"
            if model_name:
                filename += f"_{model_name}"
            filename += f"_rouge{rouge_score:.2f}.html"
            output_path = output_path + filename
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML comparison saved to: {output_path}")
    
    return html


def escape_html(text: str) -> str:
    """Escape HTML special characters and make newlines visible."""
    # First escape HTML special characters
    text = (text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;'))
    
    # Make newlines visible as \n with a special style, then break line
    text = text.replace('\n', '<span class="newline-marker">\\n</span><br>')
    
    return text


def add_separator_lines() -> str:
    """Add visual separator lines between prefix and suffix."""
    return '<div class="separator-line"></div><div class="separator-line"></div>'


def tokens_to_html_with_ids(tokens: List[int], tokenizer, css_class: str = "") -> str:
    """
    Convert tokens to HTML with token IDs shown as superscripts.
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer for decoding
        css_class: CSS class for the span elements
        
    Returns:
        HTML string with token IDs as superscripts
    """
    html_parts = []
    for token_id in tokens:
        text = tokenizer.decode([token_id])
        escaped_text = escape_html(text)
        # Add token ID as superscript
        html_parts.append(
            f'<span class="{css_class}">'
            f'{escaped_text}'
            f'<sup class="token-id">{token_id}</sup>'
            f'</span>'
        )
    return ''.join(html_parts)


