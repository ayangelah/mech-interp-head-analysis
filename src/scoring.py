"""
Scoring functions that quantify the functional behaviour of individual attention heads.

Each function takes a TransformerLens ActivationCache (and optionally the model) and
returns a float (or array of floats) measuring how strongly a given head exhibits a
named behaviour.  Higher score ↔ stronger expression of the behaviour.
"""

from __future__ import annotations

import numpy as np
import torch
from transformer_lens import ActivationCache, HookedTransformer


def induction_score(
    cache: ActivationCache,
    layer: int,
    head: int,
) -> float:
    """Compute the induction score for a single attention head.

    Induction heads implement the pattern [A][B] ... [A] → [B].  They do this by
    attending from each token back to the token that *follows* the previous occurrence
    of the current token — i.e. the attention pattern should be shifted one position
    above the main diagonal (the "induction stripe").

    Args:
        cache: ActivationCache from a forward pass on a repeated-token sequence.
               Key ``cache['pattern', layer]`` has shape
               ``[batch, heads, seq_len, seq_len]``.
        layer: Transformer layer index (0-based).
        head:  Head index within that layer.

    Returns:
        A scalar in [0, 1] representing mean attention weight on the induction
        diagonal (one above the main diagonal in the lower-left half of the matrix).

    Raises:
        NotImplementedError: Until you implement this function.
    """
    pattern = cache['pattern', layer]
    seq_len = pattern.shape[-1]
    L = seq_len // 2
    weights = []
    for p in range(L+1, seq_len):
        w = pattern[0, head, p, p - L + 1] # value that followd it.
        weights.append(w)
    return (sum(weights)/len(weights)).item()


def prev_token_score(
    cache: ActivationCache,
    layer: int,
    head: int,
) -> float:
    """Compute the previous-token score for a single attention head.

    Previous-token heads attend almost entirely to the immediately preceding token,
    i.e. the attention pattern concentrates on the subdiagonal (offset -1).

    Args:
        cache: ActivationCache from any forward pass.
               Key ``cache['pattern', layer]`` has shape
               ``[batch, heads, seq_len, seq_len]``.
        layer: Transformer layer index (0-based).
        head:  Head index within that layer.

    Returns:
        A scalar in [0, 1] representing mean attention weight on the subdiagonal.

    Raises:
        NotImplementedError: Until you implement this function.
    """
    pattern = cache['pattern', layer]
    seq_len = pattern.shape[-1] # src axis
    weights = []
    for p in range(1, seq_len):
        w = pattern[0, head, p, p - 1]
        weights.append(w)
    return (sum(weights)/len(weights)).item()


def copy_score(
    model: HookedTransformer,
    cache: ActivationCache,
    tokens: torch.Tensor,
    layer: int,
    head: int,
) -> float:
    """Compute the copy score for a single attention head.

    A copy head increases the logit of the *attended-to* token at the output position.
    For each (destination, source) pair, the head's output is projected through the
    unembedding and we read off the logit assigned to the source token's identity,
    weighted by the attention paid to that source.

    A strongly *negative* score indicates the opposite behaviour, copy suppression:
    the head writes against the attended token's unembedding direction, pushing its
    logit down (see McDougall et al. 2023, https://arxiv.org/abs/2310.04625).

    Args:
        model:  The loaded HookedTransformer (needed for W_O and W_U).
        cache:  ActivationCache from a forward pass.
                Key ``cache['z', layer]`` has shape
                ``[batch, seq_len, heads, d_head]``.
                Key ``cache['pattern', layer]`` has shape
                ``[batch, heads, seq_len, seq_len]``.
        tokens: Token IDs that produced ``cache``, shape ``[batch, seq_len]``.
                Needed to map a source *position* to its token *id*.
        layer:  Transformer layer index (0-based).
        head:   Head index within that layer.

    Returns:
        A scalar: mean attention-weighted logit boost of attended tokens.
        Positive ↔ copying; negative ↔ copy suppression.
    """

    pattern = cache['pattern', layer]   # grab pattern for the weights, has shape [batch, heads, dest, src]
    z = cache['z', layer]               # z has shape [batch, seq, heads, d_head]
    seq_len = z.shape[1]                # grab the seq from dimension here
    total = 0
    for dest in range(seq_len):
        out_d = z[0, dest, head, :] @ model.W_O[layer, head]     # out_d has shape [d_model]
        logits = out_d @ model.W_U                               # logits shape [vocab]
        for src in range(seq_len):
            weight = pattern[0, head, dest, src] # find weight
            boost = logits[tokens[0, src]]  # find the boost
            total += weight * boost # add weighted boost to total
    return total / seq_len

