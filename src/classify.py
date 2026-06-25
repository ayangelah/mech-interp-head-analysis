"""
Head classification: map each (layer, head) pair to a functional label.

A head can belong to zero or more categories depending on how its scores compare
to empirically chosen thresholds.  The goal is to produce a clean taxonomy across
all 144 heads in GPT-2 small (12 layers × 12 heads).
"""

from __future__ import annotations

import numpy as np
import torch
from transformer_lens import HookedTransformer, ActivationCache
from src.scoring import induction_score, prev_token_score, copy_score

HeadLabel = str  # one of: "induction", "prev_token", "copy", "copy_suppression", "other"


def classify_head(
    cache: ActivationCache,
    layer: int,
    head: int,
    copy_raw: float,
    thresholds: dict[str, float] | None = None,
    copy_std: float = 1.0,
) -> list[HeadLabel]:
    """Classify a single attention head into one or more functional categories.

    Compares this head's scores against ``thresholds`` to assign labels. The copy
    score is supplied pre-computed (``copy_raw``) so it isn't evaluated twice; the
    cheap attention-pattern scores are computed here from ``cache``.

    Args:
        cache:      ActivationCache from a representative forward pass.
        layer:      Layer index (0-based).
        head:       Head index within that layer.
        copy_raw:   This head's RAW (un-scaled) copy score, computed once by the
                    caller. Divided by ``copy_std`` here for scale-only normalisation.
        thresholds: Optional dict mapping category name → minimum score to assign
                    that label.  Defaults to hand-tuned values if ``None``.
        copy_std:   Std of the copy scores across the whole head grid, used to
                    scale ``copy_raw`` into std-units. Defaults to 1.0 (no scaling).

    Returns:
        List of label strings (may be empty if the head doesn't meet any threshold,
        in which case it is implicitly "other").
    """
    copy_result = copy_raw / copy_std
    induction_result = induction_score(cache, layer, head)
    prev_token_result = prev_token_score(cache, layer, head)

    total_categories = []

    if copy_result >= 0 and copy_result >= thresholds['copy']:
        total_categories.append('copy')
    if copy_result < 0 and copy_result <= thresholds['copy_suppression']:
        total_categories.append('copy_suppression')
    if induction_result >= thresholds['induction']:
        total_categories.append('induction')
    if prev_token_result >= thresholds['prev_token']:
        total_categories.append('prev_token')
    return total_categories



def classify_all_heads(
    model: HookedTransformer,
    cache: ActivationCache,
    copy_cache: ActivationCache,
    copy_tokens: torch.Tensor,
    thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """Classify every attention head in the model.

    Iterates over all ``(layer, head)`` pairs and calls ``classify_head``. The two
    pattern-based scores (induction, previous-token) read ``cache`` — best measured
    on a repeated-token sequence — while the copy score reads ``copy_cache`` — best
    measured on natural text, since "copying a random token" is ill-posed.

    Args:
        model:       Loaded HookedTransformer.
        cache:       Pattern ActivationCache (e.g. from a repeated-token sequence),
                     used for the induction and previous-token scores.
        copy_cache:  ActivationCache from a natural-text forward pass, used for the
                     copy score.
        copy_tokens: Token IDs that produced ``copy_cache``, shape ``[batch, seq_len]``.
        thresholds:  Passed through to ``classify_head``.

    Returns:
        A 2-D object array of shape ``[n_layers, n_heads]`` where each cell is a
        list of label strings (the output of ``classify_head``).
    """
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    result = np.empty((num_layers, num_heads), dtype=object)
    grid = np.empty((num_layers, num_heads))

    # --- PASS 1: compute copy_std across the whole grid (copy on natural text) ---
    # Gather every head's RAW copy score into a [num_layers, num_heads] array,
    # then take its .std() -> a single float `copy_std`.
    for layer in range(num_layers):
        for head in range(num_heads):
            grid[layer][head] = copy_score(model, copy_cache, copy_tokens, layer, head)
    copy_std = grid.std()

    # --- PASS 2: classify each head, reusing the copy scores from pass 1 ---
    # Pattern scores (induction / prev-token) read the pattern `cache`.
    for layer in range(num_layers):
        for head in range(num_heads):
            result[layer][head] = classify_head(
                cache, layer, head, grid[layer, head], thresholds, copy_std
            )

    return result

