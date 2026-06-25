"""
Activation patching utilities for causally verifying attention head hypotheses.

Activation patching answers the question: "if I replace head H's output with its
output from a *different* (counterfactual) run, does the model's behaviour change
in the way my hypothesis predicts?"  A significant change confirms that the head
is causally responsible for the behaviour on the metric you measure.
"""

from __future__ import annotations

from typing import Callable

import torch
from transformer_lens import ActivationCache, HookedTransformer


def patch_head_output(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer: int,
    head: int,
    metric: Callable[[torch.Tensor], float],
) -> float:
    """Run an activation-patching experiment on a single head.

    Performs the following causal intervention:
    1. Run the model on ``clean_tokens`` and cache all activations.
    2. Run the model on ``corrupted_tokens`` and cache all activations.
    3. Re-run the model on ``corrupted_tokens`` but *replace* the output of
       ``(layer, head)`` with the corresponding clean activation.
    4. Evaluate ``metric`` on the patched logits and return the result.

    Args:
        model:             Loaded HookedTransformer.
        clean_tokens:      Token IDs for the "correct" input, shape ``[batch, seq]``.
        corrupted_tokens:  Token IDs for the corrupted baseline, shape ``[batch, seq]``.
        layer:             Layer of the head to patch.
        head:              Head index within that layer.
        metric:            A callable that takes model logits ``[batch, seq, vocab]``
                           and returns a scalar measuring task performance.

    Returns:
        The metric value after patching — compare against the clean and corrupted
        baselines to judge causal importance.

    Raises:
        NotImplementedError: Until you implement this function.
    """
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_z = clean_cache["z", layer]

    def patch_hook(activation, hook):
        activation[:, :, head, :] = clean_z[:, :, head, :]
        return activation
    

    patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_hook)])

    return metric(patched_logits)


def verify_head_via_patching(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer: int,
    head: int,
    metric: Callable[[torch.Tensor], float],
) -> dict[str, float]:
    """Return a structured patching result for one head.

    Runs three forward passes:
    - **clean baseline**: metric on clean input.
    - **corrupted baseline**: metric on corrupted input (lower bound).
    - **patched**: metric after patching ``(layer, head)`` from clean into corrupted.

    Then computes the *normalised patching effect*:
        ``(patched − corrupted) / (clean − corrupted)``
    A value near 1 means the head fully recovers the clean behaviour; near 0 means
    it has no effect.

    Args:
        model:             Loaded HookedTransformer.
        clean_tokens:      Clean input token IDs.
        corrupted_tokens:  Corrupted input token IDs.
        layer:             Layer of the head to verify.
        head:              Head index within that layer.
        metric:            Scalar metric callable (higher = better task performance).

    Returns:
        Dict with keys ``"clean"``, ``"corrupted"``, ``"patched"``,
        ``"normalised_effect"``.

    Raises:
        NotImplementedError: Until you implement this function.
    """
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean = metric(clean_logits)
    corrupted = metric(corrupted_logits)
    patched = patch_head_output(model, clean_tokens, corrupted_tokens, layer, head, metric)

    normalised_effect = (patched - corrupted) / (clean - corrupted)

    return {
        "clean": clean,
        "corrupted": corrupted,
        "patched": patched,
        "normalised_effect": normalised_effect,
    }