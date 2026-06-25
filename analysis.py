# %% [markdown]
# # Attention Head Analysis — GPT-2 Small
#
# Driver notebook (Jupytext `# %%` format).
# Open in VS Code with the Jupyter extension, or convert with:
#   `jupytext --to notebook analysis.py`

# %% [markdown]
# ## 0. Setup

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import transformer_lens
from transformer_lens import HookedTransformer, utils

from src.scoring import induction_score, prev_token_score, copy_score
from src.patching import patch_head_output, verify_head_via_patching
from src.classify import classify_head, classify_all_heads
from src.plotting import plot_score_heatmap, plot_head_heatmap, plot_attention_pattern

# %%
# Load GPT-2 small. Downloads weights on first run (~500 MB).
model = HookedTransformer.from_pretrained("gpt2")
model.eval()

N_LAYERS = model.cfg.n_layers   # 12
N_HEADS  = model.cfg.n_heads    # 12
D_MODEL  = model.cfg.d_model    # 768
print(f"GPT-2 small — layers: {N_LAYERS}, heads: {N_HEADS}, d_model: {D_MODEL}")

# %% [markdown]
# ### Sanity forward pass

# %%
# Tokenise a natural-language passage and run a forward pass. This cache/tokens
# pair also feeds the OV/copy scores, which need realistic text (copying a random
# token's identity is ill-posed), so a longer real passage gives stabler copy stats.
prompt = (
    "Natural language models learn to predict the next token from context. "
    "When a rare name or unusual word appears once, the model often repeats it "
    "later in the same passage. Researchers study individual attention heads to "
    "understand how this copying and prediction actually work inside the network."
)
tokens = model.to_tokens(prompt)
print("Token IDs:", tokens)
print("Tokens:   ", model.to_str_tokens(prompt))

logits, cache = model.run_with_cache(tokens)
print("Logits shape:", logits.shape)   # [1, seq_len, vocab_size]

# Quick sanity: top predicted next token after the last position
last_logits = logits[0, -1, :]
top_token_id = last_logits.argmax().item()
print("Top next-token prediction:", repr(model.to_string(top_token_id)))

# %% [markdown]
# ### Repeated-token sequence (for induction / previous-token scoring)

# %%
# Build a sequence of the form [rand tokens] [same rand tokens] — induction heads
# should show strong diagonal stripes in the second half.
torch.manual_seed(42)
seq_len    = 25
rand_half  = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
rep_tokens = torch.cat([rand_half, rand_half], dim=1)  # shape [1, 2*seq_len]

_, rep_cache = model.run_with_cache(rep_tokens)
print("Repeated-token sequence shape:", rep_tokens.shape)

# %% [markdown]
# ---
# ## 1. Induction Score
#

# %%
induction_scores = np.zeros((N_LAYERS, N_HEADS))

for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        induction_scores[layer, head] = induction_score(rep_cache, layer, head)

fig = plot_score_heatmap(
    induction_scores,
    title="Induction Score (GPT-2 Small)",
    cmap="Blues",
    vmin=0, vmax=1,
)
fig.savefig("images/induction_score.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 2. Previous-Token Score
#

# %%
prev_token_scores = np.zeros((N_LAYERS, N_HEADS))

for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        prev_token_scores[layer, head] = prev_token_score(rep_cache, layer, head)

fig = plot_score_heatmap(
    prev_token_scores,
    title="Previous-Token Score (GPT-2 Small)",
    cmap="Greens",
    vmin=0, vmax=1,
)
fig.savefig("images/prev_token_score.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 3. Copy Score
#

# %%
copy_scores = np.zeros((N_LAYERS, N_HEADS))

for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        copy_scores[layer, head] = copy_score(model, cache, tokens, layer, head)

fig = plot_score_heatmap(
    copy_scores,
    title="Copy Score (GPT-2 Small)",
    cmap="RdBu_r",
)
fig.savefig("images/copy_score.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 4. Activation Patching Verification
#
# Pick a head you believe is an induction head from section 1 and verify it causally.

# %%
# Define a simple induction metric: mean logit on the correct next token.
def induction_metric(logits: torch.Tensor) -> float:
    start = seq_len
    end = 2 * seq_len - 1
    total = 0
    for p in range(start, end):
        correct_next_token = clean_tokens[0, p + 1]
        total += logits[0, p, correct_next_token]
    n = end - start
    return (total/n).item()


# Build clean / corrupted token pairs for the patching experiment.
torch.manual_seed(0)
clean_tokens      = rep_tokens                           # repeated sequence
corrupted_tokens  = torch.cat([rand_half,               # second half is *different*
                                torch.randint(0, model.cfg.d_vocab, (1, seq_len))],
                               dim=1)

# Pick a head from your Section 1 induction heatmap (e.g. L5H1 / L5H5),
# and compare against a non-induction control head (e.g. L1H4).
TARGET_LAYER, TARGET_HEAD = 5, 1
result = verify_head_via_patching(
    model, clean_tokens, corrupted_tokens,
    layer=TARGET_LAYER, head=TARGET_HEAD,
    metric=induction_metric,
)
print(result)

# %% [markdown]
# ---
# ## 5. Head Classification

# %%
# Per-category thresholds. NOTE the unit difference:
#   - copy / copy_suppression: std-units (copy score is scaled by its grid std)
#   - induction / prev_token:  raw attention weight in [0, 1]
# These are starting points — tune them against your heatmaps from sections 1–3.
thresholds = {
    "copy": 1.5,              # ≥ 1 std of token-copying
    "copy_suppression": -2.0, # ≤ 1 std of suppression (negative)
    "induction": 0.4,         # tune from the induction heatmap
    "prev_token": 0.4,        # tune from the prev-token heatmap
}

# Pattern scores read rep_cache (repeated tokens); copy reads the natural-text
# fox-sentence cache/tokens from the sanity pass.
all_labels = classify_all_heads(model, rep_cache, cache, tokens, thresholds)

fig = plot_head_heatmap(
    all_labels,
    title="Attention Head Classification — GPT-2 Small",
)
fig.savefig("images/classification.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
