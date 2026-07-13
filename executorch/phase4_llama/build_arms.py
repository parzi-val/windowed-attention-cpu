"""Phase 4.3: build deployable head-level arms (map25/50/75) from the real Llama-3.2-1B
group-level compressibility map. Pure Python/JSON, no model or device needed.

Same convention as xval_suite.py (semantic-gravity): group_mask_from_scores ranks KV groups by
cost (cheapest first) and windows the bottom `frac` fraction; group_to_head expands each group
decision to all n_rep query heads sharing that group's KV cache, since that's what the kernel's
head_windowed array (one bool per query head) actually needs, and what a real deployment must
do anyway (a group's cache only shrinks if every head sharing it is windowed together).

    .venv-executorch/Scripts/python.exe build_arms.py
"""
import json

MAP_PATH = "head_compressibility_group_Llama-3.2-1B.json"
OUT_PATH = "llama_3.2_1b_arms.json"
FRACS = [0.25, 0.50, 0.75]


def group_mask_from_scores(gscore, n_layers, n_kv_heads, frac):
    """gscore: flat list[L*NKV] of per-group cost. Returns [L][NKV] bool, True = windowed."""
    k = round(frac * n_layers * n_kv_heads)
    order = sorted(range(len(gscore)), key=lambda i: gscore[i])
    windowed = set(order[:k])
    return [[(l * n_kv_heads + g) in windowed for g in range(n_kv_heads)] for l in range(n_layers)]


def group_to_head(gmask, n_rep):
    """gmask: [L][NKV] bool -> [L][NKV*n_rep] bool, each group's decision repeated n_rep times."""
    return [[gmask[l][g] for g in range(len(gmask[l])) for _ in range(n_rep)] for l in range(len(gmask))]


mp = json.load(open(MAP_PATH))
n_layers = mp["n_layers"]
n_kv_heads = mp["n_kv_heads"]
n_rep = mp["n_rep"]
n_heads = n_layers and len(mp["dropped_mass_group"][0]) * n_rep  # sanity cross-check below
delta_ppl_group = mp["delta_ppl_group"]  # [L][NKV]
gscore_flat = [delta_ppl_group[l][g] for l in range(n_layers) for g in range(n_kv_heads)]

print(f"Model: {mp['model']}  L={n_layers}  NKV={n_kv_heads}  n_rep={n_rep}  base_ppl={mp['base_ppl']:.4f}")
print(f"Map's own free_pct_group (dppl<0.05): {mp['free_pct_group']:.1f}%\n")

arms = {}
for frac in FRACS:
    gmask = group_mask_from_scores(gscore_flat, n_layers, n_kv_heads, frac)
    hmask = group_to_head(gmask, n_rep)

    n_groups_total = n_layers * n_kv_heads
    n_groups_windowed = sum(sum(1 for g in row if g) for row in gmask)
    n_heads_total = n_layers * n_kv_heads * n_rep
    n_heads_windowed = sum(sum(1 for h in row if h) for row in hmask)

    # Naive additive sum of the selected groups' own Δppl -- NOT a joint-cost prediction (this
    # project's own earlier finding: marginal costs do not simply add), just a sanity-check
    # magnitude before the real eager-mode joint measurement in the next step.
    selected_cost_sum = sum(
        delta_ppl_group[l][g] for l in range(n_layers) for g in range(n_kv_heads) if gmask[l][g]
    )

    key = f"map{int(frac*100)}"
    arms[key] = {"frac": frac, "group_mask": gmask, "head_windowed": hmask}
    print(f"{key}: {n_groups_windowed}/{n_groups_total} groups windowed "
          f"({n_heads_windowed}/{n_heads_total} heads), naive summed dppl = {selected_cost_sum:.3f} "
          f"(sanity magnitude only, not a joint-cost prediction)")

json.dump({
    "model": mp["model"], "n_layers": n_layers, "n_kv_heads": n_kv_heads, "n_rep": n_rep,
    "base_ppl": mp["base_ppl"], "sink": mp["sink"], "window": mp["window"],
    "source_map": MAP_PATH, "arms": arms,
}, open(OUT_PATH, "w"), indent=1)
print(f"\nsaved {OUT_PATH}")
