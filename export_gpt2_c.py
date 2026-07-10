"""Exports GPT-2 small for gpt2_forward_windowed.c: the real HF weights in this repo's own
.bin format (reusing write_model from train_gpt2.py, unmodified), a small forward-only
reference-logits file for the C parity gate (own format -- no gradients needed, forward
pass only, so no backward pass to run), and the per-head lazy map (causal compressibility
scores: how much perplexity each head alone costs to window) as a flat float32 array so the
C side doesn't need a JSON parser.

Run once, from the repo root (llm.c is checked out as a submodule at ./llm.c --
`git submodule update --init` first if you haven't):

    python export_gpt2_c.py
"""
import argparse
import json
import struct
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--llmc-dir", default="llm.c", help="path to the llm.c submodule checkout")
p.add_argument("--out-dir", default=".")
p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog, and then")
p.add_argument("--map-json", default="head_compressibility_gpt2.json",
                help="per-head causal compressibility map (delta_ppl per layer/head, sink+window "
                     "replacement test); see the map's own 'sink'/'window' fields, which must "
                     "match SINK/WINDOW in gpt2_forward_windowed.c")
args = p.parse_args()

sys.path.insert(0, args.llmc_dir)
from train_gpt2 import GPT, GPTConfig, write_model  # noqa: E402

torch.manual_seed(42)

print("Loading GPT-2 small (real HF weights, Conv1D-transposed to Linear convention)...")
model = GPT.from_pretrained("gpt2")
model.eval()

bin_path = f"{args.out_dir}/gpt2_124M.bin"
write_model(model, bin_path, dtype="float32")

from transformers import GPT2Tokenizer  # noqa: E402

tok = GPT2Tokenizer.from_pretrained("gpt2")
x = tok(args.prompt, return_tensors="pt").input_ids  # (1, T)
B, T = x.shape
print(f"Prompt: {args.prompt!r} -> {T} tokens")

with torch.no_grad():
    # targets=x (self-supervised, values unused -- loss is discarded) forces GPT.forward's
    # full-sequence logits path; with targets=None it only computes the LAST position's
    # logits (an inference-time optimization), which would break the per-position parity check.
    logits, _ = model(x, targets=x, return_logits=True)  # (B, T, V) -- unpadded V

V = logits.shape[-1]
C = model.config.n_embd
L = model.config.n_layer
NH = model.config.n_head

state_path = f"{args.out_dir}/gpt2_124M_fwd_state.bin"
with open(state_path, "wb") as f:
    header = [0] * 256
    header[0] = 20260711  # magic (distinct from llm.c's own state-file format)
    header[1] = 1  # version
    header[2] = B
    header[3] = T
    header[4] = V
    header[5] = C
    header[6] = L
    header[7] = NH
    f.write(struct.pack("<256i", *header))
    f.write(x.numpy().astype("int32").tobytes())
    f.write(logits.numpy().astype("float32").tobytes())

print(f"wrote {bin_path}")
print(f"wrote {state_path}")

with open(args.map_json) as f:
    map_data = json.load(f)
assert map_data["model"] == "gpt2", f"map is for {map_data['model']!r}, not gpt2"
delta_ppl = map_data["delta_ppl"]  # (L, NH) nested list
map_sink, map_window = map_data["sink"], map_data["window"]
print(f"Map: sink={map_sink} window={map_window} baseline_ppl={map_data['baseline_ppl']:.3f} "
      f"(must match SINK/WINDOW in gpt2_forward_windowed.c -- currently 4/64)")
assert len(delta_ppl) == L and len(delta_ppl[0]) == NH, \
    f"map shape ({len(delta_ppl)},{len(delta_ppl[0])}) != model ({L},{NH})"

map_path = f"{args.out_dir}/gpt2_124M_lazymap.bin"
with open(map_path, "wb") as f:
    header = [0] * 256
    header[0] = 20260712  # magic
    header[1] = 1  # version
    header[2] = L
    header[3] = NH
    header[4] = map_sink
    header[5] = map_window
    f.write(struct.pack("<256i", *header))
    flat = [v for row in delta_ppl for v in row]  # row-major (L, NH), matches the C side's indexing
    f.write(struct.pack(f"<{len(flat)}f", *flat))
print(f"wrote {map_path}  ({sum(1 for v in flat if v < 0.05)}/{len(flat)} heads lazy at delta_ppl<0.05)")

print("Done. All three files go next to gpt2_forward_windowed.c before compiling.")
