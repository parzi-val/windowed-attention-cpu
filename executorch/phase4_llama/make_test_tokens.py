"""Phase 4.5: build a fixed real token sequence (from real Llama-3.2-1B tokenizer) to drive the
on-device prefill_decode_runner, plus an eager PC-side reference computed the same way, so the
device's output can be checked for numeric parity, not just "did it run."

    .venv-executorch/Scripts/python.exe make_test_tokens.py
"""
import struct
import torch
from transformers import AutoTokenizer

TEXT = ("The quick brown fox jumps over the lazy dog. In machine learning, attention "
        "mechanisms allow models to focus on relevant parts of the input sequence when "
        "producing each output token, which is especially useful for long sequences. "
        "Large language models are typically trained on vast corpora of text scraped from "
        "the internet, books, and other written sources, then fine-tuned on smaller, "
        "curated datasets to align their behavior with what users actually want. Attention "
        "windowing is one strategy for making these models cheaper to run on edge devices "
        "such as smartphones, where memory and compute are far more constrained than on a "
        "datacenter GPU cluster used for training.")
PREFILL_LEN = 64  # must match export_llama.py's --max-prompt-len (default 64) -- prefill's
                   # exported shape is fixed, not dynamic (see export_llama.py's docstring)
N_DECODE = 8

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
ids = tok(TEXT, return_tensors="pt").input_ids[0].tolist()
n_needed = PREFILL_LEN + N_DECODE
assert len(ids) >= n_needed, f"need {n_needed} tokens, text only tokenized to {len(ids)}"
ids = ids[:n_needed]

with open("test_tokens.bin", "wb") as f:
    f.write(struct.pack("<q", len(ids)))
    f.write(struct.pack(f"<{len(ids)}q", *ids))
    f.write(struct.pack("<q", PREFILL_LEN))

print(f"wrote test_tokens.bin: {len(ids)} tokens, prefill_len={PREFILL_LEN}, n_decode={N_DECODE}")
print("tokens:", ids)
print("decoded:", tok.decode(ids))
