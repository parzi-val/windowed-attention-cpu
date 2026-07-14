"""Phase 4.4b exit gate: does the full real Llama-3.2-1B model, with windowed_sdpa_kv_cache
wired into every layer via a map arm, survive torch.export -> to_edge -> to_executorch -> .pte?

Two separate exported graphs, not one unified graph -- attempting a single forward with BOTH
dynamic seq_len AND dynamic start_pos hit an unresolvable ConstraintViolationError from the
symbolic-shape solver (min(seq_len_max*seq_len, MAX_T*seq_len) guard). Isolating the two
dynamism axes showed dynamic start_pos alone (Tn=1 fixed) exports cleanly -- so:
  - "prefill": fixed shape (prompt padded/bucketed to max_prompt_len), start_pos=0 always.
  - "decode": Tn=1 fixed, start_pos dynamic via .item() + torch._check_is_size/torch._check
    (same convention as ExecuTorch's own reference Llama export,
    examples/models/llama/attention.py's enable_dynamic_shape branch) -- this is the graph a
    real generate loop calls every step with an incrementing start_pos, the gap Phase 3's
    fixed start_pos=5 left open.

Random-init weights here -- this validates export mechanics (shapes/ops), not real generation
quality; that comes from build_arms.py's arms + real pretrained weights on Colab in a later step.

    .venv-executorch/Scripts/python.exe export_llama.py [--arm map50]
"""
import argparse
import json
import sys

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402

from real_attention_patch import RealWindowedLlamaAttention, register_kernel_impl  # noqa: E402

register_kernel_impl(kernel_module)

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="map50", choices=["map25", "map50", "map75"])
parser.add_argument("--max-prompt-len", type=int, default=64)
parser.add_argument("--real-weights", action="store_true",
                     help="Load real pretrained meta-llama/Llama-3.2-1B (needs HF auth + ~5GB "
                          "download) instead of random-init -- for the Colab run that produces "
                          "the actual on-device benchmark artifact.")
args = parser.parse_args()

arms_data = json.load(open("llama_3.2_1b_arms.json"))
arm = arms_data["arms"][args.arm]
head_windowed_per_layer = arm["head_windowed"]  # [L][n_heads] bool
n_layers = arms_data["n_layers"]
SINK, WINDOW = arms_data["sink"], arms_data["window"]
MAX_T = args.max_prompt_len + 256  # room for prefill + a generation run, small for this exit gate

if args.real_weights:
    from transformers import AutoModelForCausalLM
    print(f"Loading real pretrained meta-llama/Llama-3.2-1B with arm={args.arm} "
          f"({sum(sum(r) for r in head_windowed_per_layer)}/{n_layers * 32} heads windowed)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", torch_dtype=torch.float32
    ).model.eval()
    assert base_model.config.num_hidden_layers == n_layers, "arms JSON layer count mismatch vs. real model"
else:
    print(f"Building Llama-3.2-1B (config only, random init) with arm={args.arm} "
          f"({sum(sum(r) for r in head_windowed_per_layer)}/{n_layers * 32} heads windowed)...")
    config = LlamaConfig(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192,
        num_hidden_layers=n_layers, num_attention_heads=32, num_key_value_heads=8,
        head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
    )
    base_model = LlamaModel(config).eval()


class WindowedLlamaForExport(nn.Module):
    """embed_tokens + N patched layers + final norm -- no lm_head, since the exit gate here is
    export survival of the windowed attention mechanism, not a full generation-ready graph."""

    def __init__(self, base, head_windowed_per_layer, sink, window, max_t):
        super().__init__()
        self.embed_tokens = base.embed_tokens
        self.rotary_emb = base.rotary_emb
        self.norm = base.norm
        self.layers = nn.ModuleList([
            RealWindowedLlamaAttention(
                base.layers[l].self_attn, head_windowed_per_layer[l], sink, window, 1, max_t
            )
            for l in range(len(base.layers))
        ])
        self.mlps = nn.ModuleList([base.layers[l].mlp for l in range(len(base.layers))])
        self.input_norms = nn.ModuleList([base.layers[l].input_layernorm for l in range(len(base.layers))])
        self.post_norms = nn.ModuleList([base.layers[l].post_attention_layernorm for l in range(len(base.layers))])

    def forward(self, input_ids: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        # Same dynamic-shape convention as ExecuTorch's own Llama export
        # (examples/models/llama/attention.py, enable_dynamic_shape branch): extract a Python
        # int from a Tensor input via .item(), then tell the tracer it's a size-like symbolic
        # value so it isn't baked in as a compile-time constant.
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        torch._check(start_pos < MAX_T)

        seq_len = input_ids.shape[1]
        hidden = self.embed_tokens(input_ids)
        position_ids = (torch.arange(seq_len, device=input_ids.device) + start_pos).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden, position_ids)

        for i in range(len(self.layers)):
            residual = hidden
            h = self.input_norms[i](hidden)
            attn_out, _ = self.layers[i](h, start_pos=start_pos, position_ids=position_ids,
                                          position_embeddings=(cos, sin))
            hidden = residual + attn_out
            residual = hidden
            h = self.post_norms[i](hidden)
            hidden = residual + self.mlps[i](h)

        return self.norm(hidden)


def check_mutation_markers(exported, name):
    sg_nodes = [n for n in exported.graph.nodes
                if n.op == "call_function" and "windowed_sdpa_kv_cache" in str(n.target)]
    assert len(sg_nodes) == n_layers, f"[{name}] expected {n_layers} op call nodes, got {len(sg_nodes)}"
    for node in sg_nodes:
        mutated_args = [a.name for a in node.target._schema.arguments if a.alias_info is not None]
        assert mutated_args == ["k_cache", "v_cache"], f"[{name}] node {node} schema mutation mismatch: {mutated_args}"
    print(f"   [{name}] {len(sg_nodes)}/{n_layers} op nodes present, all carry k_cache/v_cache mutation markers")


model = WindowedLlamaForExport(base_model, head_windowed_per_layer, SINK, WINDOW, MAX_T).eval()
vocab_size, hidden_size = base_model.config.vocab_size, base_model.config.hidden_size

prefill_inputs = (
    torch.randint(0, vocab_size, (1, args.max_prompt_len)),
    torch.zeros(1, dtype=torch.long),  # start_pos = 0, always -- prefill is a fresh generation
)
decode_inputs = (
    torch.randint(0, vocab_size, (1, 1)),
    torch.tensor([args.max_prompt_len], dtype=torch.long),  # example start_pos, made dynamic below
)

print("0. eager sanity check (prefill + one decode step)...")
out = model(*prefill_inputs)
assert out.shape == (1, args.max_prompt_len, hidden_size)
out2 = model(*decode_inputs)
assert out2.shape == (1, 1, hidden_size)
print(f"   prefill out {tuple(out.shape)}, decode out {tuple(out2.shape)} -- OK")

print("1a. torch.export prefill (fixed shape, start_pos=0)...")
prefill_exported = torch.export.export(model, prefill_inputs)
check_mutation_markers(prefill_exported, "prefill")

print("1b. torch.export decode (Tn=1 fixed, dynamic start_pos)...")
decode_exported = torch.export.export(model, decode_inputs)
check_mutation_markers(decode_exported, "decode")

print("2. to_edge_transform_and_lower (both methods together)...")
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
edge_program = to_edge_transform_and_lower(
    {"prefill": prefill_exported, "decode": decode_exported},
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
print("   to_edge OK")

print("3. to_executorch ...")
et_program = edge_program.to_executorch()
print("   to_executorch OK")

pte_path = f"llama_3.2_1b_{args.arm}.pte"
with open(pte_path, "wb") as f:
    f.write(et_program.buffer)
print(f"4. wrote {pte_path} ({len(et_program.buffer):,} bytes) -- methods: prefill, decode")
print()
print(f"PHASE 4.4 EXPORT SURVIVAL ({args.arm}): PASSED -- full {n_layers}-layer windowed "
      f"Llama-3.2-1B attention survives export -> edge -> .pte as two methods "
      f"(fixed-shape prefill, dynamic-start_pos decode).")
