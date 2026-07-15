"""Phase 4.6 diagnostic, round 3: bisecting the XNNPACK segfault further. Round 2 showed the
tiny synthetic model (just windowed_sdpa_kv_cache + mutable buffers) does NOT crash, but the
real 16-layer model does. This tests ONE real Llama layer's worth of structure -- real RoPE +
real GQA repeat_kv + our op, using RealWindowedLlamaAttention directly (no RMSNorm, no MLP, no
residual, no stacking) -- to isolate whether the trigger is RoPE/GQA-specific or requires more
of the full transformer block / multi-layer depth.

    .venv-executorch/Scripts/python.exe diagnose_xnnpack_one_layer.py
"""
import sys
import torch

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402
from real_attention_patch import RealWindowedLlamaAttention, register_kernel_impl  # noqa: E402

register_kernel_impl(kernel_module)

from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

config = LlamaConfig(
    vocab_size=128256, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=1, num_attention_heads=32, num_key_value_heads=8,
    head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
)
base_model = LlamaModel(config).eval()
layer = base_model.layers[0]
rotary_emb = base_model.rotary_emb

MAX_T = 64 + 256
SINK, WINDOW = 4, 64
head_windowed = [i % 2 == 0 for i in range(config.num_attention_heads)]
patched = RealWindowedLlamaAttention(layer.self_attn, head_windowed, SINK, WINDOW, 1, MAX_T).eval()


class OneLayerWrapper(torch.nn.Module):
    """Just: hidden_states -> RealWindowedLlamaAttention -> attn_out. No RMSNorm, no MLP, no
    residual, no stacking -- the minimal real-RoPE-plus-real-GQA-plus-our-op unit."""

    def __init__(self, attn, rotary_emb, max_t):
        super().__init__()
        self.attn = attn
        self.rotary_emb = rotary_emb
        self.max_t = max_t

    def forward(self, hidden_states: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        torch._check(start_pos < self.max_t)
        seq_len = hidden_states.shape[1]
        position_ids = (torch.arange(seq_len, device=hidden_states.device) + start_pos).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        attn_out, _ = self.attn(hidden_states, start_pos=start_pos, position_ids=position_ids,
                                 position_embeddings=(cos, sin))
        return attn_out


model = OneLayerWrapper(patched, rotary_emb, MAX_T).eval()
decode_inputs = (torch.randn(1, 1, config.hidden_size), torch.tensor([64], dtype=torch.long))

print("eager sanity check...", flush=True)
out = model(*decode_inputs)
print(f"eager OK, out shape {tuple(out.shape)}", flush=True)

print("torch.export...", flush=True)
exported = torch.export.export(model, decode_inputs)
print("export OK", flush=True)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

print("to_edge_transform_and_lower (one real layer: RoPE + GQA repeat_kv + our op)...", flush=True)
edge_program = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackDynamicallyQuantizedPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
print("to_edge OK -- no crash on one real layer", flush=True)
