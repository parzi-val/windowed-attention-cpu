"""Phase 4.7 diagnostic: is XNNPACK ACTUALLY delegating anything in our exported model? Both
partitioner variants (DQ-only and plain) gave ~18s/decode on-device, ~25x slower than the
reference's ~700ms -- suggesting delegation may not be taking effect at all, or our custom op
fragments the graph so badly that almost nothing gets delegated. This counts, per method:
  - total call_function nodes
  - executorch_call_delegate nodes (each = one XNNPACK-delegated subgraph)
  - windowed_sdpa_kv_cache nodes (our op, never delegated -- graph-fragmenting barriers)
  - the top non-delegated op types by count (what's still running on plain CPU)

Random-init bf16 (fast, structure is identical to real weights).

    .venv-executorch/Scripts/python.exe diagnose_delegation.py
"""
import sys
from collections import Counter

import torch

sys.path.insert(0, "../phase3_kv_cache/build")
import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402
from real_attention_patch import register_kernel_impl  # noqa: E402
from export_llama import WindowedLlamaForExport  # noqa: E402

register_kernel_impl(kernel_module)

from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

config = LlamaConfig(
    vocab_size=128256, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
    head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
)
_dtype = torch.float32 if (len(sys.argv) > 1 and sys.argv[1] == "fp32") else torch.bfloat16
print(f"=== dtype: {_dtype} ===")
base_model = LlamaModel(config).to(_dtype).eval()
MAX_T, SINK, WINDOW = 64 + 256, 4, 64
head_windowed_per_layer = [[i % 2 == 0 for i in range(32)] for _ in range(16)]
model = WindowedLlamaForExport(base_model, head_windowed_per_layer, SINK, WINDOW, MAX_T).eval()

decode_inputs = (torch.randint(0, config.vocab_size, (1, 1)), torch.tensor([64], dtype=torch.long))
exported = torch.export.export(model, decode_inputs)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

edge = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)

gm = edge.exported_program().graph_module
n_delegate = 0
n_windowed = 0
non_delegated_ops = Counter()
for node in gm.graph.nodes:
    if node.op != "call_function":
        continue
    tname = str(node.target)
    if "call_delegate" in tname:
        n_delegate += 1
    elif "windowed_sdpa_kv_cache" in tname:
        n_windowed += 1
    else:
        # strip to a short op name
        short = tname.split(".")[-2] if "." in tname else tname
        non_delegated_ops[short] += 1

print(f"XNNPACK delegate subgraphs (executorch_call_delegate): {n_delegate}")
print(f"windowed_sdpa_kv_cache barriers (our op, never delegated): {n_windowed}")
print(f"total non-delegated, non-custom call_function nodes: {sum(non_delegated_ops.values())}")
print("\ntop non-delegated op types still on plain CPU:")
for op, cnt in non_delegated_ops.most_common(20):
    print(f"  {cnt:4d}  {op}")
