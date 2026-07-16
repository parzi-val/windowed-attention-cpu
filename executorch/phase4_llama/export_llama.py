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

WindowedLlamaForExport is also imported directly (e.g. by compare_device_output.py) to build an
eager reference model without re-running this whole export pipeline -- everything below the
class/function definitions is guarded behind __main__ specifically so that import doesn't
re-trigger a full weight load + export + to_executorch + .pte write as a side effect.

    .venv-executorch/Scripts/python.exe export_llama.py [--arm map50]
"""
import sys

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

from real_attention_patch import RealWindowedLlamaAttention  # noqa: E402


class WindowedLlamaForExport(nn.Module):
    """embed_tokens + N patched layers + final norm -- no lm_head, since the exit gate here is
    export survival of the windowed attention mechanism, not a full generation-ready graph."""

    def __init__(self, base, head_windowed_per_layer, sink, window, max_t):
        super().__init__()
        self.embed_tokens = base.embed_tokens
        self.rotary_emb = base.rotary_emb
        self.norm = base.norm
        self.max_t = max_t
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
        torch._check(start_pos < self.max_t)

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


def check_mutation_markers(exported, n_layers, name):
    sg_nodes = [n for n in exported.graph.nodes
                if n.op == "call_function" and "windowed_sdpa_kv_cache" in str(n.target)]
    assert len(sg_nodes) == n_layers, f"[{name}] expected {n_layers} op call nodes, got {len(sg_nodes)}"
    for node in sg_nodes:
        mutated_args = [a.name for a in node.target._schema.arguments if a.alias_info is not None]
        assert mutated_args == ["k_cache", "v_cache"], f"[{name}] node {node} schema mutation mismatch: {mutated_args}"
    print(f"   [{name}] {len(sg_nodes)}/{n_layers} op nodes present, all carry k_cache/v_cache mutation markers")


if __name__ == "__main__":
    import argparse
    import json

    sys.path.insert(0, "../phase3_kv_cache/build")
    import windowed_sdpa_kv_cache_pybind as kernel_module  # noqa: E402
    from real_attention_patch import register_kernel_impl
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

    # bf16, not fp32: matches the reference model's own export recipe (-d bf16), halves the
    # .pte size (2.5GB vs 4.9GB) and halves memory-bandwidth per weight -- decode is GEMV
    # (bandwidth-bound, established in the earlier mmap-eviction investigation), so this should
    # help decode speed on top of whatever XNNPACK delegation buys, not just shrink the file.
    # windowed_sdpa_kv_cache itself stays fp32 internally (RealWindowedLlamaAttention casts
    # around the op call) since the native kernel is fp32-only.
    if args.real_weights:
        from transformers import AutoModelForCausalLM
        print(f"Loading real pretrained meta-llama/Llama-3.2-1B (bf16) with arm={args.arm} "
              f"({sum(sum(r) for r in head_windowed_per_layer)}/{n_layers * 32} heads windowed)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16
        ).model.eval()
        assert base_model.config.num_hidden_layers == n_layers, "arms JSON layer count mismatch vs. real model"
    else:
        print(f"Building Llama-3.2-1B (config only, random init, bf16) with arm={args.arm} "
              f"({sum(sum(r) for r in head_windowed_per_layer)}/{n_layers * 32} heads windowed)...")
        config = LlamaConfig(
            vocab_size=128256, hidden_size=2048, intermediate_size=8192,
            num_hidden_layers=n_layers, num_attention_heads=32, num_key_value_heads=8,
            head_dim=64, max_position_embeddings=2048, rope_theta=500000.0,
        )
        base_model = LlamaModel(config).to(torch.bfloat16).eval()

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
    with torch.no_grad():  # no backward pass needed -- avoids retaining autograd activation buffers
        out = model(*prefill_inputs)
        assert out.shape == (1, args.max_prompt_len, hidden_size)
        out2 = model(*decode_inputs)
        assert out2.shape == (1, 1, hidden_size)
    print(f"   prefill out {tuple(out.shape)}, decode out {tuple(out2.shape)} -- OK", flush=True)

    print("1a. torch.export prefill (fixed shape, start_pos=0)...")
    prefill_exported = torch.export.export(model, prefill_inputs)
    check_mutation_markers(prefill_exported, n_layers, "prefill")

    print("1b. torch.export decode (Tn=1 fixed, dynamic start_pos)...")
    decode_exported = torch.export.export(model, decode_inputs)
    check_mutation_markers(decode_exported, n_layers, "decode")
    print(flush=True)

    # Free the eager model before the heaviest-memory stages (edge lowering + flatbuffer
    # serialization both build their own representation of the ~4.9GB fp32 weights on top of
    # whatever's already resident) -- best-effort peak-RAM reduction, since a real Llama-3.2-1B
    # export on a memory-constrained free-tier VM can OOM-kill silently (no Python traceback, just
    # a truncated/0-byte .pte -- exactly what happened on the first Lightning AI run).
    import gc
    del model, base_model
    gc.collect()

    print("2. to_edge_transform_and_lower (both methods, XNNPACK-delegated)...")
    # XnnpackDynamicallyQuantizedPartitioner(), matching the reference model's own recipe
    # (ExportRecipe_1B.ipynb uses -X alone, no --xnnpack-extended-ops). windowed_sdpa_kv_cache
    # isn't an op XNNPACK recognizes, so it stays a regular CPU-executed node regardless.
    #
    # Needs a real, resolvable flatc binary -- the pip-installed executorch package doesn't
    # bundle one for Windows (_get_flatc_path() falls back to bare "flatc" on PATH, which isn't
    # there, surfacing as a crash deep inside XNNPACK serialization). Linux wheels (Colab,
    # Kaggle) likely bundle a working one via importlib.resources, so only require
    # FLATC_EXECUTABLE as a fallback if the package's own resolution comes up empty -- don't
    # assume Windows's gap applies everywhere.
    import os
    import shutil
    from executorch.exir._serialize._flatbuffer import _get_flatc_path
    resolved_flatc = _get_flatc_path()
    if not (os.path.isfile(resolved_flatc) or shutil.which(resolved_flatc)):
        if "FLATC_EXECUTABLE" not in os.environ:
            raise RuntimeError(
                f"No usable flatc found (resolved to {resolved_flatc!r}, not found on disk or "
                "PATH) and FLATC_EXECUTABLE is not set. Set it to a real flatc binary -- e.g. "
                "_build/executorch/cmake-out-desktop/third-party/flatc_ep/bin/flatc.exe"
            )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    edge_program = to_edge_transform_and_lower(
        {"prefill": prefill_exported, "decode": decode_exported},
        partitioner=[XnnpackDynamicallyQuantizedPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    print("   to_edge OK", flush=True)
    del prefill_exported, decode_exported
    gc.collect()

    print("3. to_executorch ...")
    # share_mutable_buffers=True: prefill and decode are independently exported methods, each
    # with its own lifted k_cache/v_cache buffer parameters. Without this, the runtime gives
    # each method its OWN copy of those buffers (zero-initialized), so decode would silently
    # attend over an empty cache instead of continuing from what prefill just wrote -- matches
    # Module::Module's share_memory_arenas parameter on the runtime/C++ side (both must be set
    # together; see extension/module/module.h's docstring on share_memory_arenas).
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir import ExecutorchBackendConfig
    et_program = edge_program.to_executorch(
        ExecutorchBackendConfig(memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True))
    )
    print(f"   to_executorch OK -- serialized buffer is {len(et_program.buffer):,} bytes", flush=True)

    pte_path = f"llama_3.2_1b_{args.arm}.pte"
    print(f"4. writing {pte_path} ...", flush=True)
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"   wrote {pte_path} ({len(et_program.buffer):,} bytes) -- methods: prefill, decode")
    print()
    print(f"PHASE 4.4 EXPORT SURVIVAL ({args.arm}): PASSED -- full {n_layers}-layer windowed "
          f"Llama-3.2-1B attention survives export -> edge -> .pte as two methods "
          f"(fixed-shape prefill, dynamic-start_pos decode).")
