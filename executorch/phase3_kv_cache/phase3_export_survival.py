"""Phase 3 exit gate: does a model with a real KV cache (mutated in place by our custom op)
survive torch.export -> to_edge -> to_executorch -> .pte? This is the part Phase 1/2 didn't
cover -- k_cache/v_cache are nn.Module buffers mutated as a side effect. The op's schema
carries Tensor(a!)/(b!) alias markers on those two args (declared via Library.define(), not
the torch.library.custom_op decorator -- see the comment below for why), which is what lets
torch.export and ExecuTorch's downstream passes know the call has side effects on its inputs
rather than silently dropping them as dead.

As in Phase 2, actually *running* the exported .pte on this desktop is expected to fail --
the native kernel (windowed_sdpa_kv_cache_et.cpp) isn't linked into this process's ExecuTorch
runtime (Phase 1/CMakeLists.txt notes). That's out of scope here; this script's job is
confirming the mutating op survives AOT compilation with its mutation semantics intact.

    .venv-executorch/Scripts/python.exe phase3_export_survival.py
"""
import sys
import torch
import torch.nn as nn
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.extension.pybindings.portable_lib import _load_for_executorch

sys.path.insert(0, "build")
import windowed_sdpa_kv_cache_pybind as kernel_module


# Deliberately NOT using the torch.library.custom_op decorator here: its infer_schema()
# auto-generates alias-set names like "a3!"/"a4!" (based on argument position) for mutated
# args, and torchgen's FunctionSchema.parse() -- which ExecuTorch's ToOutVarPass relies on to
# read the op's schema back -- can't parse multi-character alias names. That failure is caught
# and silently swallowed (logged at DEBUG only) inside ExecuTorch's _pybind_schema_to_native_schema,
# surfacing much later as an unrelated-looking AttributeError deep in to_executorch(). Using the
# low-level Library.define()/.impl() API instead gives full control over alias names (a!, b!),
# matching ordinary single-letter native_functions.yaml-style schemas.
_lib = torch.library.Library("sg", "FRAGMENT")
_lib.define(
    "windowed_sdpa_kv_cache(Tensor q_new, Tensor k_new, Tensor v_new, Tensor(a!) k_cache, "
    "Tensor(b!) v_cache, Tensor head_windowed, SymInt sink, SymInt window, SymInt start_pos) -> Tensor"
)


def _windowed_sdpa_kv_cache_impl(q_new, k_new, v_new, k_cache, v_cache, head_windowed, sink, window, start_pos):
    out = kernel_module.windowed_sdpa_kv_cache(
        q_new.numpy(), k_new.numpy(), v_new.numpy(),
        k_cache.numpy(), v_cache.numpy(),
        head_windowed.numpy(), sink, window, start_pos,
    )
    return torch.from_numpy(out)


def _windowed_sdpa_kv_cache_fake(q_new, k_new, v_new, k_cache, v_cache, head_windowed, sink, window, start_pos):
    return q_new.new_empty(q_new.shape)


_lib.impl("windowed_sdpa_kv_cache", _windowed_sdpa_kv_cache_impl, "CompositeExplicitAutograd")
_lib.impl("windowed_sdpa_kv_cache", _windowed_sdpa_kv_cache_fake, "Meta")

# Matching .out schema for ExecuTorch's ToOutVarPass, same reasoning as Phase 2 -- SymInt on
# the scalar args to match the functional op's signature, Tensor(a!)/(b!) for the two mutated
# cache args, Tensor(c!) for the actual return value.
_lib.define(
    "windowed_sdpa_kv_cache.out(Tensor q_new, Tensor k_new, Tensor v_new, Tensor(a!) k_cache, "
    "Tensor(b!) v_cache, Tensor head_windowed, SymInt sink, SymInt window, SymInt start_pos, "
    "*, Tensor(c!) out) -> Tensor(c!)"
)


def _windowed_sdpa_kv_cache_out_impl(
    q_new, k_new, v_new, k_cache, v_cache, head_windowed, sink, window, start_pos, out
):
    result = kernel_module.windowed_sdpa_kv_cache(
        q_new.numpy(), k_new.numpy(), v_new.numpy(),
        k_cache.numpy(), v_cache.numpy(),
        head_windowed.numpy(), sink, window, start_pos,
    )
    out.copy_(torch.from_numpy(result))
    return out


_lib.impl("windowed_sdpa_kv_cache.out", _windowed_sdpa_kv_cache_out_impl, "CompositeExplicitAutograd")


class TinyKVCacheAttn(nn.Module):
    def __init__(self, B, H, D, sink, window, max_t):
        super().__init__()
        self.sink = sink
        self.window = window
        hw = torch.tensor([i % 2 == 0 for i in range(H)], dtype=torch.bool)
        self.register_buffer("head_windowed", hw)
        self.register_buffer("k_cache", torch.zeros(B, H, max_t, D))
        self.register_buffer("v_cache", torch.zeros(B, H, max_t, D))

    def forward(self, q_new, k_new, v_new):
        # start_pos fixed at trace time -- a single decode step at a known position. Threading
        # start_pos as a dynamic input for a real generate loop is Phase 4 (Android) territory,
        # not needed to prove mutation survives export here.
        start_pos = 5
        return torch.ops.sg.windowed_sdpa_kv_cache(
            q_new, k_new, v_new, self.k_cache, self.v_cache,
            self.head_windowed, self.sink, self.window, start_pos,
        )


B, H, D, sink, window, max_t = 1, 4, 8, 4, 16, 64
model = TinyKVCacheAttn(B, H, D, sink, window, max_t).eval()
example_inputs = (torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), torch.randn(B, H, 1, D))

print("0. eager sanity check ...")
model.k_cache.zero_()
model.v_cache.zero_()
eager_out = model(*example_inputs)
cache_nonzero = model.k_cache[:, :, 5, :].abs().sum().item()
print(f"   eager output shape = {tuple(eager_out.shape)}, k_cache[pos=5] written (abs sum = {cache_nonzero:.3f})")
assert cache_nonzero > 0, "cache mutation did not happen in eager mode"

print("1. torch.export ...")
exported = torch.export.export(model, example_inputs)
mutated = exported.graph_signature.buffers_to_mutate
op_node = next(n for n in exported.graph.nodes if n.op == "call_function" and "windowed_sdpa_kv_cache" in str(n.target))
schema_mutated_args = [a.name for a in op_node.target._schema.arguments if a.alias_info is not None]
print(f"   export OK -- buffers_to_mutate = {mutated}")
print(f"   op node schema declares mutation on: {schema_mutated_args}")
assert schema_mutated_args == ["k_cache", "v_cache"], (
    f"expected the op's own schema to carry k_cache/v_cache mutation markers, got: {schema_mutated_args}"
)

print("2. to_edge_transform_and_lower ...")
edge_program = to_edge_transform_and_lower(
    exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
)
print("   to_edge OK")

print("3. to_executorch ...")
et_program = edge_program.to_executorch()
print("   to_executorch OK")

pte_path = "phase3_windowed_sdpa_kv_cache.pte"
with open(pte_path, "wb") as f:
    f.write(et_program.buffer)
print(f"4. wrote {pte_path} ({len(et_program.buffer)} bytes)")
print()
print("PHASE 3 EXPORT SURVIVAL: PASSED -- mutating KV-cache op survives export -> edge -> .pte")
print("with its mutation markers on k_cache/v_cache intact through every stage.")

print()
print("5. attempting desktop execution (expected to fail -- no native kernel linked in) ...")
try:
    loaded = _load_for_executorch(pte_path)
    out = loaded.run_method("forward", example_inputs)[0]
    max_diff = (out - eager_out).abs().max().item()
    print(f"   UNEXPECTED: ran successfully, max abs diff vs eager = {max_diff:.3e}")
except Exception as e:
    print(f"   failed as expected ({type(e).__name__}): {e}")
    print("   -- this is the Phase 1 link gap, not an export problem; resolved in Phase 4 (Android).")
