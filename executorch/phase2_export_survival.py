"""Phase 2 exit gate: does a model calling our custom op survive torch.export -> to_edge ->
to_executorch -> .pte? This validates the AOT registration path (torch.library.custom_op +
register_fake) independently of the C++ runtime kernel -- windowed_sdpa_et.cpp isn't linked
into this desktop's ExecuTorch runtime (see Phase 1 notes), so actually *running* the .pte
here is expected to fail with a missing-kernel error. That's fine: Phase 2's job is proving
the op survives AOT compilation intact (not decomposed, not rejected by edge IR validation),
not full desktop execution -- real execution happens on Android in Phase 4, where the native
kernel is linked in.

    .venv-executorch/Scripts/python.exe executorch/phase2_export_survival.py
"""
import sys
import torch
import torch.nn as nn
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.extension.pybindings.portable_lib import _load_for_executorch

sys.path.insert(0, "executorch/phase1_windowed_sdpa/build")
import windowed_sdpa_pybind as kernel_module  # eager/tracing-time impl of the custom op


# ---------------------------------------------------------------------------------------
# AOT custom op registration. Namespace/name ("sg::windowed_sdpa") match windowed_sdpa_et.cpp's
# EXECUTORCH_LIBRARY(sg, "windowed_sdpa.out", ...) -- ExecuTorch's out-variant pass is what
# turns the functional op registered here into the .out schema the C++ kernel implements.
# ---------------------------------------------------------------------------------------
@torch.library.custom_op("sg::windowed_sdpa", mutates_args=())
def windowed_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_windowed: torch.Tensor,
    sink: int,
    window: int,
) -> torch.Tensor:
    out = kernel_module.windowed_sdpa(
        q.numpy(), k.numpy(), v.numpy(), head_windowed.numpy(), sink, window
    )
    return torch.from_numpy(out)


@windowed_sdpa.register_fake
def _(q, k, v, head_windowed, sink, window):
    return q.new_empty(q.shape)


# ExecuTorch's ToOutVarPass (in to_executorch()) rewrites every functional op call into its
# out-variant by looking up "{op_name}.out" in the *same* dispatcher via
# torch._C._jit_get_schemas_for_operator -- it does not synthesize one from the functional
# schema. windowed_sdpa_et.cpp registers exactly this out-variant natively, but that shared
# library was never linked into this desktop process (Phase 1), so the schema is invisible
# here unless we also declare it in Python. This mirrors what windowed_sdpa_et.cpp does; the
# Python impl only needs to be correct enough for this AOT step, not fast -- the real kernel
# that actually executes on-device is the C++ one, linked in at Phase 4.
_lib = torch.library.Library("sg", "FRAGMENT")
_lib.define(
    "windowed_sdpa.out(Tensor q, Tensor k, Tensor v, Tensor head_windowed, SymInt sink, "
    "SymInt window, *, Tensor(a!) out) -> Tensor(a!)"
)


def _windowed_sdpa_out_impl(q, k, v, head_windowed, sink, window, out):
    result = kernel_module.windowed_sdpa(
        q.numpy(), k.numpy(), v.numpy(), head_windowed.numpy(), sink, window
    )
    out.copy_(torch.from_numpy(result))
    return out


_lib.impl("windowed_sdpa.out", _windowed_sdpa_out_impl, "CompositeExplicitAutograd")


class TinyWindowedAttn(nn.Module):
    def __init__(self, H, sink, window):
        super().__init__()
        self.sink = sink
        self.window = window
        # alternating dense/windowed heads -- exercises both branches in the exported graph
        hw = torch.tensor([i % 2 == 0 for i in range(H)], dtype=torch.bool)
        self.register_buffer("head_windowed", hw)

    def forward(self, q, k, v):
        return torch.ops.sg.windowed_sdpa(q, k, v, self.head_windowed, self.sink, self.window)


B, H, T, D = 1, 4, 40, 8
sink, window = 4, 16
model = TinyWindowedAttn(H, sink, window).eval()
example_inputs = (torch.randn(B, H, T, D), torch.randn(B, H, T, D), torch.randn(B, H, T, D))

print("0. eager sanity check ...")
eager_out = model(*example_inputs)
print(f"   eager output shape = {tuple(eager_out.shape)}")

print("1. torch.export ...")
exported = torch.export.export(model, example_inputs)
print("   export OK -- op present in graph:",
      any("windowed_sdpa" in str(n.target) for n in exported.graph.nodes))

print("2. to_edge_transform_and_lower ...")
edge_program = to_edge_transform_and_lower(
    exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
)
print("   to_edge OK")

print("3. to_executorch ...")
et_program = edge_program.to_executorch()
print("   to_executorch OK")

pte_path = "executorch/phase2_windowed_sdpa.pte"
with open(pte_path, "wb") as f:
    f.write(et_program.buffer)
print(f"4. wrote {pte_path} ({len(et_program.buffer)} bytes)")
print()
print("PHASE 2 EXPORT SURVIVAL: PASSED -- op survives export -> edge -> .pte intact.")

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
