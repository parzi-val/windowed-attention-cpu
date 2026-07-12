"""Phase 0 exit gate: export a self-authored model to .pte and run it on desktop via the
ExecuTorch Python runtime. Proves the export -> to_edge -> to_executorch -> save -> load ->
execute round trip works on this toolchain before any custom op or real LLM checkpoint is
involved. Deliberately a tiny model -- this is a plumbing test, not a benchmark.

    .venv-executorch/Scripts/python.exe executorch/phase0_toolchain_spike.py
"""
import torch
import torch.nn as nn
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import _load_for_executorch


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model = TinyMLP().eval()
example_inputs = (torch.randn(1, 16),)

print("1. torch.export ...")
exported = torch.export.export(model, example_inputs)

print("2. to_edge_transform_and_lower ...")
edge_program = to_edge_transform_and_lower(exported)

print("3. to_executorch ...")
et_program = edge_program.to_executorch()

pte_path = "executorch/tiny_mlp.pte"
with open(pte_path, "wb") as f:
    f.write(et_program.buffer)
print(f"4. wrote {pte_path} ({len(et_program.buffer)} bytes)")

print("5. loading and running via _load_for_executorch ...")
loaded = _load_for_executorch(pte_path)
out = loaded.run_method("forward", example_inputs)[0]

ref = model(*example_inputs)
max_diff = (out - ref).abs().max().item()
print(f"6. parity check (exported .pte vs eager PyTorch): max abs diff = {max_diff:.3e}")
assert max_diff < 1e-4, "PHASE 0 FAILED -- exported .pte does not match eager output"
print("PHASE 0 PASSED -- toolchain round trip verified.")
