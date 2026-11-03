"""Low-rank adapters for experiments that fine-tune the backbone.

By default, adapt layers 0..L, which produce the memory block's queries. Graph
keys remain fixed: re-encoding 10**5 to 10**6 names per step is impractical.
Unwrap MemLayer before finding target linear layers.
"""
import math

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """base(x) + scale * B(A(x)), with fp32 A and zero-initialized B."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(x.dtype)


def add_lora(layers: nn.ModuleList, upto: int, r: int = 16, alpha: float = 32.0, targets=TARGETS) -> dict:
    """Wrap every target linear in layers[0..upto]; returns {name: LoRALinear}."""
    out = {}
    for i in range(upto + 1):
        layer = layers[i]
        layer = getattr(layer, "layer", layer)                 # kgmem's MemLayer wraps the attach layer
        for name, mod in list(layer.named_modules()):
            for t in targets:
                child = getattr(mod, t, None)
                if isinstance(child, nn.Linear):
                    setattr(mod, t, LoRALinear(child, r, alpha))
                    out[f"{i}.{name}.{t}".replace("..", ".")] = getattr(mod, t)
    return out


def lora_state(mods: dict) -> dict:
    return {f"{k}.{p}": getattr(m, p).detach().cpu().clone() for k, m in mods.items() for p in ("A", "B")}


def load_lora(mods: dict, sd: dict):
    for k, m in mods.items():
        m.A.data.copy_(sd[f"{k}.A"]); m.B.data.copy_(sd[f"{k}.B"])
