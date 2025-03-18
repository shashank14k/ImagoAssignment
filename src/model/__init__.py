import src.model.model as ModelDir
from typing import Dict


def init_model(name: str, input_dim: int, device: str = "cuda", params: Dict={}):
    mod = getattr(ModelDir, name)
    mod = mod(input_dim, **params)
    return mod.to(device).train()
