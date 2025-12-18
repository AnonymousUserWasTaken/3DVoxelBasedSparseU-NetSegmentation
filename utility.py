# utility.py
from __future__ import annotations
import os, random, math
from typing import Optional
import numpy as np
import torch

def seed_all(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def as_numpy_1(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim >= 2 and x.shape[0] == 1:
        return x[0]
    return x

def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# =============================================================================
# Random rotation (utility)
# =============================================================================
def random_rotation(pc: torch.Tensor) -> torch.Tensor:
    """
    Apply a random 3D rotation to an (N,3) tensor (right-multiply by R^T).
    """
    assert pc.dim() == 2 and pc.size(-1) == 3
    ax, ay, az = torch.rand(3, device=pc.device) * (2.0 * math.pi)
    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)

    Rx = torch.tensor([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]], dtype=pc.dtype, device=pc.device)
    Ry = torch.tensor([[ cy, 0, sy],
                       [  0, 1,  0],
                       [-sy, 0, cy]], dtype=pc.dtype, device=pc.device)
    Rz = torch.tensor([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]], dtype=pc.dtype, device=pc.device)
    R = (Rz @ Ry @ Rx)
    return pc @ R.T
