# mlcore/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import build_sorted_key_index, voxelize
from .sparse_ops import SparseKernelConv

# default 3×3×3 neighborhood offsets (odd kernel)
def make_kernel_offsets(kernel_size: int = 3, dilation: int = 1) -> torch.Tensor:
    assert kernel_size % 2 == 1
    r = kernel_size // 2
    offs = []
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            for dz in range(-r, r+1):
                offs.append((dx * dilation, dy * dilation, dz * dilation))
    return torch.tensor(offs, dtype=torch.int64)

NEIGHBORHOOD_OFFSETS = make_kernel_offsets(3, 1)

def scatter_mean(src: torch.Tensor, index: torch.Tensor):
    """Mean pool features by integer index (0..max)."""
    dim_size = int(index.max().item()) + 1 if index.numel() else 0
    if dim_size == 0:
        return src.new_zeros((0, src.size(1)))
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    counts = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    ones = torch.ones(index.size(0), device=src.device, dtype=src.dtype)
    counts.index_add_(0, index, ones)
    counts = counts.clamp_min(1).unsqueeze(-1)
    return out / counts

class SparseUNetMultiTask(nn.Module):
    def __init__(self, base_channels=32, num_semantic=8, num_obj=4, num_joints=6):
        super().__init__()
        C = base_channels
        self.base_channels = C

        self.conv1 = SparseKernelConv(C, C, NEIGHBORHOOD_OFFSETS, use_film=False)
        self.conv2 = SparseKernelConv(C, 2*C, NEIGHBORHOOD_OFFSETS, use_film=False)

        self.up_proj   = nn.Linear(2*C, C)
        self.cond_proj = nn.Linear(2*C, C)
        self.fuse      = SparseKernelConv(C, C, NEIGHBORHOOD_OFFSETS, use_film=True, cond_dim=C)

        self.sem_head  = nn.Linear(C, num_semantic)  # per-voxel
        self.cls_head  = nn.Linear(C, num_obj)       # global
        self.embed_head= nn.Linear(C, C)             # contrastive
        self.kp_heads  = nn.ModuleList([nn.Linear(C,3) for _ in range(num_joints)])

    def forward(self, coords, feats, sk, ip):
        # enc
        x1 = self.conv1(feats, coords, sk, ip)              # (M,C)

        # coarse grid (downsample by factor 2)
        coarse = (coords.float() // 2).long()
        uc, invc = torch.unique(coarse, return_inverse=True, dim=0)
        _, skc, ipc = build_sorted_key_index(uc)

        x2 = self.conv2(scatter_mean(x1, invc), uc, skc, ipc)   # (Mc,2C)

        # fuse
        up   = self.up_proj(x2)[invc]                           # (M,C)
        cond = self.cond_proj(x2)[invc]                         # (M,C)
        xr   = self.fuse(x1 + up, coords, sk, ip, cond=cond)    # (M,C)

        # heads
        sem_logits = self.sem_head(xr)                          # (M,num_sem)
        pooled     = xr.mean(0, keepdim=True)                   # (1,C)
        cls_logits = self.cls_head(pooled).squeeze(0)
        embed      = F.normalize(self.embed_head(pooled), dim=-1)
        kp_preds   = torch.stack([h(pooled).squeeze(0) for h in self.kp_heads], dim=0)
        return sem_logits, cls_logits, embed, kp_preds
