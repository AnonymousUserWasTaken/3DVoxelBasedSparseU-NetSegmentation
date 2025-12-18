# mlcore/sparse_ops.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import gather_neighbors_tensorized

class SparseKernelConv(nn.Module):
    """
    Per-offset shared linear kernel (TorchSparse-style):
      out[i] = sum_k( X[nbr(i,k)] @ W[k] ), mask invalid neighbors to 0.

    Args:
        in_ch:  input feature dim
        out_ch: output feature dim
        offsets: (K,3) LongTensor of voxel offsets
        use_film: FiLM modulation (gamma/beta) with cond_dim
    """
    def __init__(self, in_ch, out_ch, offsets: torch.Tensor, use_film=False, cond_dim=None):
        super().__init__()
        K = int(offsets.size(0))
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.offsets = offsets.long()
        self.weight = nn.Parameter(torch.empty(K, in_ch, out_ch))
        self.bias   = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=math_sqrt(5.0))

        self.res_proj = None if in_ch == out_ch else nn.Linear(in_ch, out_ch)
        self.use_film = use_film
        if use_film:
            assert cond_dim is not None, "cond_dim required when use_film=True"
            self.gamma = nn.Linear(cond_dim, out_ch)
            self.beta  = nn.Linear(cond_dim, out_ch)

    def forward(self, feats, coords, sk, ip, cond=None):
        """
        feats:  (M, Cin)
        coords: (M, 3) int voxel coords
        sk:     sorted keys (from build_sorted_key_index)
        ip:     inv_perm mapping (sorted->original)
        cond:   (M, Ccond) optional FiLM conditioning
        """
        M, Cin = feats.size(0), feats.size(1)
        device = feats.device

        nbr_idx = gather_neighbors_tensorized(coords, sk, self.offsets.to(device))  # (M,K)
        mask    = (nbr_idx < 0)
        safe    = nbr_idx.clone()
        safe[mask] = 0  # dummy index

        # map sorted index -> original row index
        stoo = torch.empty_like(ip)
        stoo[ip] = torch.arange(ip.size(0), device=device)

        # gather neighbor feats
        gather_i = stoo[safe]                              # (M,K)
        K = self.offsets.size(0)
        nbr_feats = feats[gather_i.view(-1)].view(M, K, Cin)  # (M,K,Cin)
        nbr_feats = nbr_feats.masked_fill(mask.unsqueeze(-1), 0.0)

        # einsum over offsets with per-offset weights
        #   nbr_feats: (M,K,Cin), weight: (K,Cin,Cout) => (M,Cout)
        out = torch.einsum("mkc,kco->mo", nbr_feats, self.weight) + self.bias

        if self.use_film and cond is not None:
            out = self.gamma(cond) * out + self.beta(cond)

        res = feats if (self.res_proj is None) else self.res_proj(feats)
        return out + res


def math_sqrt(x: float) -> float:
    return x ** 0.5
