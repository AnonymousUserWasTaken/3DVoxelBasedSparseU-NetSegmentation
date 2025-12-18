# loss.py
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore = ignore_index; self.smooth = smooth
    def forward(self, logits, target):
        B, C = logits.shape[:2]
        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore)
        if valid.sum() == 0:
            return logits.new_tensor(0.0)
        loss = 0.0
        for c in range(C):
            pc = probs[:, c]; tc = (target == c) & valid
            pc = pc[valid]; tc = tc[valid].float()
            inter = (pc * tc).sum(); denom = pc.sum() + tc.sum()
            dice = (2*inter + self.smooth) / (denom + self.smooth)
            loss += (1 - dice)
        return loss / C

def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                          alpha: float=0.25, gamma: float=2.0, reduction="mean"):
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob*targets + (1-prob)*(1-targets)
    if alpha is not None:
        alpha_t = alpha*targets + (1-alpha)*(1-targets)
        loss = alpha_t * (1 - p_t) ** gamma * ce
    else:
        loss = (1 - p_t) ** gamma * ce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def cross_entropy_band_weighted(logits: torch.Tensor,
                                target: torch.Tensor,
                                band_mask: torch.Tensor,
                                class_weights: Optional[torch.Tensor] = None,
                                band_factor: float = 4.0):
    ce_per_voxel = F.cross_entropy(logits, target, weight=class_weights, ignore_index=255, reduction='none')
    w = torch.ones_like(ce_per_voxel)
    band = band_mask.squeeze(1)  # (B,D,H,W)
    w = w + (band_factor - 1.0) * band
    valid = (target != 255).float()
    ce = (ce_per_voxel * w * valid).sum() / (valid.sum().clamp_min(1.0))
    return ce

def mask_prof_logits_illegal(prof_logits: torch.Tensor,
                             target_sem: torch.Tensor,
                             allowed_per_class: Dict[int, set]) -> torch.Tensor:
    """
    For each voxel, set logits of profiles that don't belong to its class to -inf (very negative),
    so CE never chooses them. Leaves valid profiles untouched.
    """
    out = prof_logits.clone()
    B, P = out.size(0), out.size(1)
    neg_large = torch.finfo(out.dtype).min
    device = out.device

    for c, allowed in allowed_per_class.items():
        illegal = torch.ones(P, dtype=torch.bool, device=device)
        if allowed:
            idx = torch.tensor(sorted(list(allowed)), device=device, dtype=torch.long)
            illegal[idx] = False
        mask_vox = (target_sem == c).unsqueeze(1)  # (B,1,D,H,W)
        mask_full = mask_vox & illegal.view(1, P, 1, 1, 1)
        out.masked_fill_(mask_full, neg_large)
    return out
