# mlcore/data.py
import os, glob, math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict
from .utils import voxelize, build_sorted_key_index

def _rand_rotmat_and_trans(augment=True):
    if not augment:
        R = torch.eye(3, dtype=torch.float32); t = torch.zeros(3)
        return R, t
    ax, ay, az = [torch.rand(1).item() * 2 * math.pi for _ in range(3)]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=torch.float32)
    R = Rz @ Ry @ Rx
    t = torch.rand(3) * 0.1
    return R, t

def _pair_norm(seg_aug: torch.Tensor, full_aug: Optional[torch.Tensor]):
    if full_aug is not None and full_aug.numel():
        center = full_aug.mean(0)
        r = (full_aug - center).pow(2).sum(dim=1).sqrt()
        try:
            scale = torch.quantile(r, 0.90).clamp_min(1e-6)
        except AttributeError:
            k = max(int(0.9 * r.numel()), 1)
            scale = r.kthvalue(k).values.clamp_min(1e-6)
    else:
        center = seg_aug.mean(0)
        mins, maxs = seg_aug.min(0).values, seg_aug.max(0).values
        scale = (maxs - mins).norm().clamp_min(1e-6)
    return (seg_aug - center)/scale, None if full_aug is None else (full_aug - center)/scale

class NPZDataset(Dataset):
    """
    Loads .npz samples exported by blender_bridge/bake_dataset.py.
    Each file contains: coords [N,3], full_coords [M,3] (optional), part_label (int), class (str).
    """
    def __init__(self, folder: str, classes: List[str], augment=True):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.paths:
            raise RuntimeError(f"No .npz samples found in {folder}")
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.augment = augment
        self.K = 6  # joints

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = np.load(self.paths[idx], allow_pickle=True)
        seg = torch.from_numpy(d["coords"].astype(np.float32))          # (N,3)
        full = torch.from_numpy(d["full_coords"].astype(np.float32)) if "full_coords" in d.files else None
        part = int(d["part_label"].item())
        cls  = str(d["class"].item())
        objid= torch.tensor(self.class_to_idx.get(cls, 0), dtype=torch.long)

        R_a,t_a = _rand_rotmat_and_trans(self.augment)
        R_b,t_b = _rand_rotmat_and_trans(self.augment)

        seg_a = (seg @ R_a.T) + t_a
        seg_b = (seg @ R_b.T) + t_b
        full_a = None if full is None else (full @ R_a.T) + t_a
        full_b = None if full is None else (full @ R_b.T) + t_b

        seg_a, full_a = _pair_norm(seg_a, full_a)
        seg_b, full_b = _pair_norm(seg_b, full_b)

        K = self.K
        kp_a = torch.zeros(K,3); kp_b = torch.zeros(K,3); kp_mask = 0
        if full_a is not None and full_a.numel():
            off_a = (seg_a.mean(0) - full_a.mean(0)).unsqueeze(0).repeat(K,1)
            off_b = (seg_b.mean(0) - full_b.mean(0)).unsqueeze(0).repeat(K,1)
            kp_a, kp_b = off_a, off_b
            kp_mask = 1

        return {
            "coords_a": seg_a, "coords_b": seg_b,
            "part_label": torch.tensor(part, dtype=torch.long),
            "kp_a": kp_a, "kp_b": kp_b, "kp_mask": torch.tensor(kp_mask, dtype=torch.float32),
            "obj_label": objid
        }

def collate_as_list(batch, feat_dim: int, voxel_size: float):
    """
    Voxelizes each sample independently; returns a list so variable sizes are OK.
    """
    out = []
    for b in batch:
        uc_a, inv_a = voxelize(b["coords_a"], voxel_size)
        _, sk_a, ip_a = build_sorted_key_index(uc_a)
        feats_a = torch.ones((uc_a.size(0), feat_dim), dtype=torch.float32)
        sem_a = torch.full((uc_a.size(0),), int(b["part_label"].item()), dtype=torch.long)

        uc_b, inv_b = voxelize(b["coords_b"], voxel_size)
        _, sk_b, ip_b = build_sorted_key_index(uc_b)
        feats_b = torch.ones((uc_b.size(0), feat_dim), dtype=torch.float32)
        sem_b = torch.full((uc_b.size(0),), int(b["part_label"].item()), dtype=torch.long)

        out.append({
            "uc_a": uc_a, "feats_a": feats_a, "sk_a": sk_a, "ip_a": ip_a, "sem_a": sem_a,
            "uc_b": uc_b, "feats_b": feats_b, "sk_b": sk_b, "ip_b": ip_b, "sem_b": sem_b,
            "kp_a": b["kp_a"], "kp_b": b["kp_b"], "kp_mask": b["kp_mask"],
            "obj_label": b["obj_label"],
        })
    return out
