# inference_humans.py
# Detect humans + predict their joints from a 3D point cloud.
# Works with the model/featurization used in Voxelization.py.

import os
import sys
import json
import math
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Blender; optional outside Blender
try:
    import bpy
    _HAS_BPY = True
except Exception:
    _HAS_BPY = False

# ─────────────────────────────────────────────────────────────────────────────
# Make sure we can import local helpers from the .blend folder (or script dir)
# ─────────────────────────────────────────────────────────────────────────────
if _HAS_BPY and bpy.data.filepath:
    blend_dir = os.path.dirname(bpy.data.filepath)
else:
    # Fallback when running outside Blender
    blend_dir = os.path.dirname(os.path.abspath(__file__))

if blend_dir and blend_dir not in sys.path:
    sys.path.insert(0, blend_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Local imports (must be in the same folder as this script or .blend)
# ─────────────────────────────────────────────────────────────────────────────
from tof_helpers import (
    voxelize,
    build_sorted_key_index,
    scatter_mean,
    neighborhood_offsets,
    gather_neighbors_tensorized,
    extract_points_from_object,  # optional convenience
)

# ─────────────────────────────────────────────────────────────────────────────
# Model (must match Voxelization.py)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_rulebook(coords: torch.Tensor, sk: torch.Tensor, ip: torch.Tensor,
                   offsets: torch.Tensor) -> torch.Tensor:
    nbr_sorted = gather_neighbors_tensorized(coords, sk, offsets)  # (M,K) in sorted-key domain
    mask = (nbr_sorted < 0)
    stoo = torch.empty_like(ip)
    stoo[ip] = torch.arange(ip.numel(), device=ip.device, dtype=ip.dtype)
    nbr_feat = nbr_sorted.clone()
    nbr_feat[mask] = 0
    nbr_feat = stoo[nbr_feat]
    nbr_feat[mask] = -1
    return nbr_feat


class SparseKernelConv(nn.Module):
    def __init__(self, in_ch, out_ch, offsets, bias=True, reduce='mean',
                 use_film=False, cond_dim=None):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.offsets = offsets
        self.K       = int(offsets.size(0))
        self.reduce  = reduce
        self.weight  = nn.Parameter(torch.randn(self.K, in_ch, out_ch) * (1.0 / math.sqrt(in_ch)))
        self.bias    = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else None
        self.use_film = use_film
        if use_film:
            assert cond_dim is not None
            self.gamma = nn.Linear(cond_dim, out_ch)
            self.beta  = nn.Linear(cond_dim, out_ch)

    def forward(self, feats, *, rulebook: torch.Tensor, cond=None):
        M, Cin = feats.size(0), feats.size(1)
        mask = (rulebook >= 0).unsqueeze(-1)
        safe = rulebook.clamp(min=0)
        gathered = feats[safe.view(-1)].view(M, self.K, Cin) * mask.float()
        out_mko  = torch.einsum('mkc,kco->mko', gathered, self.weight)
        if self.reduce == 'sum':
            out = out_mko.sum(dim=1)
        else:
            denom = mask.sum(dim=1).clamp(min=1).to(out_mko.dtype)
            out   = out_mko.sum(dim=1) / denom
        if self.use_film and cond is not None:
            out = self.gamma(cond) * out + self.beta(cond)
        if self.bias is not None:
            out = out + self.bias
        out = out + (self.res_proj(feats) if self.res_proj is not None else feats)
        return out


class SparseUNetMultiTask(nn.Module):
    def __init__(self, base_channels=32, num_semantic=2, num_obj=2, num_joints=6):
        super().__init__()
        self.base_channels = base_channels
        self.num_joints    = num_joints

        self.conv1 = SparseKernelConv(base_channels, base_channels, neighborhood_offsets)
        self.conv2 = SparseKernelConv(base_channels, base_channels * 2, neighborhood_offsets)

        self.cond_proj = nn.Linear(base_channels * 2, base_channels)
        self.up_proj   = nn.Linear(base_channels * 2, base_channels)
        self.fuse_conv = SparseKernelConv(
            base_channels, base_channels, neighborhood_offsets,
            use_film=True, cond_dim=base_channels
        )

        self.sem_head   = nn.Linear(base_channels, num_semantic)
        self.cls_head   = nn.Linear(base_channels, num_obj)
        self.kp_heads   = nn.ModuleList([nn.Linear(base_channels, 3) for _ in range(num_joints)])
        self.embed_head = nn.Linear(base_channels, base_channels)

    def forward(self, coords, feats, sk, ip):
        rb_lvl1 = build_rulebook(coords, sk, ip, neighborhood_offsets)
        x1 = self.conv1(feats, rulebook=rb_lvl1, cond=None)

        # Downsample /2
        coarse = (coords.float() // 2).long()
        uc, invc = torch.unique(coarse, return_inverse=True, dim=0)
        _, skc, ipc = build_sorted_key_index(uc)
        rb_lvl2 = build_rulebook(uc, skc, ipc, neighborhood_offsets)
        x1_agg  = scatter_mean(x1, invc)

        x2 = self.conv2(x1_agg, rulebook=rb_lvl2, cond=None)

        # Fuse back
        cond   = self.cond_proj(x2)
        up     = self.up_proj(x2)[invc]
        cond_m = cond[invc]
        xr     = self.fuse_conv(x1 + up, rulebook=rb_lvl1, cond=cond_m)

        # Heads
        sem_logits = self.sem_head(xr)                 # (M, C)
        pooled     = xr.mean(0, keepdim=True)          # (1, F)
        cls_logits = self.cls_head(pooled).squeeze(0)  # (C,)
        embed      = F.normalize(self.embed_head(pooled), dim=-1).squeeze(0)

        # joints in normalized voxel space
        joints_raw = torch.stack([h(pooled).squeeze(0) for h in self.kp_heads], dim=0)  # (K,3)
        joints = 1.2 * torch.tanh(joints_raw)

        return sem_logits, cls_logits, embed, joints


# ─────────────────────────────────────────────────────────────────────────────
# Pre/post-processing (must mirror training)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_vox_coords(uc: torch.Tensor):
    ucf = uc.float()
    center = ucf.mean(0)
    r = (ucf - center).pow(2).sum(-1).sqrt()
    try:
        scale = torch.quantile(r, 0.90)
    except AttributeError:
        k = max(int(0.9 * r.numel()), 1)
        scale = r.kthvalue(k).values
    scale = scale.clamp_min(1.0)
    uc_norm = (ucf - center) / scale
    return uc_norm, center, scale


def human_components_from_sem(
    uc: torch.Tensor,
    sem_logits: torch.Tensor,
    human_class_idx: int,
    prob_thresh: float = 0.55,
    min_voxels: int = 30,
) -> List[torch.Tensor]:
    probs = F.softmax(sem_logits, dim=-1)[:, human_class_idx]
    keep = (probs >= prob_thresh).nonzero(as_tuple=False).flatten()
    if keep.numel() == 0:
        return []

    vox = uc[keep]  # (Mh,3)
    key_map: Dict[Tuple[int,int,int], int] = {tuple(int(x) for x in xyz.tolist()): i for i, xyz in enumerate(vox)}
    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    visited = torch.zeros((vox.size(0),), dtype=torch.bool)
    comps: List[torch.Tensor] = []

    for i in range(vox.size(0)):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp_idx = [i]
        while stack:
            cur = stack.pop()
            x,y,z = vox[cur].tolist()
            for dx,dy,dz in nbrs:
                nbr = (int(x+dx), int(y+dy), int(z+dz))
                j = key_map.get(nbr, None)
                if j is not None and not visited[j]:
                    visited[j] = True
                    stack.append(j)
                    comp_idx.append(j)
        comp_idx_t = keep[torch.tensor(comp_idx, dtype=torch.long)]
        if comp_idx_t.numel() >= min_voxels:
            comps.append(comp_idx_t)

    return comps


def coords_feats_for_indices(uc: torch.Tensor, idx: torch.Tensor, feat_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    uc_sub = uc[idx]
    _, sk_sub, ip_sub = build_sorted_key_index(uc_sub)
    ucn, center, scale = normalize_vox_coords(uc_sub)
    feats = torch.zeros((uc_sub.size(0), feat_dim), dtype=torch.float32)
    feats[:, :3] = ucn
    feats[:, 3]  = 1.0
    return uc_sub, sk_sub, ip_sub, feats, center, scale


def joints_norm_to_world(joints_norm: torch.Tensor, center_vox: torch.Tensor, scale_vox: torch.Tensor, voxel_size: float) -> torch.Tensor:
    j_vox = joints_norm * scale_vox + center_vox
    j_world = j_vox * voxel_size
    return j_world


# ─────────────────────────────────────────────────────────────────────────────
# Loader and main API
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_meta(checkpoint_path: str, device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckdir = os.path.dirname(os.path.abspath(checkpoint_path))
    meta_path = os.path.join(ckdir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_dim = int(meta.get("feat_dim", 32))
    num_joints = int(meta.get("num_joints", 6))
    classes = meta["classes"]
    num_cls = len(classes)

    model = SparseUNetMultiTask(
        base_channels=feat_dim,
        num_semantic=num_cls,
        num_obj=num_cls,
        num_joints=num_joints
    ).to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    return model, meta, device


@torch.no_grad()
def run_inference_on_points(
    points_world: torch.Tensor,
    checkpoint_path: str,
    human_class_name: str = "human",
    human_prob_thresh: float = 0.55,
    min_voxels_per_human: int = 30,
) -> Dict:
    model, meta, device = load_model_and_meta(checkpoint_path)
    voxel_size = float(meta["voxel_size"])
    feat_dim   = int(meta["feat_dim"])
    classes    = meta["classes"]
    K          = int(meta.get("num_joints", 6))

    if human_class_name not in classes:
        raise RuntimeError(f"'human' class not found in meta classes {classes}")
    human_idx = classes.index(human_class_name)

    uc, inv = voxelize(points_world, voxel_size=voxel_size)  # (M,3), (N,)
    _, sk, ip = build_sorted_key_index(uc)

    ucn, cen_all, scl_all = normalize_vox_coords(uc)
    feats = torch.zeros((uc.size(0), feat_dim), dtype=torch.float32, device=device)
    feats[:, :3] = ucn.to(device)
    feats[:, 3]  = 1.0

    uc_dev = uc.to(device)
    sk_dev = sk.to(device)
    ip_dev = ip.to(device)
    sem_logits, _, _, _ = model(uc_dev, feats, sk_dev, ip_dev)

    comps = human_components_from_sem(uc, sem_logits.cpu(), human_idx,
                                      prob_thresh=human_prob_thresh,
                                      min_voxels=min_voxels_per_human)

    detections = []
    for ci, comp_idx in enumerate(comps):
        uc_sub, sk_sub, ip_sub, feats_sub, c_sub, s_sub = coords_feats_for_indices(uc, comp_idx, feat_dim)

        uc_sub_d   = uc_sub.to(device)
        sk_sub_d   = sk_sub.to(device)
        ip_sub_d   = ip_sub.to(device)
        feats_sub_d= feats_sub.to(device)

        sem_sub, cls_logits, _, joints_norm = model(uc_sub_d, feats_sub_d, sk_sub_d, ip_sub_d)

        probs_h = F.softmax(sem_sub, dim=-1)[:, human_idx].mean().item()

        joints_world = joints_norm_to_world(joints_norm.cpu(), c_sub, s_sub, voxel_size)

        det = {
            "comp_index": int(ci),
            "num_voxels": int(uc_sub.size(0)),
            "center_world": (uc_sub.float().mean(0) * voxel_size).tolist(),
            "human_prob_mean": float(probs_h),
            "joints_world": joints_world,  # (K,3) torch.FloatTensor
            "pred_class_logits": cls_logits.cpu(),  # optional
        }
        detections.append(det)

    return {
        "detections": detections,
        "meta": meta,
        "voxel_size": voxel_size,
        "classes": classes,
        "points_used": int(uc.size(0)),
    }


def run_on_blender_object(
    obj_name: str,
    checkpoint_path: str,
    human_prob_thresh: float = 0.55,
    min_voxels_per_human: int = 30,
):
    if not _HAS_BPY:
        raise RuntimeError("Blender (bpy) not available; run this inside Blender or use run_inference_on_points.")
    pts = extract_points_from_object(bpy.data.objects[obj_name])  # (N,3) torch.float32 (WORLD)
    if pts.numel() == 0:
        print(f"[WARN] Object '{obj_name}' had 0 points.")
    return run_inference_on_points(
        pts, checkpoint_path,
        human_prob_thresh=human_prob_thresh,
        min_voxels_per_human=min_voxels_per_human,
    )


if __name__ == "__main__":
    ck = os.path.abspath("./checkpoints/epoch_200.pth")
    if _HAS_BPY and "ToF" in bpy.data.objects:
        res = run_on_blender_object("ToF", ck)
        print("[OK] Detections:", len(res["detections"]))
    else:
        print("Provide your own Nx3 points to run_inference_on_points(...)")
