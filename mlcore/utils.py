# mlcore/utils.py
import json
import torch

# ---------- voxel & neighbor utilities ----------

def voxelize(points: torch.Tensor, voxel_size: float):
    vcoords = torch.floor(points / voxel_size).to(torch.int64)
    unique_coords, inverse = torch.unique(vcoords, return_inverse=True, dim=0)
    return unique_coords, inverse  # (M,3), (N,)

def build_sorted_key_index(voxel_coords: torch.Tensor):
    x, y, z = voxel_coords.unbind(-1)
    keys = ((x << 40) ^ (y << 20) ^ z).long()
    sorted_keys, perm = torch.sort(keys)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=perm.device)
    return keys, sorted_keys, inv_perm  # (M,), (M,), (M,)

def gather_neighbors_tensorized(voxel_coords: torch.Tensor,
                                sorted_keys: torch.Tensor,
                                neighborhood_offsets: torch.Tensor):
    """
    Returns indices into sorted_keys for each neighbor (or -1 if missing).
    """
    device = voxel_coords.device
    if sorted_keys.device != device:
        sorted_keys = sorted_keys.to(device)
    if neighborhood_offsets.device != device:
        neighborhood_offsets = neighborhood_offsets.to(device)

    M = voxel_coords.size(0)
    K = neighborhood_offsets.size(0)
    expanded = voxel_coords.unsqueeze(1) + neighborhood_offsets.unsqueeze(0)  # (M,K,3)
    x_n, y_n, z_n = expanded.unbind(-1)
    neighbor_keys = ((x_n << 40) ^ (y_n << 20) ^ z_n).long()                  # (M,K)

    candidate_idx = torch.searchsorted(sorted_keys, neighbor_keys)
    candidate_idx = torch.clamp(candidate_idx, 0, sorted_keys.size(0) - 1)
    matched = sorted_keys[candidate_idx] == neighbor_keys
    neighbor_idx = torch.where(matched, candidate_idx, torch.full_like(candidate_idx, -1))
    return neighbor_idx  # (M,K)

# ---------- meta helpers ----------

def save_meta(path: str, meta: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
