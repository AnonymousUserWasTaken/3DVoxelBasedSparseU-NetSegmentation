# - Collect per-material vertex positions from polygons (obj.material_slots + me.polygons)
# - space='OBJECT' uses raw v.co; space='WORLD' applies obj.matrix_world
# - Joints for supervision = centroids of specific material groups
#   (e.g., Shoulders, Forearms, Hands, Hips, Legs, Feet) chosen by exact
#   material base-name (no fuzzy aliasing).
# - Also exposes exact/voxel point→material mappers.
#
# STRICT MODE: extract_human_joints_world will RAISE if any requested joint’s
# material group is missing or has too few vertices (no silent fallbacks).

from typing import Dict, List, Tuple, Optional
import re

import bpy
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _material_base(name: str) -> str:
    """Strip trailing .### from a material name (keep original casing)."""
    return re.sub(r"\.\d+$", "", str(name).strip())

def _material_base_norm(name: str) -> str:
    """Case/space-insensitive normalized base name for comparisons."""
    base = _material_base(name)
    return re.sub(r"[\s_]+", "", base).lower()

def _get_object_and_mesh(obj_name: str):
    if obj_name not in bpy.data.objects:
        raise RuntimeError(f"Object '{obj_name}' not found in scene.")
    obj = bpy.data.objects[obj_name]
    me = obj.data
    if me is None or not hasattr(me, "polygons"):
        raise RuntimeError(f"Object '{obj_name}' has no mesh data.")
    return obj, me

def _vertex_positions(me: bpy.types.Mesh, space: str, obj: bpy.types.Object) -> np.ndarray:
    """
    Pull all vertex coordinates from me.vertices into a (V,3) float32 array.
    space='OBJECT' -> raw v.co
    space='WORLD'  -> obj.matrix_world @ v.co
    """
    V = np.empty((len(me.vertices), 3), dtype=np.float32)
    me.vertices.foreach_get("co", V.ravel())
    if space.upper() == "WORLD":
        M = np.array(obj.matrix_world, dtype=np.float32)
        R = M[:3, :3]; t = M[:3, 3]
        V = (V @ R.T) + t
    return V

# ─────────────────────────────────────────────────────────────────────────────
# Materials → vertex positions / centroids
# ─────────────────────────────────────────────────────────────────────────────

def collect_material_vertices(
    obj_name: str,
    space: str = "WORLD",
    dedup: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Return { material_base -> (Nm,3) float32 positions } in requested space,
    using obj.data polygons + obj.material_slots. No evaluated.to_mesh is used.
    Raises if the object has no materials or no vertices.
    """
    obj, me = _get_object_and_mesh(obj_name)

    # Material names by slot (object slots preferred; fallback to me.materials)
    slot_list = (
        [s.material for s in obj.material_slots] if len(obj.material_slots) > 0 else list(me.materials)
    )
    if len(slot_list) == 0:
        raise RuntimeError(f"[GT ERROR] '{obj_name}' has no material slots/materials.")

    slot_names = [
        _material_base(m.name) if m is not None else f"__none_{i}"
        for i, m in enumerate(slot_list)
    ]

    # All vertex positions (OBJECT or WORLD)
    Vpos = _vertex_positions(me, space, obj)  # (V,3) float32
    if Vpos.size == 0:
        raise RuntimeError(f"[GT ERROR] '{obj_name}' has no vertices.")

    # Build vertex-index sets per material from polygons
    mat_to_vidxs: Dict[str, set] = {}
    for poly in me.polygons:
        mi = int(poly.material_index) if hasattr(poly, "material_index") else 0
        base = slot_names[mi if 0 <= mi < len(slot_names) else 0]
        s = mat_to_vidxs.get(base)
        if s is None:
            s = set()
            mat_to_vidxs[base] = s
        for vid in poly.vertices:
            s.add(int(vid))

    if not mat_to_vidxs:
        raise RuntimeError(f"[GT ERROR] '{obj_name}' polygons did not reference any material slots.")

    # Assemble tensors
    out: Dict[str, torch.Tensor] = {}
    for base, vidxs in mat_to_vidxs.items():
        ids = np.fromiter(vidxs, dtype=np.int64) if dedup else np.array(list(vidxs), dtype=np.int64)
        if ids.size == 0:
            # keep empty groups out entirely
            continue
        out[base] = torch.from_numpy(Vpos[ids].copy())  # (Nm,3) float32

    if not out:
        raise RuntimeError(f"[GT ERROR] '{obj_name}' produced no per-material vertex sets.")
    return out

def collect_material_centroids(
    obj_name: str,
    min_verts: int = 1,
    space: str = "WORLD",
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Compute per-material centroids.
    Returns:
      names: list[str] material base names (sorted)
      J:     (K,3) float32 positions in requested space
      mask:  (K,)  bool (True if that material has >= min_verts verts)
    Raises if no materials/vertices are found.
    """
    mats = collect_material_vertices(obj_name, space=space)
    if not mats:
        raise RuntimeError(f"[GT ERROR] No materials/vertices found on '{obj_name}'")

    names = sorted(mats.keys())
    K = len(names)
    J = torch.empty((K, 3), dtype=torch.float32)
    mask = torch.zeros((K,), dtype=torch.bool)
    for i, n in enumerate(names):
        pts = mats[n]
        if pts.shape[0] >= max(1, int(min_verts)):
            J[i] = pts.mean(dim=0)
            mask[i] = True
        else:
            # still set a value; mask will mark it invalid
            J[i] = pts.mean(dim=0) if pts.shape[0] > 0 else torch.zeros(3, dtype=torch.float32)

    return names, J, mask

# ─────────────────────────────────────────────────────────────────────────────
# Point → Material mapping (optional utilities)
# ─────────────────────────────────────────────────────────────────────────────

def map_points_to_materials_exact(
    points: torch.Tensor,
    obj_name: str,
    space: str = "WORLD",
    decimals: int = 6,
) -> Tuple[torch.Tensor, List[str], List[List[int]]]:
    """
    Exact (rounded) position match:
      - Build {rounded vertex position -> set(material ids)} from mesh
      - For each input point, round and look up membership
    Returns:
      labels: (N,) int64 (first material id or -1 if none)
      names:  list[str] materials in index order
      multi:  list of lists of material ids per point (may be multiple)
    Raises if the object/material data is missing.
    """
    assert points.dim() == 2 and points.size(-1) == 3
    mats = collect_material_vertices(obj_name, space=space)
    names = sorted(mats.keys())
    idx_of = {n: i for i, n in enumerate(names)}

    # Build position -> materials mapping
    pos2mats: Dict[tuple, set] = {}
    for n in names:
        mid = idx_of[n]
        P = mats[n].numpy()
        P = np.round(P, decimals=decimals)
        for p in map(tuple, P):
            s = pos2mats.get(p)
            if s is None:
                s = set()
                pos2mats[p] = s
            s.add(mid)

    # Query
    pts = np.round(points.detach().cpu().numpy(), decimals=decimals)
    N = pts.shape[0]
    labels = np.full((N,), -1, dtype=np.int64)
    multi: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        s = pos2mats.get(tuple(pts[i]))
        if s:
            ids = sorted(s)
            labels[i] = ids[0]
            multi[i] = ids

    return torch.from_numpy(labels), names, multi

def map_points_to_materials_voxel(
    points: torch.Tensor,
    obj_name: str,
    voxel_size: float,
    space: str = "WORLD",
) -> Tuple[torch.Tensor, List[str]]:
    """
    Voxel-overlap mapping:
      - floor(points/voxel_size) and floor(material_verts/voxel_size)
      - point label = any material occupying same voxel (ties -> last material wins)
    Returns:
      labels: (N,) int64 in [0..K-1] or -1 if empty
      names:  list[str] materials in index order
    Raises if the object/material data is missing.
    """
    assert points.dim() == 2 and points.size(-1) == 3
    mats = collect_material_vertices(obj_name, space=space)
    names = sorted(mats.keys())
    idx_of = {n: i for i, n in enumerate(names)}

    def _pack(v: np.ndarray) -> np.ndarray:
        x = v[:, 0].astype(np.int64)
        y = v[:, 1].astype(np.int64)
        z = v[:, 2].astype(np.int64)
        return ((x << 40) ^ (y << 20) ^ z).astype(np.int64)

    P = points.detach().cpu().numpy().astype(np.float64)
    Pvox = np.floor(P / float(voxel_size)).astype(np.int64)
    Pkey = _pack(Pvox)

    key2mat: Dict[int, int] = {}
    for n in names:
        mid = idx_of[n]
        V = mats[n].detach().cpu().numpy().astype(np.float64)
        if V.shape[0] == 0:
            continue
        Vvox = np.floor(V / float(voxel_size)).astype(np.int64)
        for k in np.unique(_pack(Vvox)):
            key2mat[int(k)] = mid

    labels = np.full((Pkey.shape[0],), -1, dtype=np.int64)
    for i, k in enumerate(Pkey):
        labels[i] = key2mat.get(int(k), -1)

    return torch.from_numpy(labels), names

# ─────────────────────────────────────────────────────────────────────────────
# Joints adapter: pick six anatomical proxies by exact material names.
# No fuzzy aliasing — change the mapping strings if your materials differ.
# STRICT: Raises if any required material is missing or invalid.
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_JOINT_TO_MAT = {
    "shoulder": "Shoulders",
    "elbow":    "Forearms",
    "wrist":    "Hands",
    "hip":      "Hips",
    "knee":     "Legs",
    "ankle":    "Feet",
}

def extract_human_joints_world(
    obj_name: str,
    joint_names: List[str],
    joint_to_material: Optional[Dict[str, str]] = None,
    min_verts: int = 10,
    space: str = "WORLD",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (K,3) joint targets from per-material centroids using *exact* material names.

    STRICT BEHAVIOR:
      - If ANY requested joint’s material is missing or has < min_verts verts,
        raise RuntimeError (no silent fallbacks).

    Returns:
      J:    (K,3) float32 in WORLD/OBJECT space
      mask: (K,)  bool (all True under strict behavior if no exception)
    """
    names, Jm, M = collect_material_centroids(obj_name, min_verts=min_verts, space=space)
    names_norm = [_material_base_norm(n) for n in names]
    name_to_idx = {names_norm[i]: i for i in range(len(names))}

    mapping = dict(_DEFAULT_JOINT_TO_MAT)
    if joint_to_material:
        mapping.update(joint_to_material)

    K = len(joint_names)
    J = torch.empty((K, 3), dtype=torch.float32)
    mask = torch.zeros((K,), dtype=torch.bool)

    missing: List[str] = []
    for k, jname in enumerate(joint_names):
        want_mat = mapping.get(jname.lower(), jname)  # default to same name if not mapped
        idx = name_to_idx.get(_material_base_norm(want_mat), None)
        if idx is not None and bool(M[idx].item()):
            J[k] = Jm[idx]
            mask[k] = True
        else:
            missing.append(f"{jname}->{want_mat}")

    if missing:
        raise RuntimeError(
            f"[GT ERROR] On object '{obj_name}', missing/invalid vertex groups for joints: "
            + ", ".join(missing)
            + f" (min_verts={min_verts}, space={space})."
        )

    return J, mask