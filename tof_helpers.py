# tof_helpers.py
# Utilities for Blender-based point extraction, catalog organization, voxelization,
# label aliasing (humans collapsed), and soft-label attachment.

from __future__ import annotations
import bpy
import torch
import numpy as np
import math
import re
from typing import Iterable, Dict, Optional, Tuple, List

from mathutils import Vector
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

# =============================================================================
# Public constants
# =============================================================================
TRAINING_ROOT = "TrainingSet"
GENERIC_TYPE = "__generic__"

# =============================================================================
# Class-name normalization (collapsing to broad labels)
# =============================================================================
_HUMAN_ALIASES = {
    "male","female","man","woman","person","people",
    "human","humans","humanoid","character","char",
    "actor","avatar","body","figure"
}

# --- add this new helper next to normalize_class ---
def normalize_profile(name: str) -> str:
    """
    Normalize a profile/type name WITHOUT collapsing male/female/etc. to 'human'.
    - lowercases
    - trims suffix numbers
    - condenses whitespace/underscores/hyphens
    """
    import re
    n = name.lower().strip()
    n = n.split('.', 1)[0]
    n = re.sub(r'[\s_-]+', ' ', n)
    n = re.sub(r'\s*\d+$', '', n)
    return n.strip()


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

# =============================================================================
# Mesh → world points
# =============================================================================
def extract_points_from_object(obj: bpy.types.Object) -> torch.Tensor:
    """
    Return world-space vertex positions for a mesh object as a FloatTensor (N,3).
    Uses evaluated depsgraph (applies modifiers).
    """
    deps = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(deps)
    me = eo.to_mesh()
    try:
        V = np.empty((len(me.vertices), 3), dtype=np.float32)
        me.vertices.foreach_get("co", V.ravel())   # local coords
        M = np.array(eo.matrix_world, dtype=np.float32)
        R = M[:3, :3]
        t = M[:3, 3]
        V = (V @ R.T) + t                           # world coords
    finally:
        eo.to_mesh_clear()
    if V.size == 0:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.from_numpy(V)

# Convenience: dataset-like lookup by (collection index, object index)
def get_instance_points(collection_name: str,
                        x: int,
                        y: int,
                        transform=None) -> torch.Tensor:
    root = bpy.data.collections.get(collection_name)
    if root is None:
        raise RuntimeError(f"Collection '{collection_name}' not found")
    subs = list(root.children)
    if x < 0 or x >= len(subs):
        raise IndexError(f"X index {x} out of range (got {len(subs)} sub-collections)")
    subcol = subs[x]
    objs = [o for o in subcol.objects if o.type == "MESH"]
    if y < 0 or y >= len(objs):
        raise IndexError(f"Y index {y} out of range (got {len(objs)} mesh objects in '{subcol.name}')")
    obj = objs[y]
    pts = extract_points_from_object(obj)
    if (transform is not None) and pts.numel() > 0:
        pts = transform(pts)
    return pts

# =============================================================================
# Catalog indexing: top-level = label, children = types/profiles, direct meshes = __generic__
# =============================================================================
def build_dataset_indices(collection_name: str = TRAINING_ROOT):
    """
    Scan:
      TrainingSet/
        <LabelA>/               -> label = normalize_class(<LabelA>)
          (meshes...)           -> type '__generic__'
          <Type1>/ (meshes...)  -> type = normalize_class(<Type1>)
          <Type2>/ ...
        <LabelB>/ ...
    Returns list of tuples: (x_index, y_index, object_name, label)
    (The x/y indices are best-effort and only for quick navigation; not strictly required.)
    """
    root = bpy.data.collections.get(collection_name)
    if root is None:
        raise RuntimeError(f"Collection '{collection_name}' not found")

    indices: List[Tuple[int,int,str,str]] = []
    class_cols = list(root.children)

    for x, top in enumerate(class_cols):
        label = normalize_class(top.name)

        # meshes directly under label → generic type
        y_counter = 0
        for o in top.objects:
            if o.type != "MESH":
                continue
            indices.append((x, y_counter, o.name, label))
            y_counter += 1

        # sub-collections as types
        for sub in top.children:
            for o in sub.objects:
                if o.type != "MESH": continue
                indices.append((x, y_counter, o.name, label))
                y_counter += 1

    # small summary
    from collections import Counter
    cls_counts = Counter([lab for *_xy, lab in indices])
    print(f"[Catalog] Indexed {len(indices)} items. Class distribution: {dict(cls_counts)}")
    return indices

def scan_training_catalog_profiles(root_name: str):
    """
    Returns:
      class_names: sorted list of labels
      types_map:   {label: [type names including '__generic__', ...]}
      samples:     [{"class": label, "type": type_name, "object_name": mesh_name}, ...]
    """
    root = bpy.data.collections.get(root_name)
    if root is None:
        raise RuntimeError(f"Collection '{root_name}' not found")

    types_map: Dict[str, List[str]] = {}
    samples: List[Dict] = []

    for top in root.children:
        # KEEP collapsing for the class (top-level)
        label = normalize_class(top.name)
        types_map.setdefault(label, [])

        # meshes directly under label → generic type
        generic_meshes = [o for o in top.objects if o.type == "MESH"]
        if generic_meshes:
            if GENERIC_TYPE not in types_map[label]:
                types_map[label].append(GENERIC_TYPE)
            for o in generic_meshes:
                samples.append({"class": label, "type": GENERIC_TYPE, "object_name": o.name})

        # sub-collections as types: **DO NOT collapse male/female to 'human'**
        for sub in top.children:
            tname = normalize_profile(sub.name)   # <-- was normalize_class(...)
            if tname not in types_map[label]:
                types_map[label].append(tname)
            for o in sub.objects:
                if o.type == "MESH":
                    samples.append({"class": label, "type": tname, "object_name": o.name})

    class_names = sorted(types_map.keys())
    for k in list(types_map.keys()):
        types_map[k] = sorted(types_map[k])
    samples = sorted(samples, key=lambda d: (d["class"], d["type"], d["object_name"]))
    return class_names, types_map, samples


# =============================================================================
# Voxel / hashing utilities
# =============================================================================
def voxelize(points: torch.Tensor, voxel_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize points into integer voxel coords. Returns (unique_coords, inverse_index).
    """
    assert points.dim() == 2 and points.size(-1) == 3
    vcoords = torch.floor(points / float(voxel_size)).to(torch.int64)
    unique_coords, inverse = torch.unique(vcoords, return_inverse=True, dim=0)
    return unique_coords, inverse

def build_sorted_key_index(voxel_coords: torch.Tensor):
    """
    Build Morton-like mixed key for fast neighborhood lookups via searchsorted.
    Returns (keys, sorted_keys, inv_perm).
    """
    assert voxel_coords.dim() == 2 and voxel_coords.size(-1) == 3
    x, y, z = voxel_coords.to(torch.int64).unbind(-1)
    keys = ((x << 40) ^ (y << 20) ^ z).long()
    sorted_keys, perm = torch.sort(keys)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return keys, sorted_keys, inv_perm

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Simple scatter-mean along 'dim' using integer 'index'.
    """
    if dim != 0:
        src = src.transpose(0, dim)
    N = src.size(0)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    expanded = index.unsqueeze(-1).expand(N, src.size(1))
    sum_t = torch.zeros((dim_size, src.size(1)), device=src.device, dtype=src.dtype)
    sum_t = sum_t.scatter_add_(0, expanded, src)
    counts = torch.zeros((dim_size,), device=src.device, dtype=src.dtype)
    counts = counts.scatter_add_(0, index, torch.ones((N,), device=src.device, dtype=src.dtype))
    counts = counts.clamp(min=1).unsqueeze(-1)
    mean = sum_t / counts
    if dim != 0:
        mean = mean.transpose(0, dim)
    return mean

# =============================================================================
# Neighborhood offsets & gather
# =============================================================================
def make_kernel_offsets(kernel_size: int, dilation: int = 1) -> torch.Tensor:
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    r = kernel_size // 2
    offs = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                offs.append((dx * dilation, dy * dilation, dz * dilation))
    return torch.tensor(offs, dtype=torch.int64)

neighborhood_offsets = make_kernel_offsets(3, dilation=1)

def gather_neighbors_tensorized(voxel_coords: torch.Tensor,
                                sorted_keys: torch.Tensor,
                                neighborhood_offsets: torch.Tensor) -> torch.Tensor:
    """
    For each voxel, find indices of its K neighbors (K = offsets).
    Returns (M,K) int64 tensor of indices into sorted_keys; -1 for missing.
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
    neighbor_keys = ((x_n.to(torch.int64) << 40) ^
                     (y_n.to(torch.int64) << 20) ^
                     z_n.to(torch.int64)).long()                                # (M,K)

    candidate_idx = torch.searchsorted(sorted_keys, neighbor_keys)
    candidate_idx = torch.clamp(candidate_idx, 0, sorted_keys.size(0) - 1)
    matched = (sorted_keys[candidate_idx] == neighbor_keys)
    neighbor_idx = torch.where(matched, candidate_idx, torch.full_like(candidate_idx, -1))
    return neighbor_idx

# =============================================================================
# Vertex-Group binding (attach ToF points to rig skin groups)
# =============================================================================
_REGION_SYNONYMS = {
    "shoulder": ("shoulder", "shldr", "clav", "clavicle", "collar"),
    "forearm":  ("forearm", "lowerarm", "lower_arm", "ulna", "radius"),
    "elbow":    ("elbow", "elb"),
    "hand":     ("hand", "palm"),
    "wrist":    ("wrist",),
    "hip":      ("hip", "upperleg", "upper_leg", "thigh", "glute"),
    "pelvis":   ("pelvis", "pelv", "hips", "root"),
    "shin":     ("shin", "calf", "lowerleg", "lower_leg", "shank", "tibia"),
    "knee":     ("knee", "kne"),
    "foot":     ("foot", "feet"),
    "ankle":    ("ankle", "ankl"),
}
_SIDE_PATTERNS = (
    (re.compile(r'(^|[^a-z])l(eft)?([^a-z]|$)'), "l"),
    (re.compile(r'(^|[^a-z])r(ight)?([^a-z]|$)'), "r"),
    (re.compile(r'\.l$'), "l"),
    (re.compile(r'\.r$'), "r"),
    (re.compile(r'_l$'), "l"),
    (re.compile(r'_r$'), "r"),
)
_ALIAS_CACHE: Dict[tuple, Dict[str, str]] = {}

def _norm_text(s: str) -> str:
    s = s.lower().replace(".", "_")
    s = re.sub(r'[\s\-]+', "_", s)
    return s

def _infer_side(name_norm: str) -> Optional[str]:
    for pat, side in _SIDE_PATTERNS:
        if pat.search(name_norm):
            return side
    if name_norm.endswith("left") or name_norm.startswith("left_"):
        return "l"
    if name_norm.endswith("right") or name_norm.startswith("right_"):
        return "r"
    return None

def _infer_region(name_norm: str) -> Optional[str]:
    for region, keys in _REGION_SYNONYMS.items():
        for k in keys:
            if k and k in name_norm:
                return region
    return None

def _build_aliases_for_names(source_names: Iterable[str], target_groups: Iterable[str]) -> Dict[str, str]:
    tset = set(target_groups); aliases: Dict[str, str] = {}
    for raw in source_names:
        nm  = _norm_text(raw)
        side = _infer_side(nm)
        region = _infer_region(nm)
        canon = None
        if region == "pelvis":
            if "pelvis" in tset:
                canon = "pelvis"
        else:
            if (side is not None) and (region is not None):
                candidate = f"{side}_{region}"
                if candidate in tset:
                    canon = candidate
                else:
                    # a few forgiving swaps
                    if region == "hand" and f"{side}_wrist" in tset:
                        canon = f"{side}_wrist"
                    elif region == "wrist" and f"{side}_hand" in tset:
                        canon = f"{side}_hand"
                    elif region == "hip" and f"{side}_thigh" in tset:
                        canon = f"{side}_thigh"
        if canon is not None:
            aliases[raw] = canon
    return aliases

def build_aliases_for_skin(skin_obj_name: str, target_groups: Iterable[str]) -> Dict[str, str]:
    key = ("vg", skin_obj_name, tuple(sorted(target_groups)))
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    skin = bpy.data.objects.get(skin_obj_name)
    if skin is None:
        _ALIAS_CACHE[key] = {}
        return {}
    names = [vg.name for vg in skin.vertex_groups]
    aliases = _build_aliases_for_names(names, target_groups)
    _ALIAS_CACHE[key] = aliases
    if aliases:
        print(f"[VG-Alias] {skin_obj_name}: mapped {len(aliases)} groups.")
    return aliases

# ---- SkinBinding ------------------------------------------------------------
class SkinBinding:
    def __init__(self,
                 skin_obj_name: str,
                 target_groups: Iterable[str],
                 aliases: Optional[Dict[str, str]] = None):
        self.skin_obj_name = skin_obj_name
        self.target_groups = list(target_groups)
        self.aliases = dict(aliases or {})
        self.canonical_to_id = {n: i for i, n in enumerate(self.target_groups)}
        self.id_to_canonical = self.target_groups
        self._build()

    def _build(self):
        deps = bpy.context.evaluated_depsgraph_get()
        skin_obj = bpy.data.objects[self.skin_obj_name]
        skin_eval = skin_obj.evaluated_get(deps)
        me_eval = skin_eval.data
        V = np.empty((len(me_eval.vertices), 3), dtype=np.float64)
        me_eval.vertices.foreach_get("co", V.ravel())
        M = np.array(skin_eval.matrix_world, dtype=np.float64)
        R = M[:3, :3]; t = M[:3, 3]
        self.V_world = V @ R.T + t

        vg_index_to_gid: Dict[int, Optional[int]] = {}
        for vg in skin_obj.vertex_groups:
            name = self.aliases.get(vg.name, vg.name)
            vg_index_to_gid[vg.index] = self.canonical_to_id.get(name, None)

        me_orig = skin_obj.data
        v_groups: List[List[Tuple[int, float]]] = [[] for _ in range(len(me_orig.vertices))]
        for v in me_orig.vertices:
            acc = []
            for g in v.groups:
                gid = vg_index_to_gid.get(g.group, None)
                if gid is not None and g.weight > 0.0:
                    acc.append((gid, float(g.weight)))
            if acc:
                s = sum(w for _, w in acc)
                v_groups[v.index] = [(gid, w / s) for gid, w in acc]
        self.v_groups = v_groups

        kd = KDTree(len(self.V_world))
        for i, p in enumerate(self.V_world):
            kd.insert(Vector((float(p[0]), float(p[1]), float(p[2]))), i)
        kd.balance()
        self.kd = kd

    def attach_points(self,
                      points_world: np.ndarray,
                      k: int = 8,
                      eps: float = 1e-6,
                      hard: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        N = int(points_world.shape[0])
        B = len(self.target_groups)
        soft = np.zeros((N, B), dtype=np.float32)
        for i in range(N):
            p = Vector((float(points_world[i, 0]), float(points_world[i, 1]), float(points_world[i, 2])))
            hits = self.kd.find_n(p, k)
            for _, vidx, dist in hits:
                lst = self.v_groups[vidx]
                if not lst:
                    continue
                w_spatial = 1.0 / (float(dist) + eps)
                for gid, wvg in lst:
                    soft[i, gid] += w_spatial * float(wvg)
            s = soft[i].sum()
            if s > 0:
                soft[i] /= s
        labels = soft.argmax(axis=1).astype(np.int32)
        if hard:
            soft[:] = 0.0
            soft[np.arange(N), labels] = 1.0
        return soft, labels

_SB_CACHE: Dict[Tuple[str, Tuple[str, ...]], SkinBinding] = {}

def get_skin_binding(skin_obj_name: str,
                     target_groups: Iterable[str],
                     aliases: Optional[Dict[str, str]] = None) -> SkinBinding:
    key = (skin_obj_name, tuple(target_groups))
    if key in _SB_CACHE:
        return _SB_CACHE[key]
    sb = SkinBinding(skin_obj_name, target_groups, aliases)
    _SB_CACHE[key] = sb
    return sb

def invalidate_skin_binding(skin_obj_name: str, target_groups: Iterable[str]) -> None:
    key = (skin_obj_name, tuple(target_groups))
    _SB_CACHE.pop(key, None)

def attach_points_to_skin(points_world_t: torch.Tensor,
                          skin_obj_name: str,
                          target_groups: Iterable[str],
                          aliases: Optional[Dict[str, str]] = None,
                          k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    assert points_world_t.dim() == 2 and points_world_t.size(-1) == 3
    sb = get_skin_binding(skin_obj_name, target_groups, aliases)
    P = points_world_t.detach().cpu().numpy()
    soft, labels = sb.attach_points(P, k=k)
    return torch.from_numpy(soft), torch.from_numpy(labels).long()

def get_group_vertex_positions(skin_obj_name: str,
                               target_groups: Iterable[str],
                               aliases: Optional[Dict[str, str]] = None,
                               weight_threshold: float = 0.0) -> Dict[str, np.ndarray]:
    sb = get_skin_binding(skin_obj_name, target_groups, aliases)
    G = {name: [] for name in target_groups}
    for vidx, lst in enumerate(sb.v_groups):
        if not lst: continue
        p = sb.V_world[vidx]
        for gid, w in lst:
            if w > weight_threshold:
                name = sb.id_to_canonical[gid]
                G[name].append(p)
    return {k: (np.array(v, dtype=np.float32) if v else np.empty((0, 3), dtype=np.float32))
            for k, v in G.items()}

# =============================================================================
# Material attachment utilities
# =============================================================================
def _get_eval_object_and_mesh(obj_name: str):
    deps = bpy.context.evaluated_depsgraph_get()
    obj = bpy.data.objects[obj_name]
    obj_e = obj.evaluated_get(deps)
    me_e = obj_e.to_mesh()
    return obj_e, me_e, deps

def build_aliases_for_materials(obj_name: str, target_groups: Iterable[str]) -> Dict[str, str]:
    key = ("mat", obj_name, tuple(sorted(target_groups)))
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        _ALIAS_CACHE[key] = {}
        return {}
    names = [slot.name for slot in obj.material_slots]
    aliases = _build_aliases_for_names(names, target_groups)
    _ALIAS_CACHE[key] = aliases
    if aliases:
        print(f"[MAT-Alias] {obj_name}: mapped {len(aliases)} materials.")
    return aliases

def attach_points_to_materials(points_world_t: torch.Tensor,
                               obj_name: str,
                               target_groups: Iterable[str],
                               aliases: Optional[Dict[str, str]] = None,
                               k: int = 1,
                               radius: Optional[float] = None,
                               eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map points to material-based groups (by nearest polygon / neighborhood).
    """
    assert points_world_t.dim() == 2 and points_world_t.size(-1) == 3
    if obj_name not in bpy.data.objects:
        raise RuntimeError(f"Object '{obj_name}' not found for material mapping.")
    obj, me, deps = _get_eval_object_and_mesh(obj_name)
    try:
        tree = BVHTree.FromObject(obj, deps, epsilon=0.0)
        B = len(list(target_groups))
        canon_to_idx = {n: i for i, n in enumerate(target_groups)}

        # map material slots to canonical group names
        mat_index_to_canon: Dict[int, Optional[str]] = {}
        slots = list(obj.material_slots)
        for midx, slot in enumerate(slots):
            raw = slot.name
            canon = aliases.get(raw, raw) if aliases else raw
            mat_index_to_canon[midx] = canon if canon in canon_to_idx else None

        # polygon → gid
        poly_to_gid = np.full((len(me.polygons),), -1, dtype=np.int32)
        for pi, poly in enumerate(me.polygons):
            canon = mat_index_to_canon.get(poly.material_index, None)
            if canon is not None:
                poly_to_gid[pi] = canon_to_idx[canon]

        P = points_world_t.detach().cpu().numpy()
        N = P.shape[0]
        soft = np.zeros((N, B), dtype=np.float32)

        if k == 1:
            for i in range(N):
                loc, normal, index, dist = tree.find_nearest(Vector(P[i]))
                if index is None:
                    continue
                gid = int(poly_to_gid[index])
                if gid >= 0:
                    soft[i, gid] = 1.0
        else:
            if radius is None:
                bb = np.array(obj.bound_box, dtype=np.float32)
                diag = np.linalg.norm(bb.max(axis=0) - bb.min(axis=0))
                radius = max(1e-6, 0.01 * diag)
            for i in range(N):
                hits = tree.find_nearest_range(Vector(P[i]), radius)
                if not hits:
                    continue
                acc: Dict[int, float] = {}
                for loc, normal, index, dist in hits:
                    if index is None:
                        continue
                    gid = int(poly_to_gid[index])
                    if gid < 0:
                        continue
                    acc[gid] = acc.get(gid, 0.0) + 1.0 / (float(dist) + eps)
                if acc:
                    s = sum(acc.values())
                    for gid, w in acc.items():
                        soft[i, gid] = w / s

        labels = soft.argmax(axis=1).astype(np.int32)
        return torch.from_numpy(soft), torch.from_numpy(labels).long()
    finally:
        obj.to_mesh_clear()

def extract_material_vertex_positions(obj_name: str,
                                      aliases: Optional[Dict[str, str]] = None,
                                      target_groups: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
    """
    Collect vertex positions per material group (canonicalized).
    """
    if obj_name not in bpy.data.objects:
        return {}
    obj, me, _ = _get_eval_object_and_mesh(obj_name)
    try:
        V = np.empty((len(me.vertices), 3), dtype=np.float32)
        me.vertices.foreach_get("co", V.ravel())
        slots = list(obj.material_slots)
        midx_to_name = {i: slots[i].name for i in range(len(slots))}
        mats: Dict[str, set] = {}
        for poly in me.polygons:
            raw = midx_to_name.get(poly.material_index, "NoMat")
            name = aliases.get(raw, raw) if aliases else raw
            if (target_groups is not None) and (name not in target_groups):
                continue
            s = mats.setdefault(name, set())
            for v in poly.vertices:
                s.add(int(v))
        out = {k: (V[list(idxs)] if idxs else np.empty((0, 3), dtype=np.float32))
               for k, idxs in mats.items()}
        return out
    finally:
        obj.to_mesh_clear()

# =============================================================================
# Aggregate point-wise soft labels → voxels
# =============================================================================
def aggregate_soft_to_voxels(points_world_t: torch.Tensor,
                             soft_labels_t: torch.Tensor,
                             voxel_size: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Average N×B soft labels into V×B voxel slots.
    Returns (unique_voxel_coords[V,3], inverse_index[N], voxel_soft[V,B]).
    """
    uc, inv = voxelize(points_world_t, voxel_size)
    V = uc.size(0)
    B = soft_labels_t.size(1)

    soft = torch.zeros((V, B), dtype=torch.float32, device=soft_labels_t.device)
    soft.index_add_(0, inv, soft_labels_t)

    counts = torch.zeros((V, 1), dtype=torch.float32, device=soft_labels_t.device)
    counts.scatter_add_(0, inv, torch.ones((soft_labels_t.size(0), 1), dtype=torch.float32, device=soft_labels_t.device))
    counts = counts.clamp(min=1.0)

    soft = soft / counts
    return uc.cpu(), inv.cpu(), soft.cpu()


# Collapses many human-ish names to a single class label "human"
def normalize_class(name: str) -> str:
    """
    Normalize a top-level class name into a broad label.
    - lowercases
    - trims suffix numbers
    - condenses whitespace/underscores/hyphens
    - collapses human-ish aliases to 'human'
    """
    import re
    n = name.lower().strip()
    n = n.split('.', 1)[0]
    n = re.sub(r'[\s_-]+', ' ', n)
    n = re.sub(r'\s*\d+$', '', n)
    n = n.strip()
    if n in _HUMAN_ALIASES:
        return "human"
    return n

# =============================================================================
# __all__
# =============================================================================
__all__ = [
    "TRAINING_ROOT", "GENERIC_TYPE",
    "normalize_class", "normalize_profile",
    "extract_points_from_object", "get_instance_points",
    "build_dataset_indices", "scan_training_catalog_profiles",
    "random_rotation",
    # voxel/hash
    "voxelize", "build_sorted_key_index", "scatter_mean",
    "make_kernel_offsets", "neighborhood_offsets", "gather_neighbors_tensorized",
    # vertex groups
    "SkinBinding", "build_aliases_for_skin", "get_skin_binding",
    "invalidate_skin_binding", "attach_points_to_skin", "get_group_vertex_positions",
    # materials
    "build_aliases_for_materials", "attach_points_to_materials",
    "extract_material_vertex_positions",
    # aggregation
    "aggregate_soft_to_voxels",
]
