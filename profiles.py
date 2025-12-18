# profiles.py
from __future__ import annotations
"""
Profiles utilities

Goal:
- Build a GLOBAL profile id space across ALL classes.
- Keep helpers to (de)serialize profiles, validate catalog coverage,
  and produce per-class legality masks.

Design:
- A "profile" is the pair (class, type). We encode it as "class::type".
- prof2id is contiguous 0..P-1 across *all* classes.
- allowed_per_class maps a class_id to the set of legal profile ids.
"""

from typing import Dict, List, Set, Tuple, Optional

PROFILE_KEY_SEPARATOR = "::"

# -----------------------------------------------------------------------------#
# Key helpers
# -----------------------------------------------------------------------------#
def make_prof_key(cname: str, tname: str) -> str:
    """Create canonical profile key 'class::type'."""
    return f"{cname}{PROFILE_KEY_SEPARATOR}{tname}"

def parse_prof_key(key: str) -> Tuple[str, str]:
    """Inverse of make_prof_key: 'class::type' -> (class, type)."""
    if PROFILE_KEY_SEPARATOR not in key:
        # Be lenient: treat whole string as a class and type='__generic__'
        return key, "__generic__"
    c, t = key.split(PROFILE_KEY_SEPARATOR, 1)
    return c, t

# -----------------------------------------------------------------------------#
# Builders
# -----------------------------------------------------------------------------#
def build_global_profiles(class_names: List[str],
                          types_map: Dict[str, List[str]]
                         ) -> Tuple[Dict[str,int], Dict[int, Tuple[str,str]], Dict[str, List[int]]]:
    """
    Build a global, contiguous profile id space across ALL classes.

    Args
    ----
    class_names: list of class labels (already normalized), e.g. ['human','furniture',...]
    types_map:   {class: [type1, type2, ...]} where type names are already normalized
                 and may contain the special generic type like '__generic__'.

    Returns
    -------
    prof2id:     {"class::type": pid}  contiguous 0..P-1
    id2prof:     {pid: (class, type)}
    class_to_pids: {class: [pid, ...]} convenience index
    """
    prof2id: Dict[str, int] = {}
    id2prof: Dict[int, Tuple[str, str]] = {}
    class_to_pids: Dict[str, List[int]] = {}

    # Deterministic ordering for stability across runs
    for cname in sorted(class_names):
        for t in sorted(types_map.get(cname, [])):
            key = make_prof_key(cname, t)
            if key not in prof2id:
                pid = len(prof2id)
                prof2id[key] = pid
                id2prof[pid] = (cname, t)
                class_to_pids.setdefault(cname, []).append(pid)

    return prof2id, id2prof, class_to_pids

def fill_allowed_per_class(cls2id: Dict[str,int],
                           types_map: Dict[str, List[str]],
                           prof2id: Dict[str,int]) -> Dict[int, Set[int]]:
    """
    Produce {class_id: set(profile_ids)} where profiles belong to that class.

    Args
    ----
    cls2id:   {'human':1, 'furniture':2, ...}  (background not included here)
    types_map:{'human':['male','female',...], ...}
    prof2id:  {"class::type": pid}

    Returns
    -------
    allowed_per_class: {cid: set([pid, ...])}
    """
    allowed: Dict[int, Set[int]] = {}
    for cname, cid in cls2id.items():
        s: Set[int] = set()
        for t in types_map.get(cname, []):
            key = make_prof_key(cname, t)
            if key in prof2id:
                s.add(prof2id[key])
        allowed[cid] = s
    return allowed

# -----------------------------------------------------------------------------#
# Introspection / validation
# -----------------------------------------------------------------------------#
def invert_prof2id(prof2id: Dict[str,int]) -> Dict[int, str]:
    """Return {pid: 'class::type'} inverse mapping."""
    return {pid: key for key, pid in prof2id.items()}

def sanity_check_profiles(class_names: List[str],
                          types_map: Dict[str, List[str]],
                          prof2id: Dict[str,int]) -> List[str]:
    """
    Check that every (class, type) in types_map has an entry in prof2id.
    Returns a list of human-readable warnings (empty if all good).
    """
    warnings: List[str] = []
    for cname in class_names:
        for t in types_map.get(cname, []):
            key = make_prof_key(cname, t)
            if key not in prof2id:
                warnings.append(f"[profiles] Missing pid for '{key}'")
    return warnings

def subset_profiles_for_classes(target_classes: List[str],
                                prof2id: Dict[str,int]) -> Set[int]:
    """
    Collect all profile-ids that belong to any of the target classes.
    """
    out: Set[int] = set()
    for key, pid in prof2id.items():
        c, _ = parse_prof_key(key)
        if c in target_classes:
            out.add(pid)
    return out

# -----------------------------------------------------------------------------#
# Serialization helpers for checkpoints / meta.json
# -----------------------------------------------------------------------------#
def export_profiles_meta(prof2id: Dict[str,int],
                         allowed_per_class: Dict[int, Set[int]],
                         cls2id: Dict[str,int]) -> Dict:
    """
    Produce a JSON-serializable meta dictionary with stable ordering.
    """
    meta = {
        "profiles": {k: int(v) for k, v in sorted(prof2id.items(), key=lambda kv: kv[1])},
        "allowed_per_class": {int(cid): sorted(list(pids)) for cid, pids in allowed_per_class.items()},
        "classes": {k: int(v) for k, v in sorted(cls2id.items(), key=lambda kv: kv[1])},
    }
    return meta

def import_profiles_meta(meta: Dict) -> Tuple[Dict[str,int], Dict[int, Set[int]], Dict[str,int]]:
    """
    Inverse of export (best-effort). Accepts same structure and returns (prof2id, allowed_per_class, cls2id).
    """
    prof2id = {str(k): int(v) for k, v in meta.get("profiles", {}).items()}
    allowed_per_class = {int(k): set(int(x) for x in v) for k, v in meta.get("allowed_per_class", {}).items()}
    cls2id = {str(k): int(v) for k, v in meta.get("classes", {}).items()}
    return prof2id, allowed_per_class, cls2id

# -----------------------------------------------------------------------------#
# Convenience: remap/guard helpers
# -----------------------------------------------------------------------------#
def clamp_profile_targets(Yp: "np.ndarray", n_profiles: int, ignore_val: int = -1) -> "np.ndarray":
    """
    Ensure Yp (voxelwise profile targets) stays in [-1, n_profiles-1].
    Any id outside range is set to ignore_val.
    """
    import numpy as np
    Yp = Yp.copy()
    bad = (Yp >= n_profiles) | (Yp < -1)
    if bad.any():
        Yp[bad] = ignore_val
    return Yp

def map_points_to_profile_ids(class_labels_pts: "np.ndarray",
                              type_labels_pts: "np.ndarray",
                              prof2id: Dict[str,int]) -> "np.ndarray":
    """
    Convenience: given per-point class string labels and type string labels,
    assemble 'class::type' keys and map to global profile ids (or -1 if missing).
    """
    import numpy as np
    assert class_labels_pts.shape == type_labels_pts.shape
    N = class_labels_pts.shape[0]
    out = np.full((N,), -1, dtype=np.int64)
    for i in range(N):
        key = make_prof_key(str(class_labels_pts[i]), str(type_labels_pts[i]))
        out[i] = prof2id.get(key, -1)
    return out
