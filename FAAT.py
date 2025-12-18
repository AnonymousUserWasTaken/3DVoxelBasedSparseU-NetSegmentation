# feature_alignment_helpers
# FAAT = FeatureAlignmentAttentionTokens
# Lightweight class-prototype alignment features for 3D voxel grids.
# Works on NumPy tensors and plugs into your existing "feats, Yc, meta" path.

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional

def _safe_l2_normalize(x: np.ndarray, axis: int, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def _make_box_filter_3d(arr: np.ndarray, ks: int) -> np.ndarray:
    """
    Simple 3D box smoothing over (D,H,W). Pure NumPy, no SciPy dependency.
    ks must be odd. If ks<=1 returns arr unchanged.
    """
    if ks is None or ks <= 1:
        return arr
    assert ks % 2 == 1, "smooth_ks must be odd"
    pad = ks // 2
    D, H, W = arr.shape
    # integral volume trick (summed area)
    acc = np.pad(arr, ((1,0),(1,0),(1,0)), mode='constant', constant_values=0).astype(np.float64)
    acc = acc.cumsum(0).cumsum(1).cumsum(2)

    def box_sum(x0, x1, y0, y1, z0, z1):
        return (acc[x1, y1, z1]
              - acc[x0, y1, z1]
              - acc[x1, y0, z1]
              - acc[x1, y1, z0]
              + acc[x0, y0, z1]
              + acc[x0, y1, z0]
              + acc[x1, y0, z0]
              - acc[x0, y0, z0])

    out = np.empty_like(arr, dtype=np.float32)
    for x in range(D):
        x0 = max(0, x - pad); x1 = min(D - 1, x + pad)
        for y in range(H):
            y0 = max(0, y - pad); y1 = min(H - 1, y + pad)
            for z in range(W):
                z0 = max(0, z - pad); z1 = min(W - 1, z + pad)
                s = box_sum(x0, x1+1, y0, y1+1, z0, z1+1)
                nvox = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
                out[x, y, z] = s / max(1, nvox)
    return out

def _gather_class_proto(
    feats: np.ndarray,      # (C,D,H,W)
    Yc: np.ndarray,         # (D,H,W) uint8/int
    class_id: int,
    proto_channels: List[int],
    band: Optional[np.ndarray],
    occ: Optional[np.ndarray],
    clip: int,
) -> Optional[np.ndarray]:
    """
    Compute a class prototype by averaging selected channels over voxels where Yc==class_id.
    Weight by band and occ if provided. Returns (len(proto_channels),) or None if too few voxels.
    """
    mask = (Yc == class_id)
    if not mask.any():
        return None
    # Gather per-voxel feature vectors from the specified channels
    Csel = len(proto_channels)
    D, H, W = Yc.shape
    V = mask.sum()
    if V <= 0:
        return None

    # Limit to at most 'clip' voxels (uniform)
    if V > clip:
        idxs = np.flatnonzero(mask.ravel())
        pick = np.random.choice(idxs, size=clip, replace=False)
        pick_mask = np.zeros((D*H*W,), dtype=bool); pick_mask[pick] = True
        mask = pick_mask.reshape(D, H, W)

    X = feats[proto_channels]  # (Csel,D,H,W)
    X = X[:, mask]             # (Csel, V_eff)

    # Optional positive weighting for band/occ
    if band is not None:
        bw = band.squeeze(0)[mask].astype(np.float32)  # (V_eff,)
    else:
        bw = None
    if occ is not None:
        ow = occ.squeeze(0)[mask].astype(np.float32)   # (V_eff,)
    else:
        ow = None

    if bw is not None:
        w = bw
        if ow is not None:
            w = w + ow
    else:
        w = ow

    if w is None:
        proto = X.mean(axis=1)  # (Csel,)
    else:
        w = w.reshape(1, -1)
        s = np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
        proto = (X * w).sum(axis=1) / s.squeeze(1)  # (Csel,)

    return proto.astype(np.float32)

def augment_features_with_alignment(
    feats: np.ndarray,                   # (C,D,H,W)
    Yc: np.ndarray,                      # (D,H,W)
    class_id_map: Dict[str, int],        # e.g., {'human':1, 'floor':2, 'staircase':3}
    target: str = "human",
    negatives: Optional[List[str]] = None,
    band: Optional[np.ndarray] = None,   # (1,D,H,W) or None
    occ: Optional[np.ndarray] = None,    # (1,D,H,W) or None
    proto_channels: Optional[List[int]] = None,  # which input channels to embed (default: [0,1,2,4,5])
    per_class_clip: int = 20000,
    temperature: float = 0.07,
    band_gain: float = 2.0,
    occ_gain: float = 1.0,
    smooth_ks: int = 0,
    include_sim_channels: bool = False,
    return_debug: bool = False,
):
    """
    Adds an alignment margin channel:
        margin = (cos(x, proto_target) - max_j cos(x, proto_neg_j))
    optionally scaled by (1 + band_gain*band + occ_gain*occ);
    optionally smoothed with a 3D box filter (ks odd).

    Returns:
        feats_aug: (C + 1 [+ 2 if include_sim], D,H,W)
        info: dict with debug info if return_debug=True
    """
    assert feats.ndim == 4
    C, D, H, W = feats.shape

    if proto_channels is None:
        proto_channels = [0, 1, 2, 4, 5]  # x,y,z + tsdf_mag + band

    # Collect target/neg ids
    if target not in class_id_map:
        # nothing to do
        return feats, {"added": 0, "reason": "target_not_in_map"} if return_debug else feats
    tid = class_id_map[target]
    neg_names = [k for k in class_id_map.keys() if k != target] if negatives is None else list(negatives)
    neg_ids = [class_id_map[n] for n in neg_names if n in class_id_map]

    # Build prototypes
    # Extract band/occ masks if passed
    band_np = band if (band is not None and band.shape[0] == 1) else None
    occ_np  = occ  if (occ  is not None and occ.shape[0]  == 1) else None

    tgt_proto = _gather_class_proto(
        feats, Yc, tid, proto_channels, band_np, occ_np, clip=per_class_clip
    )
    if tgt_proto is None:
        return feats, {"added": 0, "reason": "no_target_voxels"} if return_debug else feats

    neg_protos = []
    for nid in neg_ids:
        pr = _gather_class_proto(
            feats, Yc, nid, proto_channels, band_np, occ_np, clip=per_class_clip
        )
        if pr is not None:
            neg_protos.append(pr)
    if len(neg_protos) == 0:
        # No negatives → no-op
        return feats, {"added": 0, "reason": "no_negative_protos"} if return_debug else feats

    tgt_proto = _safe_l2_normalize(tgt_proto, axis=0)
    neg_protos = np.stack(neg_protos, axis=0)  # (Nneg, Csel)
    neg_protos = _safe_l2_normalize(neg_protos, axis=1)

    # Per-voxel descriptors over selected channels
    X = feats[proto_channels]                     # (Csel,D,H,W)
    Xv = X.reshape(len(proto_channels), -1).T     # (V, Csel)
    Xv = _safe_l2_normalize(Xv, axis=1)          # cosine-safe

    # Cosine similarities
    sim_t = (Xv @ tgt_proto.reshape(-1, 1)).reshape(D, H, W)                # (D,H,W)
    sim_n_all = (Xv @ neg_protos.T).reshape(D, H, W, -1)                    # (D,H,W,Nneg)
    sim_n = sim_n_all.max(axis=3)                                           # (D,H,W)

    margin = sim_t - sim_n                                                  # (D,H,W)

    # Optional scale by band/occ emphasis
    if band_np is not None or occ_np is not None:
        scale = 1.0
        if band_np is not None:
            scale = scale + band_gain * band_np.squeeze(0)
        if occ_np is not None:
            scale = scale + occ_gain * occ_np.squeeze(0)
        margin = margin * scale

    # Temperature (soft sharpening)
    if temperature is not None and temperature > 0:
        margin = margin / float(temperature)

    # Optional smoothing
    if smooth_ks and smooth_ks > 1:
        margin = _make_box_filter_3d(margin.astype(np.float32), smooth_ks)

    margin = margin.astype(np.float32)
    margin = margin[np.newaxis, ...]  # (1,D,H,W)

    if include_sim_channels:
        simt = sim_t[np.newaxis, ...].astype(np.float32)
        simn = sim_n[np.newaxis, ...].astype(np.float32)
        feats_aug = np.concatenate([feats, margin, simt, simn], axis=0)
        info = {"added": 3, "channels": ["align_margin", "sim_target", "sim_negmax"]}
    else:
        feats_aug = np.concatenate([feats, margin], axis=0)
        info = {"added": 1, "channels": ["align_margin"]}

    if return_debug:
        info.update({
            "target_id": tid,
            "neg_ids": neg_ids,
            "proto_len": len(proto_channels),
            "DHW": (D, H, W)
        })
        return feats_aug, info
    return feats_aug
