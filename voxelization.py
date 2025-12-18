# voxelization.py
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch

# ==============================
# Voxel sizing & rasterization
# ==============================
def auto_fit_voxel_size(P: np.ndarray, grid: int,
                        desired_vs: float,
                        cover_q: float = 0.95,
                        pad_empty_border: int = 1,
                        vs_min: float = 0.01,
                        vs_max: float = 0.20) -> float:
    if P.shape[0] == 0:
        return desired_vs
    c = P.mean(0, keepdims=True)
    R = P - c
    r = np.max(np.abs(R), axis=1)
    L_need = max(2.0 * float(np.quantile(r, cover_q)), 1e-6)
    vs = L_need / float(grid - 2*pad_empty_border)
    return float(np.clip(vs, vs_min, vs_max))

def random_origin_shift(grid: int, vs: float, frac: float) -> np.ndarray:
    half_extent = 0.5 * grid * vs
    jitter_frac = min(frac, 0.10)
    delta = (np.random.uniform(-jitter_frac, jitter_frac, size=3).astype(np.float32)) * half_extent
    return delta

def voxelize_points(P: np.ndarray, vs: float, G: int, origin_shift: Optional[np.ndarray]=None):
    c = P.mean(0, keepdims=True) if P.shape[0] else np.zeros((1,3), np.float32)
    half = (G * vs)/2.0
    mn = (c[0] - half)
    if origin_shift is not None:
        mn = mn + origin_shift.astype(np.float32)
    gc = np.floor((P - mn)/vs).astype(np.int32) if P.shape[0] else np.zeros((0,3), np.int32)
    valid_mask = (np.all((gc >= 0) & (gc < G), axis=1) if gc.size else np.zeros((0,), bool))
    gc_in = gc[valid_mask]
    if gc_in.size == 0:
        return mn, np.zeros((0,3),np.int32), np.zeros((0,),np.int64), np.zeros((0,),np.int32), valid_mask
    U, inv = np.unique(gc_in, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=U.shape[0]).astype(np.int32)
    return mn, U, inv.astype(np.int64), counts, valid_mask

# ==============================
# PCA-guided SURFACE voxelization
# ==============================
def _pca_surface_features_for_bins(P_valid: np.ndarray,
                                   uc: np.ndarray,
                                   inv: np.ndarray,
                                   counts: np.ndarray,
                                   G: int,
                                   vs: float,
                                   origin: np.ndarray,
                                   plane_thickness_vox: float = 0.5,
                                   inplane_k: float = 2.5,
                                   min_axis_vox: float = 0.75,
                                   max_axis_vox: float = 4.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-guided surface voxelization.

    For each occupied voxel, fit PCA to its points, determine local plane (normal = smallest-eigenvalue vec),
    and 'splat' an elliptical patch into neighboring voxels:
        |u| <= a, |v| <= b in the PCA plane and |w| <= t (thin thickness).
    The ellipse radii a,b are proportional to sqrt(lambda1), sqrt(lambda2) (clamped in voxel units).

    Args:
      P_valid: (N,3) points that landed inside the grid (after valid_mask)
      uc:      (V,3) unique occupied voxel indices (gx,gy,gz)
      inv:     (N,)  mapping point->voxel bin index in [0,V)
      counts:  (V,)  per-voxel point counts
      G:       grid size (cubic GxGxG)
      vs:      voxel size (meters)
      origin:  world-space min-corner of the grid (mn from voxelize_points)
      plane_thickness_vox: half-thickness in *voxels* for the sheet (±t), default 0.5 voxel
      inplane_k: scale factor on sqrt eigenvalues to set ellipse size
      min_axis_vox / max_axis_vox: clamp ellipse radii (in voxels)

    Returns:
      n_norm_grid: (G,G,G) normalized counts
      curv_grid:   (G,G,G) curvature proxy lambda3/sum
      surf_grid:   (G,G,G) surface occupancy (1 where the PCA-splat hits)
    """
    nvox = uc.shape[0]
    n_norm_grid = np.zeros((G,G,G), np.float32)
    curv_grid   = np.zeros((G,G,G), np.float32)
    surf_grid   = np.zeros((G,G,G), np.float32)

    if nvox == 0:
        return n_norm_grid, curv_grid, surf_grid

    # Build bins (point indices per voxel)
    bins: List[List[int]] = [[] for _ in range(nvox)]
    for p_i, v_i in enumerate(inv):
        bins[int(v_i)].append(p_i)

    max_cnt = float(counts.max() if counts.size else 1.0)
    eps = 1e-8

    # helper: voxel center (world) from integer grid index
    # centers at origin + (i + 0.5) * vs
    def vox_center_world(ix:int, iy:int, iz:int) -> np.ndarray:
        return origin + np.array([(ix + 0.5)*vs, (iy + 0.5)*vs, (iz + 0.5)*vs], dtype=np.float32)

    for vi, lst in enumerate(bins):
        gx, gy, gz = int(uc[vi,0]), int(uc[vi,1]), int(uc[vi,2])

        # base channels (counts + default curvature)
        n_norm = (len(lst) / max_cnt) if max_cnt > 0 else 0.0
        n_norm_grid[gx, gy, gz] = float(n_norm)

        if len(lst) < 3:
            # Not enough points to estimate a plane → mark the voxel as a surface hit only
            surf_grid[gx, gy, gz] = 1.0 if len(lst) > 0 else 0.0
            curv_grid[gx, gy, gz] = 0.0
            continue

        # PCA on this voxel's points
        Q  = P_valid[lst]  # (k,3)
        mu = Q.mean(axis=0, keepdims=True)
        X  = Q - mu
        C  = (X.T @ X) / max(len(lst)-1, 1)
        w, V = np.linalg.eigh(C)        # ascending eigenvalues; columns in V are eigenvectors
        w = np.clip(w, 0.0, None)
        # Largest two span the tangent plane; smallest is the normal
        lam_small, lam_mid, lam_large = float(w[0]), float(w[1]), float(w[2])
        e_small = V[:,0]  # normal (perpendicular)
        e_mid   = V[:,1]
        e_large = V[:,2]
        # Orthonormal basis R with columns [e_large, e_mid, e_small]
        R = np.stack([e_large, e_mid, e_small], axis=1).astype(np.float32)  # (3,3)

        # curvature proxy
        s = lam_small + lam_mid + lam_large + eps
        curvature = lam_small / s
        curv_grid[gx, gy, gz] = float(curvature)

        # Ellipse radii in-plane (meters), from sqrt(lambda)
        a_m = inplane_k * math.sqrt(max(lam_large, eps))
        b_m = inplane_k * math.sqrt(max(lam_mid,   eps))
        # Convert to voxel units and clamp
        a_v = float(np.clip(a_m / vs, min_axis_vox, max_axis_vox))
        b_v = float(np.clip(b_m / vs, min_axis_vox, max_axis_vox))

        # Thickness (meters)
        t_v = plane_thickness_vox
        t_m = t_v * vs

        # Neighborhood search window in voxel coords
        rad_v = int(math.ceil(max(a_v, b_v) + t_v))  # conservative
        ix0, ix1 = max(0, gx - rad_v), min(G-1, gx + rad_v)
        iy0, iy1 = max(0, gy - rad_v), min(G-1, gy + rad_v)
        iz0, iz1 = max(0, gz - rad_v), min(G-1, gz + rad_v)

        mu_w = mu[0].astype(np.float32)

        # Scan neighbor voxel centers; accept those near plane & inside ellipse
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    c_w = vox_center_world(ix, iy, iz)
                    # Local coords in PCA frame: u along e_large, v along e_mid, w along normal
                    d   = c_w - mu_w
                    uvw = R.T @ d  # (3,)
                    # plane proximity & in-plane ellipse test
                    if abs(uvw[2]) <= t_m and ((uvw[0] / (a_v * vs + eps))**2 + (uvw[1] / (b_v * vs + eps))**2) <= 1.0:
                        surf_grid[ix, iy, iz] = 1.0

        # Always include the seed voxel
        surf_grid[gx, gy, gz] = 1.0

    return n_norm_grid, curv_grid, surf_grid

# ==============================
# Feature builder & targets
# ==============================
def build_feature_grid(P: np.ndarray,
                       Lc_pts: np.ndarray,
                       Lp_pts: np.ndarray,
                       Inst_pts: np.ndarray,
                       cfg,
                       scene_id: str = "",
                       train_mode: bool = True):
    """
    Returns feats(7,D,H,W), Yc(D,H,W), Yp(D,H,W), meta.

    Channels:
      0: x_rel    (coord)
      1: y_rel    (coord)
      2: z_rel    (coord)
      3: ones     (occupancy indicator channel base)
      4: n_norm   (normalized per-voxel point count)
      5: band     (SURFACE occupancy from PCA-guided splat)
      6: curvature(lambda3/sum)
    """
    G = cfg.grid_size
    vs_des = cfg.voxel_size_m
    if cfg.auto_fit_voxel_size:
        vs_used = auto_fit_voxel_size(P, G, vs_des,
                                      cfg.auto_fit_cover_frac,
                                      cfg.pad_empty_border,
                                      cfg.auto_fit_min_vs,
                                      cfg.auto_fit_max_vs)
    else:
        vs_used = vs_des

    origin_shift = random_origin_shift(G, vs_used, cfg.origin_jitter_frac) if train_mode else None
    origin, uc, inv, counts, valid_mask = voxelize_points(P, vs_used, G, origin_shift=origin_shift)

    # Retry if empty but there are points
    if uc.shape[0] == 0 and P.shape[0] > 0:
        origin, uc, inv, counts, valid_mask = voxelize_points(P, vs_used, G, origin_shift=None)
    if uc.shape[0] == 0 and P.shape[0] > 0:
        vs_auto = auto_fit_voxel_size(P, G, vs_des,
                                      cfg.auto_fit_cover_frac,
                                      cfg.pad_empty_border,
                                      cfg.auto_fit_min_vs,
                                      cfg.auto_fit_max_vs)
        origin, uc, inv, counts, valid_mask = voxelize_points(P, vs_auto, G, origin_shift=None)
        vs_used = vs_auto

    feats_dense = np.zeros((4, G, G, G), np.float32)
    Yc = np.zeros((G, G, G), np.uint8)
    Yp = np.full((G, G, G), -1, np.int32)

    if uc.shape[0] == 0 and P.shape[0] > 0:
        # grid is empty due to auto-fit/shift; return empty features
        z = np.zeros((G,G,G), np.float32)
        feats = np.concatenate([feats_dense, np.stack([z,z,z], axis=0)], axis=0)
        meta = dict(grid_origin_m=origin, uc=uc, inv=inv, valid_mask=valid_mask,
                    voxel_size_m=vs_used, empty=True,
                    point_instance_ids=np.zeros((0,), np.int64))
        return feats, Yc, Yp, meta

    # Majority votes
    Lc = Lc_pts[valid_mask]
    Lp = Lp_pts[valid_mask]
    bins_idx = [[] for _ in range(uc.shape[0])]
    for p_i, v_i in enumerate(inv):
        bins_idx[int(v_i)].append(p_i)

    maj_c = np.empty((uc.shape[0],), np.int64)
    maj_p = np.full((uc.shape[0],), -1, np.int64)
    for i, lst in enumerate(bins_idx):
        if not lst:
            maj_c[i] = 0; maj_p[i] = -1
        else:
            cvals, ccnts = np.unique(Lc[lst], return_counts=True)
            cid = int(cvals[ccnts.argmax()])
            maj_c[i] = cid
            lst_c = [k for k in lst if Lc[k] == cid]
            if lst_c:
                pvals, pcnts = np.unique(Lp[lst_c], return_counts=True)
                maj_p[i] = int(pvals[pcnts.argmax()])
            else:
                maj_p[i] = -1

    gx, gy, gz = uc[:,0], uc[:,1], uc[:,2]
    Yc[gx,gy,gz] = maj_c.astype(np.uint8)
    Yp[gx,gy,gz] = maj_p.astype(np.int32)

    # Coordinate channels ([-1,1]) optionally centered
    ucn = (uc.astype(np.float32)/max(G-1,1))*2.0 - 1.0
    if cfg.use_relative_coords:
        ctr = ucn.mean(axis=0, keepdims=True).astype(np.float32)
        ucn = ucn - ctr
    feats_dense[0, gx, gy, gz] = ucn[:,0]
    feats_dense[1, gx, gy, gz] = ucn[:,1]
    feats_dense[2, gx, gy, gz] = ucn[:,2]
    feats_dense[3, gx, gy, gz] = 1.0
    if train_mode and cfg.coorddrop_p > 0.0 and np.random.rand() < cfg.coorddrop_p:
        feats_dense[0:3, ...] = 0.0

    # PCA-guided SURFACE voxelization + features
    P_valid = P[valid_mask] if P.shape[0] else np.zeros((0,3), np.float32)
    n_norm_grid, curv_grid, surf_grid = _pca_surface_features_for_bins(
        P_valid, uc, inv, counts, G, vs_used, origin
    )

    # 'band' now means SURFACE occupancy
    band_grid = surf_grid

    # Assemble features (7 channels): [x_rel,y_rel,z_rel,ones,n_norm,band(surface),curvature]
    feats = np.concatenate(
        [feats_dense,
         np.stack([n_norm_grid, band_grid, curv_grid], axis=0)],
        axis=0
    )

    # Compact instance ids for instance head
    compact_inst = Inst_pts[valid_mask] if Inst_pts.size>0 else np.zeros((0,), np.int64)
    meta = dict(grid_origin_m=origin, uc=uc, inv=inv, valid_mask=valid_mask,
                voxel_size_m=vs_used, empty=False,
                point_instance_ids=compact_inst)
    return feats, Yc, Yp, meta

# ==========================================
# Instance targets (center heatmap + offsets)
# ==========================================
def gaussian_3d_on_grid(center, shape, sigma):
    D,H,W = shape
    cx,cy,cz = center
    r = int(max(1, math.ceil(3*sigma)))
    x0,x1 = max(0,int(cx)-r), min(D-1,int(cx)+r)
    y0,y1 = max(0,int(cy)-r), min(H-1,int(cy)+r)
    z0,z1 = max(0,int(cz)-r), min(W-1,int(cz)+r)
    xs = np.arange(x0, x1+1)[:,None,None]
    ys = np.arange(y0, y1+1)[None,:,None]
    zs = np.arange(z0, z1+1)[None,None,:]
    g = np.exp(-(((xs-cx)**2 + (ys-cy)**2 + (zs-cz)**2) / (2*sigma*sigma))).astype(np.float32)
    return (x0,x1,y0,y1,z0,z1), g

def make_instance_targets_from_ids(uc: np.ndarray,
                                   voxel_yc: np.ndarray,
                                   inv: np.ndarray,
                                   point_instance_ids_compact: np.ndarray,
                                   target_class_id: int,
                                   sigma: float):
    D,H,W = voxel_yc.shape
    center = np.zeros((D,H,W), np.float32)
    offsets = np.zeros((3,D,H,W), np.float32)
    mask_off = np.zeros((D,H,W), bool)

    if uc.shape[0] == 0 or inv.shape[0] == 0 or point_instance_ids_compact.shape[0] == 0:
        return center, offsets, mask_off, center

    V = uc.shape[0]
    bins = [[] for _ in range(V)]
    for p_i, v_i in enumerate(inv):
        bins[int(v_i)].append(int(point_instance_ids_compact[p_i]))

    inst_to_voxels: Dict[int, list] = {}
    gx,gy,gz = uc[:,0], uc[:,1], uc[:,2]
    for vi, lst in enumerate(bins):
        if not lst: continue
        if voxel_yc[gx[vi],gy[vi],gz[vi]] != target_class_id:
            continue
        ids, cnts = np.unique(np.array(lst, dtype=np.int64), return_counts=True)
        inst = int(ids[cnts.argmax()])
        if inst <= 0: continue
        inst_to_voxels.setdefault(inst, []).append(vi)

    norm_x = max(D-1, 1); norm_y = max(H-1, 1); norm_z = max(W-1, 1)

    for _, vox_indices in inst_to_voxels.items():
        coords = uc[np.array(vox_indices, dtype=np.int64)]
        c = coords.mean(axis=0)
        (x0,x1,y0,y1,z0,z1), g = gaussian_3d_on_grid(c, (D,H,W), sigma)
        center[x0:x1+1, y0:y1+1, z0:z1+1] = np.maximum(center[x0:x1+1, y0:y1+1, z0:z1+1], g)

        vx = coords[:,0]; vy = coords[:,1]; vz = coords[:,2]
        dx = (c[0] - vx).astype(np.float32) / float(norm_x)
        dy = (c[1] - vy).astype(np.float32) / float(norm_y)
        dz = (c[2] - vz).astype(np.float32) / float(norm_z)
        offsets[0, vx, vy, vz] = dx
        offsets[1, vx, vy, vz] = dy
        offsets[2, vx, vy, vz] = dz
        mask_off[vx, vy, vz] = True

    return center, offsets, mask_off, center

def build_bev_targets_from_centers(center_3d: np.ndarray):
    return center_3d.max(axis=2).transpose(1,0)  # (H,W)
