# train.py --This is the Main Hub as I like to reference it, where all of the knobs are at
"""
Training with PCA/geometry voxel features (SURFACE-aware; no TSDF):
 - Feature builder returns 7 channels:
   [x_rel, y_rel, z_rel, ones, n_norm, band(surface), curvature]
 - Dataset/mixer unchanged
 - Loss still supports 'band' weighting (band = SURFACE occupancy)
 - Current Patch (12/16/2025), Surface Feature Extraction 
"""

from __future__ import annotations
import os, sys, json, time, traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Blender / UPBGE bridge ───────────────────────────────────────────────────
import bpy
BLEND_DIR = os.path.dirname(bpy.data.filepath)
if BLEND_DIR and BLEND_DIR not in sys.path:
    sys.path.insert(0, BLEND_DIR)

# Local imports
from utility import seed_all, as_numpy_1, resolve_device
from tof_helpers import scan_training_catalog_profiles, extract_points_from_object, TRAINING_ROOT, GENERIC_TYPE
from profiles import build_global_profiles, fill_allowed_per_class
from mixer import MixerParams, SceneMixer
from voxelization import build_feature_grid, make_instance_targets_from_ids, build_bev_targets_from_centers
from models.unet3d import UNet3DInstBEV
from loss import DiceLoss, focal_bce_with_logits, cross_entropy_band_weighted, mask_prof_logits_illegal

# =============================================================================
# Config
#If YOU'RE wondering why this is here, I have secret plans
# When we want to create new Model checkpoints we want Metadata initially 
#   To avoid hard-crashes, we can detect the kind of parameters we're dealing 
#    With already instead of throwing hard tensor crashes when loading checkpoints
#      also good for initial pre-processing steps I've come to discover, for 
#       Datasets...
# =============================================================================
@dataclass
class Cfg:
    out_dir: str = os.path.join(BLEND_DIR, "checkpoints")
    run_name: str = "unet3d_pca_surface_features"   # renamed to reflect SURFACE features

    training_root: str = TRAINING_ROOT
    use_profiles: bool = True
    min_profiles_to_enable: int = 2

    # voxel grid Type Hinting goes brazy
    voxel_size_m: float = 0.03 # Sets the voxel Grid initial size Float 3D 0.03 scale = smaller scale = More resolution.
    grid_size: int = 128 # Grid size of the cubic voxel bins (128x128x128) 
    pad_empty_border: int = 1 # padding empty border size so if we have empty border at the ends what do we pad the voxel down to 1x1x1? 
    auto_fit_voxel_size: bool = True # Overrides voxel_size_m and will autofit pointclouds instead 
    auto_fit_cover_frac: float = 0.95 # Amount of points to cover entirely, Avoiding farther outliers, taking more centroid position of points within the entire cloud
    auto_fit_min_vs: float = 0.01 # Avoids OOM errors FML
    auto_fit_max_vs: float = 0.20 # The idea is that we clamp the fitted voxel size so it can't voxelize at insanely tiny scales or large scales.
    # Don't tweak with any of the auto features just yet, I have to patch up and omit a few things in some future implementations
    
    #Coordinate features (x_rel, y_rel, z_rel)
    use_relative_coords: bool = True #X,Y,Z = (x_relu,y_relu,z_relu)
    coorddrop_p: float = 0.30 

# augs (Augmentations) I hope to remove most of these in future, they are really just here for some experimenting
    
    # Point Drop Randomizer
    aug_point_drop_min: float = 0.1 # amount of points to randomly drop minimal
    aug_point_drop_max: float = 0.3 # amount of points to randomly drop maximum
    
    #Point Scale Randomizer
    aug_scale_min: float = 0.95 #amount of points to minimally scale by
    aug_scale_max: float = 1.05 #amount of points to maximally scale by
    
    #Translation Augmenting
    aug_translate_frac: float = 0.10 # Amount to translate by
    aug_yaw_only: bool = True # Just on Yaw Axis?

    origin_jitter_frac: float = 0.20 # Point Jitter Augmenting (We shift the points a bit from an origin)
# -End OF #  augs

# Heads    

    # BEV head --I like this Bird's Eye View Idea, Orthographic Top-Down View
    # Essentially we collapse the Height in the voxel dimension (Our 3D cubic Bins of the original data)
    enable_bev_head: bool = True  # Enable Bird's Eye View in training?
    bev_band_only: bool = True     # band = SURFACE occupancy now
    bev_loss_w: float = 0.5         # Bird's Eye View Loss starting float (Default 0.5)

    # instance head
    center_gauss_sigma_vox: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    offset_loss_w: float = 0.25
    center_loss_w: float = 1.0

    # losses
    ce_w: float = 1.0
    dice_w: float = 1.0
    band_weight_factor: float = 4.0   # still used (with SURFACE as the band)
    use_dice: bool = True
    class_weights: Optional[List[float]] = None
    background_id: int = 0

    # train loop
    epochs: int = 20
    batch_size: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    amp: bool = True

    save_every: int = 1
    seed: int = 1337
    debug_prints: bool = True

cfg = Cfg() # Dr. Rees = >:(

def make_outdir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def atomic_torch_save(state_dict: Dict, final_path: str):
    directory = os.path.dirname(final_path)
    make_outdir(directory)
    tmp_path = final_path + ".tmp"
    torch.save(state_dict, tmp_path)
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp_path, final_path)
    try:
        dir_fd = os.open(directory, os.O_DIRECTORY)
        os.fsync(dir_fd)
        os.close(dir_fd)
    except Exception:
        pass

# =============================================================================
# Catalog scan
# =============================================================================
HUMAN_NAME_CANDIDATES = {"human","humans","person","people","humanoid","character","char"}
def detect_default_instance_class(class_names: List[str],
                                  types_map: Dict[str, List[str]],
                                  samples: List[Dict]) -> str:
    lower_to_orig = {c.lower(): c for c in class_names}
    for cand in HUMAN_NAME_CANDIDATES:
        if cand in lower_to_orig:
            return lower_to_orig[cand]
    from collections import Counter
    cnt = Counter([s["class"] for s in samples])
    return cnt.most_common(1)[0][0] if cnt else (class_names[0] if class_names else "human")

# =============================================================================
# Curriculum
# =============================================================================
def curriculum_params(epoch:int, class_names: List[str]) -> MixerParams:
    human_like = None
    for c in class_names:
        if c.lower() in HUMAN_NAME_CANDIDATES:
            human_like = c; break
    base_pmf = {c: 1.0 for c in class_names}
    if human_like:
        base_pmf = {c: (3.0 if c==human_like else 1.0) for c in class_names}

    if epoch <= 4:
        pmf = {1:1.0}
        return MixerParams(pmf_counts=pmf,
                           min_separation_m=(1.2, 1.6),
                           overhead_mode="off",
                           sensor_dropout=(0.05, 0.10),
                           region_xy_m=(2.0, 2.0),
                           class_sampling_pmf=base_pmf)
    elif epoch <= 10:
        pmf = {1:0.3, 2:0.5, 3:0.2}
        return MixerParams(pmf_counts=pmf,
                           min_separation_m=(0.7, 0.9),
                           overhead_mode="off",
                           sensor_dropout=(0.08, 0.18),
                           region_xy_m=(2.5, 2.5),
                           class_sampling_pmf=base_pmf)
    else:
        pmf = {1:0.15, 2:0.35, 3:0.25, 5:0.15, 7:0.10}
        return MixerParams(pmf_counts=pmf,
                           min_separation_m=(0.35, 0.6),
                           overhead_mode="heads_only" if (epoch % 2 == 0) else "off",
                           z_crop_top_cm=(20.0, 60.0),
                           sensor_dropout=(0.10, 0.30),
                           region_xy_m=(2.4, 2.4),
                           class_sampling_pmf=base_pmf)

# =============================================================================
# Dataset
# =============================================================================
class BlenderMixtureDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        self.split = split
        self.class_names, self.types_map, self.samples = scan_training_catalog_profiles(cfg.training_root)

        self.class_names = sorted(self.class_names)
        self.cls2id = {c: (i+1) for i, c in enumerate(self.class_names)}
        self.id2cls = {i: c for c, i in self.cls2id.items()}

        # Global profiles
        self.prof2id, _, _ = build_global_profiles(self.class_names, self.types_map)
        self.n_profiles = len(self.prof2id)

        self.allowed_per_class = fill_allowed_per_class(self.cls2id, self.types_map, self.prof2id)

        if cfg.debug_prints:
            print(f"[INIT] classes={self.class_names}")
            print(f"[INIT] total profiles={self.n_profiles}")

        self.instance_class = detect_default_instance_class(self.class_names, self.types_map, self.samples)
        self.mixer = SceneMixer(self.class_names, self.types_map, self.samples, self.cls2id, self.prof2id)
        self.val_items = list(self.samples)
    def __len__(self):
        return 100  #i'm to lazy to redesign  this
    def __getitem__(self, idx):
        if self.split == "train":
            epoch = getattr(self, "_epoch", 7)
            mp = curriculum_params(epoch, self.class_names)
            scene = self.mixer.sample_scene(mp)
            P = scene["points"]
            Lc = scene["labels_class"]
            Lp = scene["labels_prof"]
            Ii = scene["instance_ids"]
            sid = f"mix_e{epoch}"
            return {"points": P, "labels_class": Lc, "labels_prof": Lp, "instance_ids": Ii, "scene_id": sid}
        else:
            s = self.val_items[idx % len(self.val_items)]
            obj = bpy.data.objects[s["object_name"]]
            pts = extract_points_from_object(obj)
            P = pts.detach().cpu().numpy().astype(np.float32)
            cid = self.cls2id[s["class"]]
            key = f"{s['class']}::{s['type']}"
            pid = self.prof2id.get(key, -1)
            Lc = np.full((P.shape[0],), cid, np.int64)
            Lp = np.full((P.shape[0],), pid, np.int64)
            Ii = np.ones((P.shape[0],), np.int64)
            sid = f"val/{s['class']}/{s['type']}/{s['object_name']}"
            return {"points": P, "labels_class": Lc, "labels_prof": Lp, "instance_ids": Ii, "scene_id": sid}

# =============================================================================
# Training
# =============================================================================
def train(device: Optional[str] = None):
    seed_all(cfg.seed)
    device = resolve_device(device)
    out_dir = make_outdir(os.path.join(cfg.out_dir, cfg.run_name))
    print(f"[PATH] BLEND_DIR={BLEND_DIR}")
    print(f"[PATH] Checkpoints dir={out_dir}")

    ds_tr = BlenderMixtureDataset("train")
    ds_va = BlenderMixtureDataset("val")

    def set_epoch_on_loader(loader, epoch):
        setattr(loader.dataset, "_epoch", epoch)

    n_classes = max(ds_tr.cls2id.values()) + 1
    n_profiles = ds_tr.n_profiles if (cfg.use_profiles and ds_tr.n_profiles >= cfg.min_profiles_to_enable) else 0
    instance_class_id = ds_tr.cls2id.get(ds_tr.instance_class, 1)

    tr_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    va_loader = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # 7 input channels as described above
    model = UNet3DInstBEV(in_ch=7, n_classes=n_classes, n_profiles=n_profiles,
                          base=32, enable_bev_head=cfg.enable_bev_head).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type=="cuda")

    dice_loss = DiceLoss(ignore_index=255).to(device)
    class_weights_t = (torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
                       if cfg.class_weights else None)

    best_val = -1e9
    for epoch in range(1, cfg.epochs+1):
        print(f"\n[TRAIN] === Epoch {epoch}/{cfg.epochs} ===")
        model.train()
        set_epoch_on_loader(tr_loader, epoch)
        t0 = time.time()
        tr_loss = 0.0; n_steps = 0

        for batch in tr_loader:
            P  = as_numpy_1(batch["points"])
            Lc = as_numpy_1(batch["labels_class"])
            Lp = as_numpy_1(batch["labels_prof"])
            Ii = as_numpy_1(batch["instance_ids"])
            sid = batch.get("scene_id", ["unknown"])[0] if isinstance(batch.get("scene_id"), list) else batch.get("scene_id","unknown")

            # lightweight augs
            if P.shape[0] > 0:
                drop_p = float(np.random.uniform(cfg.aug_point_drop_min, cfg.aug_point_drop_max))
                if drop_p > 0:
                    keep = (np.random.rand(P.shape[0]) > drop_p)
                    if keep.sum() >= 256:
                        P, Lc, Lp, Ii = P[keep], Lc[keep], Lp[keep], Ii[keep]
                s = float(np.random.uniform(cfg.aug_scale_min, cfg.aug_scale_max))
                P = P * s
                if cfg.aug_yaw_only:
                    th = float(np.random.uniform(0, 2*np.pi))
                    R = np.array([[np.cos(th),-np.sin(th),0],
                                  [np.sin(th), np.cos(th),0],
                                  [0,0,1]], np.float32)
                else:
                    ax = np.random.uniform(0, 2*np.pi, size=3)
                    Rx = np.array([[1,0,0],[0,np.cos(ax[0]),-np.sin(ax[0])],[0,np.sin(ax[0]),np.cos(ax[0])]])
                    Ry = np.array([[np.cos(ax[1]),0,np.sin(ax[1])],[0,1,0],[-np.sin(ax[1]),0,np.cos(ax[1])]])
                    Rz = np.array([[np.cos(ax[2]),-np.sin(ax[2]),0],[np.sin(ax[2]),np.cos(ax[2]),0],[0,0,1]])
                    R = (Rz @ Ry @ Rx).astype(np.float32)
                P = (R @ P.T).T
                if cfg.aug_translate_frac > 0 and P.shape[0] > 0:
                    pmin, pmax = P.min(0), P.max(0)
                    tvec = (np.random.uniform(-cfg.aug_translate_frac, cfg.aug_translate_frac, 3).astype(np.float32)) * (pmax - pmin)
                    P = P + tvec

            feats, Yc_np, Yp_np, meta = build_feature_grid(
                P, Lc, Lp, Ii, cfg, scene_id=sid, train_mode=True
            )
            if meta.get("empty", False):
                if cfg.debug_prints: print(f"[SKIP][train] empty {sid}")
                continue

            X = torch.from_numpy(feats).unsqueeze(0).to(device)         # (1,7,D,H,W)
            Yc = torch.from_numpy(Yc_np.astype(np.int64)).unsqueeze(0).to(device)
            Yp = torch.from_numpy(Yp_np.astype(np.int64)).unsqueeze(0).to(device)
            band_t = X[:, 5:6]  # band channel (SURFACE occupancy)

            Cctr_np, Off_np, Moff_np, OffW_np = make_instance_targets_from_ids(
                meta["uc"], Yc_np, meta["inv"], meta["point_instance_ids"],
                target_class_id=instance_class_id, sigma=cfg.center_gauss_sigma_vox
            )
            Cctr = torch.from_numpy(Cctr_np).unsqueeze(0).unsqueeze(1).to(device)
            Off  = torch.from_numpy(Off_np).unsqueeze(0).to(device)
            Moff = torch.from_numpy(Moff_np).unsqueeze(0).unsqueeze(1).to(device)
            OffW = torch.from_numpy(OffW_np).unsqueeze(0).unsqueeze(1).to(device)

            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type=="cuda"):
                logits_c, logits_p, logits_ctr, logits_off, logits_bev = model(X, band=band_t)

                ce_c = cross_entropy_band_weighted(
                    logits_c, Yc, band_t, class_weights_t, band_factor=cfg.band_weight_factor
                )
                dl_c = DiceLoss(ignore_index=255)(logits_c, Yc) if cfg.use_dice else logits_c.new_tensor(0.0)

                if cfg.use_profiles and logits_p is not None and n_profiles > 0:
                    logits_p_masked = mask_prof_logits_illegal(logits_p, Yc, ds_tr.allowed_per_class)
                    valid_prof = (Yp >= 0) & (Yc != 255)
                    if valid_prof.any():
                        lp = logits_p_masked.permute(0,2,3,4,1)[valid_prof]
                        yp = Yp[valid_prof]
                        ce_p = torch.nn.functional.cross_entropy(lp, yp, reduction='mean')
                    else:
                        ce_p = logits_c.new_tensor(0.0)
                else:
                    ce_p = logits_c.new_tensor(0.0)

                center_loss = focal_bce_with_logits(logits_ctr, Cctr,
                                                    alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, reduction="mean")

                if Moff.any():
                    near_mask = (Cctr > 0.05) & Moff
                    if not near_mask.any():
                        near_mask = Moff
                    near_mask_3 = near_mask.expand_as(logits_off)
                    weights_3   = OffW.expand_as(logits_off)
                    l1_abs = torch.abs(logits_off - Off)
                    num = (l1_abs * weights_3)[near_mask_3].sum()
                    den = weights_3[near_mask_3].sum().clamp_min(1.0)
                    off_loss = num / den
                else:
                    off_loss = logits_off.new_tensor(0.0)

                bev_loss = logits_c.new_tensor(0.0)
                if cfg.enable_bev_head and (logits_bev is not None):
                    target_bev_np = build_bev_targets_from_centers(Cctr_np)
                    target_bev = torch.from_numpy(target_bev_np).unsqueeze(0).unsqueeze(0).to(device).clamp_(0,1)
                    bev_loss = focal_bce_with_logits(logits_bev, target_bev,
                                                     alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, reduction="mean") * cfg.bev_loss_w

                loss_sem = cfg.ce_w*(ce_c + ce_p) + cfg.dice_w*dl_c
                loss_inst = cfg.center_loss_w*center_loss + cfg.offset_loss_w*off_loss
                loss = loss_sem + loss_inst + bev_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item()); n_steps += 1
            if cfg.debug_prints:
                ce_p_print = float(ce_p) if (cfg.use_profiles and n_profiles>0) else 0.0
                print(f"[LOSS] {sid} | CEc={float(ce_c):.4f} Dice={float(dl_c):.4f} "
                      f"CEp={ce_p_print:.4f} Ctr={float(center_loss):.4f} Off={float(off_loss):.4f} "
                      f"BEV={float(bev_loss):.4f} -> {float(loss):.4f}")

        # validation
        model.eval()
        with torch.no_grad():
            miou_sum, cnt = 0.0, 0
            for batch in va_loader:
                P  = as_numpy_1(batch["points"])
                Lc = as_numpy_1(batch["labels_class"])
                Lp = as_numpy_1(batch["labels_prof"])
                Ii = as_numpy_1(batch["instance_ids"])
                sid = batch.get("scene_id", ["unknown"])[0] if isinstance(batch.get("scene_id"), list) else batch.get("scene_id","unknown")

                feats, Yc_np, Yp_np, meta = build_feature_grid(
                    P, Lc, Lp, Ii, cfg, scene_id=sid, train_mode=False
                )
                if meta.get("empty", False):  continue

                X = torch.from_numpy(feats).unsqueeze(0).to(device)
                Yc = torch.from_numpy(Yc_np.astype(np.int64)).unsqueeze(0).to(device)

                logits_c, _, _, _, _ = model(X, band=X[:,5:6])
                pred = logits_c.argmax(1)
                valid = (Yc != 255)
                if valid.sum().item() == 0:  continue
                ious = []
                for c in range(n_classes):
                    inter = ((pred==c) & (Yc==c) & valid).sum().item()
                    union = (((pred==c) | (Yc==c)) & valid).sum().item()
                    if union > 0:
                        ious.append(inter/union)
                if ious:
                    miou_sum += sum(ious)/len(ious); cnt += 1
            val_miou = (miou_sum/cnt) if cnt>0 else 0.0

        dt = time.time()-t0
        avg_tr = tr_loss/max(1,n_steps)
        print(f"[E{epoch:03d}] train_loss={avg_tr:.4f}  val_mIoU={val_miou:.3f}  ({dt:.1f}s)")

        # ========= ALWAYS SAVE AT END OF EPOCH =========
        ckpt_dir = make_outdir(os.path.join(cfg.out_dir, cfg.run_name))
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        try:
            print(f"[SAVE DEBUG] Saving checkpoint → {ckpt_path}")
            atomic_torch_save(model.state_dict(), ckpt_path)
            print(f"[SAVE OK] exists? {os.path.exists(ckpt_path)} size={os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else 'NA'} bytes")
        except Exception as e:
            print("[SAVE ERROR]", e)
            traceback.print_exc()

        # meta.json
        try:
            meta_out = {
                "classes": ds_tr.cls2id,
                "profiles": ds_tr.prof2id,
                "allowed_per_class": {int(k): sorted(list(v)) for k, v in ds_tr.allowed_per_class.items()},
                "instance_class": ds_tr.instance_class,
                "voxel_size": cfg.voxel_size_m,
                "grid_size": cfg.grid_size,
                "feature_channels": ["x_rel","y_rel","z_rel","ones","n_norm","band(surface)","curv"],
                "auto_fit_voxel_size": getattr(cfg, "auto_fit_voxel_size", True),
                "enable_bev_head": getattr(cfg, "enable_bev_head", True),
                "use_profiles": getattr(cfg, "use_profiles", True),
            }
            meta_path = os.path.join(ckpt_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_out, f, indent=2)
            print(f"[META] wrote {meta_path}")
        except Exception as e:
            print("[META ERROR]", e)
            traceback.print_exc()

        if epoch == 1 or val_miou > best_val:
            best_val = val_miou
            best_path = os.path.join(ckpt_dir, "best.pth")
            try:
                atomic_torch_save(model.state_dict(), best_path)
                print(f"[BEST] val_mIoU={val_miou:.3f} -> {best_path}")
            except Exception as e:
                print("[BEST SAVE ERROR]", e)
                traceback.print_exc()

        sys.stdout.flush()

if __name__ == "__main__":
    train("cuda:0")
