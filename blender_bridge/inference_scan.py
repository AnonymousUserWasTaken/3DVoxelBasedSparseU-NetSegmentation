# blender_bridge/inference_scan.py
import os, sys, json
import bpy
import torch
import torch.nn.functional as F

BLEND_DIR = bpy.path.abspath("//")
if BLEND_DIR not in sys.path: sys.path.insert(0, BLEND_DIR)
sys.path.insert(0, os.path.join(BLEND_DIR, "mlcore"))

from mlcore.model import SparseUNetMultiTask
from mlcore.utils import voxelize, build_sorted_key_index, load_meta
from blender_bridge.io import points_from_object, drop_marker

OBJECT_NAME = "SceneSensor"
CHECKPOINTS_DIR = os.path.join(BLEND_DIR, "checkpoints")
META_PATH = os.path.join(CHECKPOINTS_DIR, "meta.json")
CKPT_FILE = "epoch_120.pth"  # pick your best

@torch.no_grad()
def forward_once(model, device, pts, feat_dim, voxel_size):
    uc, inv = voxelize(pts, voxel_size)
    _, sk, ip = build_sorted_key_index(uc)
    feats = torch.ones((uc.size(0), feat_dim), dtype=torch.float32)
    uc, feats, sk, ip = uc.to(device), feats.to(device), sk.to(device), ip.to(device)
    sem_logits, cls_logits, embed, kp_preds = model(uc, feats, sk, ip)
    sem_vox = sem_logits.argmax(dim=1).cpu()
    sem_pts = sem_vox[inv]
    probs   = F.softmax(cls_logits, dim=-1).cpu()
    return sem_pts, sem_vox, probs, kp_preds.cpu()

def mark_top_clusters(points, labels_per_point, classes=None, max_markers=3):
    pts = points.float()
    labs = labels_per_point.long()
    centroids = []
    for lab in labs.unique():
        mask = labs == lab
        count = int(mask.sum().item())
        if count == 0: continue
        cen = pts[mask].mean(0)
        centroids.append((count, int(lab.item()), cen))
    centroids.sort(reverse=True, key=lambda x: x[0])
    for i,(count, lab, cen) in enumerate(centroids[:max_markers]):
        name = f"PredCluster_label{lab}_n{count}"
        drop_marker(name, cen)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_meta(META_PATH)
    feat_dim   = int(meta["feat_dim"])
    voxel_size = float(meta["voxel_size"])
    num_sem    = int(meta["num_semantic"])
    num_obj    = int(meta["num_obj"])

    model = SparseUNetMultiTask(base_channels=feat_dim,
                                num_semantic=num_sem,
                                num_obj=num_obj,
                                num_joints=int(meta["num_joints"]))
    state = torch.load(os.path.join(CHECKPOINTS_DIR, CKPT_FILE), map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    pts = points_from_object(OBJECT_NAME)
    sem_pts, sem_vox, probs, kps = forward_once(model, device, pts, feat_dim, voxel_size)

    # Print class probs
    classes = meta.get("classes", [f"class_{i}" for i in range(num_obj)])
    print({c: float(p) for c, p in zip(classes, probs)})

    # Drop markers for largest predicted voxel labels
    mark_top_clusters(pts, sem_pts, classes=classes, max_markers=3)
    print("[INFO] Dropped cluster markers.")

if __name__ == "__main__":
    main()
