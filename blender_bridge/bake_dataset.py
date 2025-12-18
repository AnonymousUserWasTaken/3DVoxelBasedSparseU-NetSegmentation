# blender_bridge/bake_dataset.py
import os, sys, json
import numpy as np
import bpy
import torch

BLEND_DIR = bpy.path.abspath("//")
if BLEND_DIR not in sys.path: sys.path.insert(0, BLEND_DIR)

# we reuse your tof_helpers (inside Blender)
from tof_helpers import get_instance_points

EXPORT_DIR = os.path.join(BLEND_DIR, "dataset_npz")
CKPT_DIR   = os.path.join(BLEND_DIR, "checkpoints")

def build_dataset_indices(collection_name="TrainingSet"):
    root = bpy.data.collections.get(collection_name)
    if root is None:
        raise RuntimeError(f"Collection '{collection_name}' not found")
    children = list(root.children)
    if not children:
        raise RuntimeError(f"No sub-collections in '{collection_name}'")

    indices = []
    # x=0 full meshes
    for y, obj in enumerate(children[0].objects):
        indices.append((0, y, obj.name, "full"))
    # x>=1 parts/furniture collections
    for x, seg_col in enumerate(children[1:], start=1):
        for y, obj in enumerate(seg_col.objects):
            indices.append((x, y, obj.name, seg_col.name))
    return indices, [c.name for c in children]

def main(tof_object="ToF", training_collection="TrainingSet"):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    idxs, child_names = build_dataset_indices(training_collection)
    class_names = sorted({cls for (_,_,_,cls) in idxs})

    count = 0
    for x, y, objname, cls in idxs:
        seg = get_instance_points(tof_object, x, y, transform=None)  # torch (N,3)
        if seg.numel() == 0:
            continue
        full = None
        if x != 0:
            fp = get_instance_points(tof_object, 0, x-1, transform=None)
            if fp.numel() > 0:
                full = fp

        sample = {
            "coords": seg.numpy(),
            "part_label": np.array(x if x>0 else 0, dtype=np.int32),
            "class": np.array(cls, dtype=object),
        }
        if full is not None:
            sample["full_coords"] = full.numpy()

        path = os.path.join(EXPORT_DIR, f"sample_{count:06d}.npz")
        np.savez_compressed(path, **sample)
        count += 1

    meta = {
        "classes": sorted(class_names),
        "num_obj": len(class_names),
        "num_semantic": int(max(x for x,_,_,_ in idxs) + 1),  # 0..max part id
        "num_joints": 6,
        "feat_dim": 32,
        "voxel_size": 0.05,
        "dataset_dir": os.path.relpath(EXPORT_DIR, BLEND_DIR)
    }
    with open(os.path.join(CKPT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Exported {count} samples to {EXPORT_DIR}")
    print(f"[INFO] Wrote meta.json to {CKPT_DIR}")

if __name__ == "__main__":
    main()
