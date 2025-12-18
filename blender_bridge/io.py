# blender_bridge/io.py
import bpy
import numpy as np
import torch

def find_object_case_insensitive(name: str):
    low = name.lower()
    for obj in bpy.data.objects:
        if obj.name.lower() == low:
            return obj
    return None

def points_from_object(obj_name: str) -> torch.Tensor:
    """World-space vertices from a Blender object (incl. modifiers/GeoNodes)."""
    obj = find_object_case_insensitive(obj_name)
    if obj is None:
        raise RuntimeError(f"Object '{obj_name}' not found")

    deps = bpy.context.evaluated_depsgraph_get()
    eo   = obj.evaluated_get(deps)
    mesh = eo.to_mesh()
    try:
        if len(mesh.vertices) == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts)
        verts = verts.reshape(-1, 3)
        mw = np.array(obj.matrix_world, dtype=np.float32)
        verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1), np.float32)], axis=1)
        verts_w = (verts_h @ mw.T)[:, :3]
        return torch.from_numpy(verts_w)
    finally:
        eo.to_mesh_clear()

def drop_marker(name, location, color=(1,0,0,1), size=0.06):
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'SPHERE'
    empty.empty_display_size = size
    bpy.context.scene.collection.objects.link(empty)
    empty.location = tuple(float(x) for x in location)
    try: empty.color = color
    except Exception: pass
    return empty
