import bpy

# ----------------- Config -----------------
COLLECTION_NAME   = "TrainingSet"
GN_GROUP_NAME     = "instance_index"   # geometry nodes group name
MODIFIER_NAME     = "InstanceIndex"    # modifier name shown on objects
ATTR_NAME         = "instance_index"   # stored attribute name
STORE_DOMAIN      = "INSTANCE"         # try INSTANCE domain first
# ------------------------------------------


def ensure_instance_index_group(name=GN_GROUP_NAME,
                                attr_name=ATTR_NAME,
                                domain=STORE_DOMAIN):
    """Create (or reuse) a Geometry Nodes group that stores Instance Index
    into a named attribute on the INSTANCE domain.
    """
    ng = bpy.data.node_groups.get(name)
    if ng and ng.bl_idname == "GeometryNodeTree":
        return ng

    ng = bpy.data.node_groups.new(name=name, type="GeometryNodeTree")

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    # IO
    n_in  = nodes.new("NodeGroupInput");  n_in.location  = (-400, 0)
    n_out = nodes.new("NodeGroupOutput"); n_out.location = ( 400, 0)
    ng.inputs.new("NodeSocketGeometry",  "Geometry")
    ng.outputs.new("NodeSocketGeometry", "Geometry")

    # Instance Index node
    n_idx = nodes.new("GeometryNodeInputInstanceIndex"); n_idx.location = (-100, 0)

    # Store Named Attribute
    n_store = nodes.new("GeometryNodeStoreNamedAttribute"); n_store.location = (150, 0)
    n_store.inputs["Name"].default_value = attr_name
    n_store.data_type = 'INT'
    # Domain may not support INSTANCE in older builds, so guard it
    try:
        n_store.domain = domain
    except Exception:
        n_store.domain = 'POINT'  # fallback

    # Wires
    links.new(n_in.outputs["Geometry"], n_store.inputs["Geometry"])
    links.new(n_idx.outputs["Instance Index"], n_store.inputs["Value"])
    links.new(n_store.outputs["Geometry"], n_out.inputs["Geometry"])

    return ng


def iter_objects_in_collection(coll, recursive=True):
    """Yield all objects within collection (and nested children if recursive)."""
    for obj in coll.objects:
        yield obj
    if recursive:
        for child in coll.children:
            yield from iter_objects_in_collection(child, recursive=True)


def ensure_modifier(obj, gn_group, mod_name=MODIFIER_NAME):
    """Add a Geometry Nodes modifier using gn_group if not already present."""
    # Only makes sense for mesh-like objects
    if obj.type not in {"MESH", "CURVE", "POINTCLOUD", "GPENCIL"}:
        return False

    for m in obj.modifiers:
        if m.type == 'NODES' and m.node_group == gn_group:
            return False  # already has it

    mod = obj.modifiers.new(name=mod_name, type='NODES')
    mod.node_group = gn_group
    # optional visibility toggles
    mod.show_in_editmode = True
    mod.show_viewport = True
    return True


def main():
    root = bpy.data.collections.get(COLLECTION_NAME)
    if root is None:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found")

    gn_group = ensure_instance_index_group()

    added = 0
    skipped = 0
    total = 0

    for obj in iter_objects_in_collection(root, recursive=True):
        total += 1
        changed = ensure_modifier(obj, gn_group)
        if changed:
            added += 1
            print(f"[ADD] {obj.name}")
        else:
            skipped += 1

    print(f"[DONE] Total objects seen: {total}  | Added: {added}  | Already had: {skipped}")


if __name__ == "__main__":
    main()
