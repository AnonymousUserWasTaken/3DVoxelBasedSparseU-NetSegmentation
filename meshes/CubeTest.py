import open3d as o3d
import argparse
import os
import time

def main():
    # 1) Compute defaults in cwd
    cwd = os.getcwd()
    default_in  = os.path.join("C:\\Users\\escob\\Documents\\upbge-0.44-windows-x86_64\\points_raw.ply")
    default_out = os.path.join(cwd, "points_with_normals.ply")

    p = argparse.ArgumentParser(
        description="Load raw PLY, compute normals, write back to project folder")
    p.add_argument("in_ply",  nargs="?", default=default_in,
                   help=f"Input PLY (default: {default_in})")
    p.add_argument("out_ply", nargs="?", default=default_out,
                   help=f"Output PLY w/ normals (default: {default_out})")
    p.add_argument("--radius", type=float, default=0.1,
                   help="Search radius for normals (default: 0.1)")
    p.add_argument("--max_nn", type=int, default=30,
                   help="Max neighbors for normals (default: 30)")
    args = p.parse_args()

    # 2) Read the raw XYZ cloud
    pcd = o3d.io.read_point_cloud(args.in_ply)
    if not pcd.has_points():
        print("❌ No points loaded.")
        return

    # 3) Time the normal estimation
    start = time.perf_counter()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.radius, max_nn=args.max_nn))
    pcd.normalize_normals()
    duration = time.perf_counter() - start
    print(f"⏱ Normal estimation took {duration:.3f} seconds")

    # 4) Write back into project folder
    o3d.io.write_point_cloud(args.out_ply, pcd, write_ascii=True)
    print(f"→ Wrote normals‑baked PLY to\n  {args.out_ply}")

if __name__ == "__main__":
    main()
