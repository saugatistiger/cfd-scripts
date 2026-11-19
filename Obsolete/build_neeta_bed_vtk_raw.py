#!/usr/bin/env python3
"""Export the RAW (unsmoothed) neeta bed surface as legacy VTK ASCII PolyData
files, one per timestep, written into CFD_0to20/vtk_raw/.

Differences from build_neeta_bed_vtk.py:
  1. No laplacian_smooth() -- height(column) is just max-y-where-neeta==1 per
     footprint column, with only genuine transient gaps (a real bed column
     that momentarily has no neeta==1 row) filled via nearest-neighbor.
  2. Footprint is masked to the actual DEM/bed domain (x:[0,200], z:[-50,150],
     from DEM.in) *before* triangulating. Roughly half the CFD mesh's (x,z)
     footprint sits outside the sediment domain (inlet/outlet development
     zones) and never has a single neeta==1 reading at any timestep -- left
     in, fill_gaps() would nearest-neighbor-paint a fake "bed" across all of
     it. Dropping those columns before triangulation removes that halo
     entirely instead of filling it.
  3. Delaunay triangles that bridge the pier's footprint hole (the CFD mesh
     has zero cells inside the pier -- it's a grid-with-a-hole blockMesh, so
     Delaunay doesn't know to leave that as a hole and instead spans across
     it with abnormally long edges) are dropped via a max-edge-length filter.
     Regular mesh spacing is ~2.8 cm; the gap-bridging triangles all have an
     edge >8 cm (verified: exactly the 26 triangles centred on the pier
     footprint, nothing elsewhere is touched by this cutoff).

Self-contained -- no longer depends on build_neeta_heightfield_stl.py being
present alongside it (load_footprint()/compute_column_height() are inlined
below instead of imported), so this file can be copied on its own to a new
stage directory (CFD_XtoY) and run from there directly.
"""
import gzip
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix

ROUND_DECIMALS = 3  # merges floating-point-noisy (x,z) duplicates across y-layers

CFD = Path("../20260820/30_circle/SIM009_us8.896/CFD/")
CSV_DIR = CFD / "postProcessing" / "neeta_coords"
OUT_DIR = CFD / "vtk_raw"

T_START = 33.4
T_END = 323.16    # inclusive
T_STEP = 0.8    # -> 201 snapshots (20.0, 20.1, 20.2, ..., 40.0)


def fmt_time(t):
    """Match the neeta_{t}.csv.gz naming: integers with no decimal ('1', '20'),
    fractional values with one decimal place ('0.1', '10.7')."""
    r = round(t, 1)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:.1f}"

# DEM.in: xlo/xhi/zlo/zhi -- real sediment/bed domain, used to drop the
# far-field CFD-mesh-only footprint columns that never carry neeta data.
DOMAIN_X = (0.0, 200.0)
DOMAIN_Z = (-50.0, 150.0)

MAX_EDGE = 8.0  # cm; drops Delaunay triangles bridging the pier hole (see module docstring)


def load_footprint():
    """Static mesh -> derive footprint columns once from any timestep CSV."""
    ref = next(CSV_DIR.glob("neeta_*.csv.gz"))
    with gzip.open(ref, "rt") as f:
        f.readline()
        data = np.loadtxt(f, delimiter=",")
    x, z = data[:, 0], data[:, 2]
    xz = np.round(np.column_stack([x, z]), ROUND_DECIMALS)
    footprint, inverse = np.unique(xz, axis=0, return_inverse=True)
    return footprint, inverse.reshape(-1)


def compute_column_height(csv_path: Path, footprint_lookup_dec, n_footprint):
    with gzip.open(csv_path, "rt") as f:
        f.readline()
        data = np.loadtxt(f, delimiter=",")
    x, y, z, neeta = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    mask = neeta == 1

    xz = np.round(np.column_stack([x[mask], z[mask]]), ROUND_DECIMALS)
    col_idx = np.array([footprint_lookup_dec.get(tuple(p), -1) for p in xz], dtype=np.int64)
    yv = y[mask]

    height = np.full(n_footprint, -np.inf)
    valid = col_idx >= 0
    np.maximum.at(height, col_idx[valid], yv[valid])
    height[~np.isfinite(height)] = np.nan
    return height


def build_mesh_topology_masked(footprint):
    """Like build_neeta_heightfield_stl.build_mesh_topology(), but triangulates
    only the masked footprint and drops gap-bridging triangles (see MAX_EDGE)."""
    tri = Delaunay(footprint)
    simplices = tri.simplices

    e1 = np.linalg.norm(footprint[simplices[:, 0]] - footprint[simplices[:, 1]], axis=1)
    e2 = np.linalg.norm(footprint[simplices[:, 1]] - footprint[simplices[:, 2]], axis=1)
    e3 = np.linalg.norm(footprint[simplices[:, 2]] - footprint[simplices[:, 0]], axis=1)
    maxedge = np.maximum(np.maximum(e1, e2), e3)
    dropped = int((maxedge > MAX_EDGE).sum())
    simplices = simplices[maxedge <= MAX_EDGE]
    print(f"  dropped {dropped} gap-bridging triangles (max edge > {MAX_EDGE} cm)", flush=True)

    n = footprint.shape[0]
    edges = np.concatenate([
        simplices[:, [0, 1]], simplices[:, [1, 0]],
        simplices[:, [1, 2]], simplices[:, [2, 1]],
        simplices[:, [2, 0]], simplices[:, [0, 2]],
    ], axis=0)
    edges = np.unique(edges, axis=0)
    row, col = edges[:, 0], edges[:, 1]
    deg = np.bincount(row, minlength=n).astype(np.float64)
    deg[deg == 0] = 1.0
    weights = 1.0 / deg[row]
    adjacency = csr_matrix((weights, (row, col)), shape=(n, n))

    kdtree = cKDTree(footprint)
    return simplices, adjacency, kdtree


def write_legacy_vtk_polydata(path: Path, points, simplices, title="neeta bed surface (raw)"):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {points.shape[0]} float\n")
        for x, y, z in points:
            f.write(f"{x:g} {y:g} {z:g}\n")
        n_polys = simplices.shape[0]
        f.write(f"POLYGONS {n_polys} {n_polys * 4}\n")
        for a, b, c in simplices:
            f.write(f"3 {a} {b} {c}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building static footprint + Delaunay mesh (once)...", flush=True)
    footprint_full, _ = load_footprint()
    n_full = footprint_full.shape[0]
    in_domain = (
        (footprint_full[:, 0] >= DOMAIN_X[0]) & (footprint_full[:, 0] <= DOMAIN_X[1])
        & (footprint_full[:, 1] >= DOMAIN_Z[0]) & (footprint_full[:, 1] <= DOMAIN_Z[1])
    )
    footprint = footprint_full[in_domain]
    n_footprint = footprint.shape[0]
    print(f"  {n_full} CFD footprint columns -> {n_footprint} inside DEM domain "
          f"(dropped {n_full - n_footprint} far-field columns)", flush=True)

    footprint_lookup = {tuple(p): i for i, p in enumerate(footprint)}
    simplices, adjacency, kdtree = build_mesh_topology_masked(footprint)
    print(f"  {n_footprint} footprint columns, {simplices.shape[0]} triangles", flush=True)

    n_steps = round((T_END - T_START) / T_STEP) + 1
    selected = [round(T_START + i * T_STEP, 1) for i in range(n_steps)]
    print(f"Selected {len(selected)} timesteps: {T_START} to {T_END} step {T_STEP}", flush=True)

    for n, t in enumerate(selected):
        t_str = fmt_time(t)
        csv_path = CSV_DIR / f"neeta_{t_str}.csv.gz"       # actual sim time (0.1 s spacing) -- input lookup
        out_path = OUT_DIR / f"neeta_bed_{n}.vtk"           # sequential index (0,1,2,...) -- output name

        height = compute_column_height(csv_path, footprint_lookup, n_footprint)
        # Only fills genuine transient gaps now -- the far-field halo and the
        # pier hole are already excluded from `footprint`/`simplices` above,
        # so this can no longer paint fake data into either of them.
        nanmask = np.isnan(height)
        if nanmask.any():
            valid_idx = np.where(~nanmask)[0]
            valid_tree = cKDTree(footprint[valid_idx])
            _, nn = valid_tree.query(footprint[nanmask], k=1)
            height[nanmask] = height[valid_idx[nn]]

        points = np.column_stack([footprint[:, 0], height, footprint[:, 1]])

        write_legacy_vtk_polydata(out_path, points, simplices, title=f"neeta bed surface (raw) t={t_str}")

        if (n + 1) % 20 == 0 or n == len(selected) - 1:
            print(f"  [{n + 1}/{len(selected)}] {out_path.name} (t={t_str})", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
