import gzip
import re
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix

ROUND_DECIMALS = 3  # merges floating-point-noisy (x,z) duplicates across y-layers

CFD = Path("../20260820/30_circle/SIM009_us8.896/CFD/")
CSV_DIR = CFD / "postProcessing" / "neeta_coords"
OUT_DIR = CFD / "vtk_raw"

T_START = 33.4
T_END = 117.7    # inclusive
T_STEP = 0.3      # target spacing between snapshots; nearest available file on
                   # disk is used for each target (CFD write times aren't on a
                   # clean grid, e.g. 179.96, so exact arithmetic match is unsafe)
T_TOL = T_STEP / 2  # max allowed distance between target time and nearest real file


def fmt_time(t):
    """Match the neeta_{t}.csv.gz naming: integers with no decimal ('1', '20'),
    fractional values with one decimal place ('0.1', '10.7'). Display/title label only --
    actual file lookup comes from discover_timesteps()/select_timesteps(), so this
    can't cause a mismatch."""
    r = round(t, 1)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:.1f}"


def discover_timesteps(csv_dir: Path):
    """Read actual times off disk instead of predicting them from T_START/T_STEP,
    since CFD write times aren't guaranteed to land on a clean grid (e.g. 179.96)."""
    pat = re.compile(r"neeta_([0-9.]+)\.csv\.gz$")
    pairs = []
    for p in csv_dir.glob("neeta_*.csv.gz"):
        m = pat.search(p.name)
        if m:
            pairs.append((float(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def select_timesteps(all_steps, t_start, t_end, t_step, tol):
    """Walk targets t_start, t_start+t_step, ... up to t_end and snap each one to
    the nearest real file on disk (within tol). Preserves the T_STEP spacing
    behavior without assuming target times exist verbatim as filenames."""
    if not all_steps:
        return []
    times = np.array([t for t, _ in all_steps])
    targets = np.arange(t_start, t_end + 1e-9, t_step)

    selected = []
    for target in targets:
        idx = np.searchsorted(times, target)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(times)]
        best = min(candidates, key=lambda i: abs(times[i] - target))
        if abs(times[best] - target) <= tol:
            selected.append(all_steps[best])
        else:
            print(f"  WARNING: no file within {tol} of target t={target:.3f}, skipping", flush=True)
    return selected


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

    all_steps = discover_timesteps(CSV_DIR)
    selected = select_timesteps(all_steps, T_START, T_END, T_STEP, T_TOL)
    print(f"Selected {len(selected)} timesteps: target {T_START} to {T_END} step {T_STEP} "
          f"(nearest real file used within tol={T_TOL}, {len(all_steps)} total found on disk)", flush=True)

    for n, (t, csv_path) in enumerate(selected):
        t_str = fmt_time(t)                                  # display/title label only now
        out_path = OUT_DIR / f"neeta_bed_{n}.vtk"             # sequential index (0,1,2,...) -- output name

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
