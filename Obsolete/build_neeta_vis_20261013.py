import gzip
import re
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu

ROUND_DECIMALS = 3  # merges floating-point-noisy (x,z) duplicates across y-layers

CFD = Path("/work/e773/e773/ss1604/Circle_20260812/Flow_analysis_20261006/CFD/")
CSV_DIR = CFD / "postProcessing" / "neeta_coords"
OUT_DIR = CFD / "vtk_raw"

def fmt_time(t):
    """Match the neeta_{t}.csv.gz naming: integers with no decimal ('1', '20'),
    fractional values with one decimal place ('0.1', '10.7'). Display/title label only --
    actual file lookup comes from discover_timesteps(), so this
    can't cause a mismatch."""
    r = round(t, 1)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:.1f}"


def discover_timesteps(csv_dir: Path):
    """Read actual times off disk instead of predicting them from a fixed grid,
    since CFD write times aren't guaranteed to land on a clean grid (e.g. 179.96)."""
    pat = re.compile(r"neeta_([0-9.]+)\.csv\.gz$")
    pairs = []
    for p in csv_dir.glob("neeta_*.csv.gz"):
        m = pat.search(p.name)
        if m:
            pairs.append((float(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


# DEM.in: xlo/xhi/zlo/zhi -- real sediment/bed domain, used to drop the
# far-field CFD-mesh-only footprint columns that never carry neeta data.
DOMAIN_X = (0.0, 200.0)
DOMAIN_Z = (-50.0, 150.0)

MAX_EDGE = 8.0  # cm; drops Delaunay triangles bridging the pier hole (see module docstring)

# Pier fill: the pier hole (an interior gap enclosed by bed columns) is seeded
# with a grid of FILL_SPACING points so the surface is closed there. Their
# height is a harmonic (Laplace) interpolation of the surrounding bed each
# timestep, tau_vis is 0, and they are tagged with the `pier` scalar.
FILL_PIER = True
FILL_SPACING = 1.5  # cm; matches the ~1.5 cm column spacing around the pier

# Cells in the deep pit around the pier (y ~ -40..-20) aren't vertically
# aligned layer to layer, so after (x,z) rounding each one becomes its own
# 1-2 cell "column". Those orphan columns can only ever report a bottom-of-pit
# height, which shows up as spurious deep neeta points. Real columns have
# 70/90/110 cells, so anything below this is dropped from the footprint.
MIN_CELLS_PER_COLUMN = 10

# Despike: residual single-column outliers left after the bottom-cell check in
# compute_column_height(). Any column deviating from the median of its SPIKE_K nearest neighbours by more than
# SPIKE_TOL is replaced by that median. Iterated because spikes can be adjacent.
SPIKE_K = 12
SPIKE_TOL = 5.0  # cm; genuine scour slopes give < ~3 cm neighbour deviation
SPIKE_ITERS = 3


# tau_vis on corrected columns (gap-filled or despiked heights) is rebuilt from
# neighbouring columns: a frontier fill grows inward from valid neighbours, then
# TAU_RELAX_ITERS Jacobi sweeps relax each corrected value to the mean of its
# neighbours while valid columns stay fixed (discrete Laplace interpolation).
TAU_RELAX_ITERS = 50


def despike(height, neighbours):
    """Returns a mask of the columns whose height was replaced."""
    changed = np.zeros(height.shape[0], dtype=bool)
    for _ in range(SPIKE_ITERS):
        med = np.median(height[neighbours], axis=1)
        bad = np.abs(height - med) > SPIKE_TOL
        if not bad.any():
            break
        height[bad] = med[bad]
        changed |= bad
    return changed


def fill_from_neighbours(values, corrected, neighbours, footprint):
    """Overwrite values[corrected] by iterating over neighbouring columns.
    values: (n, ncomp). Valid (uncorrected) columns are never modified."""
    values = values.copy()
    known = ~corrected
    if known.all():
        return values
    if not known.any():
        values[:] = 0.0
        return values

    # 1) frontier fill: each pass fills columns with >= 1 known neighbour
    #    (mean over known neighbours), then marks them known for the next pass.
    while not known.all():
        todo = np.where(~known)[0]
        nb = neighbours[todo]
        w = known[nb]
        cnt = w.sum(axis=1)
        can = cnt > 0
        if not can.any():
            break
        acc = (values[nb] * w[..., None]).sum(axis=1)
        values[todo[can]] = acc[can] / cnt[can, None]
        known[todo[can]] = True

    # disconnected leftovers (no known neighbour chain) -> nearest valid column
    if not known.all():
        valid_idx = np.where(~corrected)[0]
        _, nn = cKDTree(footprint[valid_idx]).query(footprint[~known], k=1)
        values[~known] = values[valid_idx[nn]]

    # 2) relaxation: corrected values -> mean of all neighbours, valid fixed.
    idx = np.where(corrected)[0]
    nb = neighbours[idx]
    for _ in range(TAU_RELAX_ITERS):
        values[idx] = values[nb].mean(axis=1)
    return values


def load_footprint():
    """Static mesh -> derive footprint columns once from any timestep CSV."""
    ref = next(CSV_DIR.glob("neeta_*.csv.gz"))
    with gzip.open(ref, "rt") as f:
        f.readline()
        data = np.loadtxt(f, delimiter=",")
    x, z = data[:, 0], data[:, 2]
    xz = np.round(np.column_stack([x, z]), ROUND_DECIMALS)
    footprint, inverse = np.unique(xz, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse, minlength=footprint.shape[0])
    return footprint, counts


def compute_column_height(csv_path: Path, footprint_lookup_dec, n_footprint):
    """Returns (height, tau): bed height per column (NaN where no surface was
    found) and tau_vis (n_footprint, 3) taken from that same top neeta=1 cell.
    tau is zero if the CSV predates the tau_x,tau_y,tau_z columns."""
    with gzip.open(csv_path, "rt") as f:
        f.readline()
        data = np.loadtxt(f, delimiter=",")
    x, y, z, neeta = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    has_tau = data.shape[1] >= 7
    tau_cells = data[:, 4:7] if has_tau else np.zeros((data.shape[0], 3))
    mask = neeta == 1

    xz = np.round(np.column_stack([x[mask], z[mask]]), ROUND_DECIMALS)
    col_idx = np.array([footprint_lookup_dec.get(tuple(p), -1) for p in xz], dtype=np.int64)
    yv = y[mask]

    height = np.full(n_footprint, -np.inf)
    valid = col_idx >= 0
    np.maximum.at(height, col_idx[valid], yv[valid])
    height[~np.isfinite(height)] = np.nan

    # tau from the cell that defines the height (top neeta=1 cell per column)
    tau = np.zeros((n_footprint, 3))
    ci, yy, tt = col_idx[valid], yv[valid], tau_cells[mask][valid]
    top = yy == height[ci]
    tau[ci[top]] = tt[top]

    # neeta=1 only in a column's bottom cell is the pit floor, not the bed
    # surface -> treat as missing so the gap fill uses real neighbours.
    xz_all = np.round(np.column_stack([x, z]), ROUND_DECIMALS)
    col_all = np.array([footprint_lookup_dec.get(tuple(p), -1) for p in xz_all], dtype=np.int64)
    ok = col_all >= 0
    col_ymin = np.full(n_footprint, np.inf)
    np.minimum.at(col_ymin, col_all[ok], y[ok])
    height[height <= col_ymin + 1e-6] = np.nan
    return height, tau, has_tau


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


def pier_fill_points(footprint):
    """Grid points covering interior holes of the footprint (the pier).
    A hole is the set of gap-bridging Delaunay triangles (max edge > MAX_EDGE)
    that touch no convex-hull vertex, so concave notches on the outer edge
    of the domain are never filled."""
    tri = Delaunay(footprint)
    s = tri.simplices
    maxedge = np.max([np.linalg.norm(footprint[s[:, i]] - footprint[s[:, (i + 1) % 3]], axis=1)
                      for i in range(3)], axis=0)
    on_hull = np.zeros(footprint.shape[0], dtype=bool)
    on_hull[np.unique(tri.convex_hull)] = True
    hole = (maxedge > MAX_EDGE) & ~on_hull[s].any(axis=1)
    if not hole.any():
        return np.empty((0, 2))

    corners = footprint[s[hole]].reshape(-1, 2)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    gx, gz = np.meshgrid(np.arange(lo[0], hi[0] + 1e-9, FILL_SPACING),
                         np.arange(lo[1], hi[1] + 1e-9, FILL_SPACING))
    grid = np.column_stack([gx.ravel(), gz.ravel()])
    idx = tri.find_simplex(grid)
    inside = idx >= 0
    inside[inside] = hole[idx[inside]]
    grid = grid[inside]
    # keep clear of existing ring columns to avoid slivers
    d, _ = cKDTree(footprint).query(grid, k=1)
    return grid[d > 0.5 * FILL_SPACING]


def build_harmonic_fill(simplices, n_known, n_total):
    """Pre-factorised Laplace solve on the mesh graph: unknown nodes
    (index >= n_known) get the mean of their edge neighbours, known nodes
    are fixed. Returns f(h_known) -> h_unknown."""
    edges = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]])
    adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_total, n_total))
    lap = diags(np.asarray(adj.sum(axis=1)).ravel()) - adj
    lap = lap.tocsr()
    lu = splu(lap[n_known:, n_known:].tocsc())
    b = -lap[n_known:, :n_known]
    return lambda h_known: lu.solve(b @ h_known)


def write_legacy_vtk_polydata(path: Path, points, simplices, tau, corrected, pier,
                              title="neeta bed surface (raw)"):
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
        f.write(f"POINT_DATA {points.shape[0]}\n")
        f.write("VECTORS tau_vis float\n")
        for tx, ty, tz in tau:
            f.write(f"{tx:g} {ty:g} {tz:g}\n")
        f.write("SCALARS tau_mag float 1\nLOOKUP_TABLE default\n")
        for v in np.linalg.norm(tau, axis=1):
            f.write(f"{v:g}\n")
        f.write("SCALARS corrected int 1\nLOOKUP_TABLE default\n")
        for c in corrected.astype(int):
            f.write(f"{c}\n")
        f.write("SCALARS pier int 1\nLOOKUP_TABLE default\n")
        for c in pier.astype(int):
            f.write(f"{c}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building static footprint + Delaunay mesh (once)...", flush=True)
    footprint_full, cell_counts = load_footprint()
    n_full = footprint_full.shape[0]
    in_domain = (
        (footprint_full[:, 0] >= DOMAIN_X[0]) & (footprint_full[:, 0] <= DOMAIN_X[1])
        & (footprint_full[:, 1] >= DOMAIN_Z[0]) & (footprint_full[:, 1] <= DOMAIN_Z[1])
    )
    full_column = cell_counts >= MIN_CELLS_PER_COLUMN
    keep = in_domain & full_column
    footprint = footprint_full[keep]
    n_footprint = footprint.shape[0]
    print(f"  {n_full} CFD footprint columns -> {n_footprint} kept "
          f"(dropped {(~in_domain).sum()} far-field, "
          f"{(in_domain & ~full_column).sum()} orphan pit columns with < {MIN_CELLS_PER_COLUMN} cells)", flush=True)

    footprint_lookup = {tuple(p): i for i, p in enumerate(footprint)}
    fill_xz = pier_fill_points(footprint) if FILL_PIER else np.empty((0, 2))
    n_fill = fill_xz.shape[0]
    mesh_xz = np.vstack([footprint, fill_xz])   # bed columns first, pier fill after
    simplices, adjacency, _ = build_mesh_topology_masked(mesh_xz)
    pier = np.r_[np.zeros(n_footprint, dtype=bool), np.ones(n_fill, dtype=bool)]
    fill_height = build_harmonic_fill(simplices, n_footprint, mesh_xz.shape[0]) if n_fill else None
    if n_fill:
        print(f"  pier hole filled with {n_fill} points (x {fill_xz[:, 0].min():.1f}-{fill_xz[:, 0].max():.1f}, "
              f"z {fill_xz[:, 1].min():.1f}-{fill_xz[:, 1].max():.1f} cm)", flush=True)
    print(f"  {n_footprint} footprint columns, {simplices.shape[0]} triangles", flush=True)
    kdtree = cKDTree(footprint)  # despike/tau neighbours: real bed columns only
    _, neighbours = kdtree.query(footprint, k=SPIKE_K + 1)
    neighbours = neighbours[:, 1:]  # exclude self

    selected = discover_timesteps(CSV_DIR)
    if not selected:
        print(f"No neeta_*.csv.gz files found in {CSV_DIR}", flush=True)
        return
    print(f"Processing all {len(selected)} timesteps found on disk "
          f"(t={selected[0][0]:g} to {selected[-1][0]:g})", flush=True)

    for n, (t, csv_path) in enumerate(selected):
        t_str = fmt_time(t)                                  # display/title label only now
        out_path = OUT_DIR / f"neeta_bed_{n}.vtk"             # sequential index (0,1,2,...) -- output name

        height, tau, has_tau = compute_column_height(csv_path, footprint_lookup, n_footprint)
        if not has_tau and n == 0:
            print(f"  WARNING: {csv_path.name} has no tau_x,tau_y,tau_z columns -> tau_vis written as 0. "
                  f"Re-run extract_neeta_coords.py first.", flush=True)
        # Only fills genuine transient gaps now -- the far-field halo and the
        # pier hole are already excluded from `footprint`/`simplices` above,
        # so this can no longer paint fake data into either of them.
        nanmask = np.isnan(height)
        if nanmask.any():
            valid_idx = np.where(~nanmask)[0]
            valid_tree = cKDTree(footprint[valid_idx])
            _, nn = valid_tree.query(footprint[nanmask], k=1)
            height[nanmask] = height[valid_idx[nn]]

        spiked = despike(height, neighbours)
        n_spikes = int(spiked.sum())

        # tau at corrected columns came from a wrong/missing cell -> rebuild it
        # from the neighbouring columns.
        corrected = nanmask | spiked
        tau = fill_from_neighbours(tau, corrected, neighbours, footprint)

        if n_fill:
            height = np.r_[height, fill_height(height)]
            tau = np.vstack([tau, np.zeros((n_fill, 3))])  # no bed shear inside the pier
            corrected = np.r_[corrected, np.zeros(n_fill, dtype=bool)]

        points = np.column_stack([mesh_xz[:, 0], height, mesh_xz[:, 1]])

        write_legacy_vtk_polydata(out_path, points, simplices, tau, corrected, pier,
                                  title=f"neeta bed surface (raw) t={t_str}")

        if (n + 1) % 20 == 0 or n == len(selected) - 1:
            print(f"  [{n + 1}/{len(selected)}] {out_path.name} (t={t_str}, {nanmask.sum()} gap-filled, {n_spikes} spikes removed)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
