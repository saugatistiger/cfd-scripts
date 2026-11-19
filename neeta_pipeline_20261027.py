#!/usr/bin/env python3
"""Bed-surface pipeline for one decomposed OpenFOAM case. Combines
neeta_vis.20261012.py and build_neeta_vis_20261013.py, and adds the cell-centre
step they both depend on.

Steps (all run by default, in this order):

  centres  processor*/<t>/C, Cx, Cy, Cz via
           `postProcess -func writeCellCentres -time <t> -parallel`.
           Skipped if every processor already has C. <t> is 0 if it exists,
           otherwise the earliest time directory (restart segments often
           have no 0/).
  extract  x,y,z (cell centres) + neeta + tau_vis + Us + voidfraction per
           cell, per timestep
           -> <CFD>/postProcessing/neeta_coords/neeta_{t}.csv.gz
           (tau_vis kept on neeta=1 cells only; Us, voidfraction on every cell)
  vtk      bed-surface height field per timestep, with tau_vis, Us and
           voidfraction of the bed cell, and the depth-integrated sediment flux
           q_sed = sum over the column of (1 - voidfraction) * Us * dy
                           -> <CFD>/vtk_raw/neeta_bed_{n}.vtk
           plus the cells within NEAR_BED_BELOW..NEAR_BED_ABOVE of the bed,
           as points with Us, voidfraction, sed_flux = (1 - voidfraction) * Us,
           neeta and height above bed (binary VTK)
                           -> <CFD>/vtk_raw/near_bed_{n}.vtk

The case directory is asked for at startup, so nothing needs editing:

    python3 neeta_pipeline_20261027.py
    Path to decomposed CFD case: /work/e773/e773/ss1604/20260820/30_square/SIM009_us8.896/CFD_22

Inside an sbatch job there is no keyboard, so pass it instead:

    python3 neeta_pipeline_20261027.py --cfd /work/.../CFD_22
    python3 neeta_pipeline_20261027.py --cfd /work/.../CFD_22 --steps extract,vtk
    python3 neeta_pipeline_20261027.py --cfd /work/.../CFD_22 --times 10 10.7

The centres step needs OpenFOAM loaded (postProcess on PATH) and must run where
MPI can launch: inside a Slurm job it uses `srun -n <nprocs>`, otherwise
`mpirun -np <nprocs>`. Override with e.g. MPI_LAUNCHER="srun --hint=nomultithread -n 64".
"""
import argparse
import gzip
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu

# Each worker holds a full 1.48M x 8 array + its CSV text (a few hundred MB),
# so os.cpu_count() (256 on ARCHER2) gets workers OOM-killed. Override with
# the N_WORKERS env var, e.g. N_WORKERS=64 inside a Slurm job.
N_WORKERS = int(os.environ.get("N_WORKERS", 16))

HEADER = "x,y,z,neeta,tau_x,tau_y,tau_z,tau_mag,Us_x,Us_y,Us_z,Us_mag,voidfraction"

NUM_RE = re.compile(rb"-?[0-9.]+(?:[eE][+-]?[0-9]+)?")
TIME_RE = re.compile(r"[0-9]+(\.[0-9]+)?")

ROUND_DECIMALS = 3  # merges floating-point-noisy (x,z) duplicates across y-layers

# DEM.in: xlo/xhi/zlo/zhi -- real sediment/bed domain, used to drop the
# far-field CFD-mesh-only footprint columns that never carry neeta data.
DOMAIN_X = (0.0, 200.0)
DOMAIN_Z = (-50.0, 150.0)

MAX_EDGE = 8.0  # cm; drops Delaunay triangles bridging the pier hole

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

# Near-bed slab: every cell whose centre lies between NEAR_BED_BELOW under and
# NEAR_BED_ABOVE over the (corrected) bed height of its column is written to
# near_bed_{n}.vtk with its Us, to follow the sediment motion in and around
# the scour hole. Set WRITE_NEAR_BED = False to skip (saves disk space).
WRITE_NEAR_BED = True
NEAR_BED_BELOW = 2.0  # cm
NEAR_BED_ABOVE = 2.0  # cm


# ---------------------------------------------------------------------------
# Case directory
# ---------------------------------------------------------------------------

def clean_path(s: str) -> Path:
    """Tolerates pasted quotes, a trailing '>' or '/', and ~."""
    s = s.strip().strip("'\"").rstrip(">").strip()
    return Path(os.path.expanduser(s))


def check_case(cfd: Path):
    """Returns an error message, or None if cfd looks like a decomposed case."""
    if not cfd.is_dir():
        return f"{cfd} is not a directory"
    if not (cfd / "processor0").is_dir():
        return f"no processor0/ in {cfd} (is the case decomposed?)"
    return None


def ask_case(given):
    if given:
        cfd = clean_path(given)
        err = check_case(cfd)
        if err:
            sys.exit(f"ERROR: {err}")
        return cfd.resolve()
    while True:
        try:
            cfd = clean_path(input("Path to decomposed CFD case: "))
        except EOFError:
            sys.exit("\nERROR: nothing to read the path from (batch job?) -- pass it with --cfd PATH")
        err = check_case(cfd)
        if err is None:
            return cfd.resolve()
        print(f"  {err}, try again", flush=True)


def count_processors(cfd: Path) -> int:
    n = 0
    while (cfd / f"processor{n}").is_dir():
        n += 1
    return n


def list_timesteps(cfd: Path):
    times = [d.name for d in (cfd / "processor0").iterdir()
             if d.is_dir() and TIME_RE.fullmatch(d.name)]
    times.sort(key=float)
    return times


# ---------------------------------------------------------------------------
# Step: centres
# ---------------------------------------------------------------------------

def find_centres_time(cfd: Path, n_procs: int):
    """First time directory (0 preferred) in which every processor has C."""
    times = list_timesteps(cfd)
    if "0" in times:
        times.remove("0")
        times.insert(0, "0")
    for t in times:
        if all((cfd / f"processor{p}" / t / "C").is_file() for p in range(n_procs)):
            return t
    return None


def mpi_launcher(n_procs: int):
    env = os.environ.get("MPI_LAUNCHER")
    if env:
        return shlex.split(env)
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        return ["srun", "-n", str(n_procs)]
    return ["mpirun", "-np", str(n_procs)]


def write_cell_centres(cfd: Path, n_procs: int):
    t = find_centres_time(cfd, n_procs)
    if t is not None:
        print(f"Cell centres already present in processor*/{t}/C -- skipping", flush=True)
        return

    times = list_timesteps(cfd)
    if not times:
        sys.exit(f"ERROR: no time directories in {cfd / 'processor0'}")
    t = "0" if "0" in times else times[0]
    cmd = mpi_launcher(n_procs) + ["postProcess", "-func", "writeCellCentres",
                                   "-time", t, "-parallel"]
    print(f"Writing C, Cx, Cy, Cz into processor*/{t}/ ({n_procs} ranks):\n  {shlex.join(cmd)}",
          flush=True)
    if shutil.which("postProcess") is None:
        sys.exit("ERROR: postProcess not found -- load OpenFOAM first "
                 "(e.g. `module load openfoam` + source its bashrc), or run the command above yourself")

    log = cfd / "log.writeCellCentres"
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=cfd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or find_centres_time(cfd, n_procs) is None:
        sys.exit(f"ERROR: writeCellCentres failed (exit {rc}), see {log}")
    print(f"  done (log: {log})", flush=True)


# ---------------------------------------------------------------------------
# Step: extract   (was neeta_vis.20261012.py)
# ---------------------------------------------------------------------------

def read_internal_field(path: Path, ncomp: int, ncells: int) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    marker = b"internalField"
    start = data.index(marker)
    line_end = data.index(b"\n", start)
    header = data[start:line_end]
    if b"uniform" in header and b"nonuniform" not in header:
        nums = NUM_RE.findall(header.split(b"uniform", 1)[1])
        value = np.array(nums, dtype=np.float64)
        if ncomp > 1:
            return np.tile(value.reshape(1, ncomp), (ncells, 1))
        return np.full(ncells, value[0], dtype=np.float64)
    open_paren = data.index(b"(", start)
    close_paren = data.index(b"\n)", open_paren)
    body = data[open_paren + 1 : close_paren]
    nums = NUM_RE.findall(body)
    arr = np.array(nums, dtype=np.float64)
    if ncomp > 1:
        arr = arr.reshape(-1, ncomp)
    if arr.shape[0] != ncells:
        raise ValueError(f"{path}: expected {ncells} entries, parsed {arr.shape[0]}")
    return arr


def count_cells(path: Path) -> int:
    with open(path, "rb") as f:
        data = f.read()
    start = data.index(b"internalField")
    open_paren = data.index(b"(", start)
    count_line = data[start:open_paren]
    return int(re.search(rb"[0-9]+", count_line.split(b"nonuniform")[-1]).group())


def load_cell_centres(cfd: Path, n_procs: int):
    t = find_centres_time(cfd, n_procs)
    if t is None:
        sys.exit("ERROR: no processor*/<time>/C found -- run the centres step first")
    print(f"  reading processor*/{t}/C", flush=True)
    centres = []
    counts = []
    for p in range(n_procs):
        cpath = cfd / f"processor{p}" / t / "C"
        n = count_cells(cpath)
        c = read_internal_field(cpath, ncomp=3, ncells=n)
        centres.append(c)
        counts.append(n)
    return np.concatenate(centres, axis=0), counts


# --- worker-side globals, set once per process by the initializer ---
# (passed explicitly rather than read from module globals, so this also works
# with the spawn/forkserver start methods)
_cfd = None
_out_dir = None
_xyz = None
_counts = None


def _init_worker(cfd, out_dir, xyz, counts):
    global _cfd, _out_dir, _xyz, _counts
    _cfd = cfd
    _out_dir = out_dir
    _xyz = xyz
    _counts = counts


def process_timestep(t: str):
    out_path = _out_dir / f"neeta_{t}.csv.gz"
    if out_path.exists():
        # Older files lack tau_vis and/or Us -> regenerate those.
        with gzip.open(out_path, "rt") as f:
            if f.readline().strip() == HEADER:
                return t, "skipped"

    neeta_parts = []
    for p in range(len(_counts)):
        fpath = _cfd / f"processor{p}" / t / "neeta"
        neeta_parts.append(read_internal_field(fpath, ncomp=1, ncells=_counts[p]))
    neeta = np.concatenate(neeta_parts, axis=0)

    if neeta.shape[0] != _xyz.shape[0]:
        raise ValueError(f"t={t}: cell count mismatch ({neeta.shape[0]} vs {_xyz.shape[0]})")

    tau_parts = []
    for p in range(len(_counts)):
        fpath = _cfd / f"processor{p}" / t / "tau_vis"
        tau_parts.append(read_internal_field(fpath, ncomp=3, ncells=_counts[p]))
    tau = np.concatenate(tau_parts, axis=0)

    # Keep tau_vis only on the cells flagged neeta=1 (the bed interface).
    tau[neeta != 1] = 0.0
    tau_mag = np.linalg.norm(tau, axis=1)

    # Us and voidfraction on every cell: the motion above/below the interface
    # matters too, and together they give the sediment flux.
    def read_all(field, ncomp):
        if not (_cfd / "processor0" / t / field).is_file():
            raise FileNotFoundError(f"no {field} field in processor0/{t}")
        return np.concatenate([read_internal_field(_cfd / f"processor{p}" / t / field,
                                                   ncomp=ncomp, ncells=_counts[p])
                               for p in range(len(_counts))], axis=0)

    us = read_all("Us", 3)
    us_mag = np.linalg.norm(us, axis=1)
    eps = read_all("voidfraction", 1)

    out = np.column_stack([_xyz, neeta, tau, tau_mag, us, us_mag, eps])
    with gzip.open(out_path, "wt") as f:
        f.write(HEADER + "\n")
        np.savetxt(f, out, delimiter=",", fmt="%.6g")

    return t, "done"


def run_extract(cfd: Path, n_procs: int, csv_dir: Path, only):
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("Loading static cell centres (once)...", flush=True)
    xyz, counts = load_cell_centres(cfd, n_procs)
    print(f"  {xyz.shape[0]} cells total", flush=True)

    # time dirs without a neeta field (e.g. 0/ holding only C) are skipped
    times = [t for t in list_timesteps(cfd) if (cfd / "processor0" / t / "neeta").is_file()]
    print(f"Found {len(times)} timesteps with neeta", flush=True)

    todo = [t for t in times if not only or t in only]

    print(f"Processing {len(todo)} timesteps with {N_WORKERS} workers...", flush=True)

    done_count = 0
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, initializer=_init_worker, initargs=(cfd, csv_dir, xyz, counts)
    ) as ex:
        futures = {ex.submit(process_timestep, t): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                print(f"  [ERROR] t={t}: {e}", flush=True)
                continue
            done_count += 1
            if done_count % 20 == 0 or done_count == len(todo):
                print(f"  [{done_count}/{len(todo)}] t={t} ({status})", flush=True)


# ---------------------------------------------------------------------------
# Step: vtk   (was build_neeta_vis_20261013.py)
# ---------------------------------------------------------------------------

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


def load_footprint(csv_dir: Path):
    """Static mesh -> derive footprint columns once from any timestep CSV."""
    ref = next(csv_dir.glob("neeta_*.csv.gz"))
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
    """Returns (height, bed, has, q_sed, cells):
      height  bed height per column (NaN where no surface was found)
      bed     {"tau", "Us", "voidfraction"}: (n_footprint, ncomp) values of the
              top neeta=1 cell that defines the height
      has     {field: bool}: False if the CSV predates that field's columns
              (tau/Us then 0, voidfraction 1 = no sediment)
      q_sed   (n_footprint, 3) depth-integrated sediment flux per unit width,
              sum over the column of (1 - voidfraction) * Us * cell thickness
      cells   per-cell xyz, column index (-1 = outside footprint), Us,
              voidfraction and neeta, for the near-bed slab."""
    with gzip.open(csv_path, "rt") as f:
        names = f.readline().strip().split(",")
        data = np.loadtxt(f, delimiter=",", ndmin=2)
    col_of = {name: i for i, name in enumerate(names)}

    def vector(prefix):
        keys = [f"{prefix}_{k}" for k in "xyz"]
        if all(k in col_of for k in keys):
            return data[:, [col_of[k] for k in keys]], True
        return np.zeros((data.shape[0], 3)), False

    x, y, z, neeta = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    tau_cells, has_tau = vector("tau")
    us_cells, has_us = vector("Us")
    has_eps = "voidfraction" in col_of
    eps_cells = data[:, col_of["voidfraction"]] if has_eps else np.ones(data.shape[0])
    mask = neeta == 1

    xz = np.round(np.column_stack([x[mask], z[mask]]), ROUND_DECIMALS)
    col_idx = np.array([footprint_lookup_dec.get(tuple(p), -1) for p in xz], dtype=np.int64)
    yv = y[mask]

    height = np.full(n_footprint, -np.inf)
    valid = col_idx >= 0
    np.maximum.at(height, col_idx[valid], yv[valid])
    height[~np.isfinite(height)] = np.nan

    # tau, Us, voidfraction from the cell that defines the height (top neeta=1 cell per column)
    tau = np.zeros((n_footprint, 3))
    us = np.zeros((n_footprint, 3))
    eps = np.ones((n_footprint, 1))
    ci, yy = col_idx[valid], yv[valid]
    top = yy == height[ci]
    tau[ci[top]] = tau_cells[mask][valid][top]
    us[ci[top]] = us_cells[mask][valid][top]
    eps[ci[top], 0] = eps_cells[mask][valid][top]

    # neeta=1 only in a column's bottom cell is the pit floor, not the bed
    # surface -> treat as missing so the gap fill uses real neighbours.
    xz_all = np.round(np.column_stack([x, z]), ROUND_DECIMALS)
    col_all = np.array([footprint_lookup_dec.get(tuple(p), -1) for p in xz_all], dtype=np.int64)
    ok = col_all >= 0
    col_ymin = np.full(n_footprint, np.inf)
    np.minimum.at(col_ymin, col_all[ok], y[ok])
    height[height <= col_ymin + 1e-6] = np.nan

    # Sediment flux per unit width: integrate (1 - voidfraction) * Us over each
    # column. Pure fluid (voidfraction 1) and the static bed (Us 0) add nothing,
    # so the whole column can be summed without picking a layer thickness.
    # Cell thickness = mean distance to the cell centres above and below
    # (one-sided at the column ends).
    idx = np.where(ok)[0]
    order = idx[np.lexsort((y[idx], col_all[idx]))]
    c_sorted, y_sorted = col_all[order], y[order]
    gap = np.where(np.diff(c_sorted) == 0, np.diff(y_sorted), np.nan)
    below, above = np.r_[np.nan, gap], np.r_[gap, np.nan]
    thick = np.where(np.isnan(below), above, np.where(np.isnan(above), below, 0.5 * (below + above)))
    thick = np.nan_to_num(thick)
    q_sed = np.zeros((n_footprint, 3))
    np.add.at(q_sed, c_sorted,
              (1.0 - eps_cells[order])[:, None] * us_cells[order] * thick[:, None])

    bed = dict(tau=tau, Us=us, voidfraction=eps)
    has = dict(tau=has_tau, Us=has_us, voidfraction=has_eps)
    cells = dict(xyz=data[:, :3], col=col_all, us=us_cells, eps=eps_cells, neeta=neeta)
    return height, bed, has, q_sed, cells


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


def write_legacy_vtk_polydata(path: Path, points, simplices, tau, us, eps, q_sed, corrected, pier,
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
        f.write("VECTORS Us float\n")
        for ux, uy, uz in us:
            f.write(f"{ux:g} {uy:g} {uz:g}\n")
        f.write("SCALARS Us_mag float 1\nLOOKUP_TABLE default\n")
        for v in np.linalg.norm(us, axis=1):
            f.write(f"{v:g}\n")
        f.write("SCALARS voidfraction float 1\nLOOKUP_TABLE default\n")
        for v in eps.ravel():
            f.write(f"{v:g}\n")
        f.write("VECTORS q_sed float\n")
        for qx, qy, qz in q_sed:
            f.write(f"{qx:g} {qy:g} {qz:g}\n")
        f.write("SCALARS q_sed_mag float 1\nLOOKUP_TABLE default\n")
        for v in np.linalg.norm(q_sed, axis=1):
            f.write(f"{v:g}\n")
        f.write("SCALARS corrected int 1\nLOOKUP_TABLE default\n")
        for c in corrected.astype(int):
            f.write(f"{c}\n")
        f.write("SCALARS pier int 1\nLOOKUP_TABLE default\n")
        for c in pier.astype(int):
            f.write(f"{c}\n")


def write_near_bed_vtk(path: Path, xyz, us, eps, neeta, dy, title="near-bed cells"):
    """Point cloud (one vertex per cell centre), binary legacy VTK: ASCII
    would be ~3x larger for the ~10^5 points per frame. sed_flux is the local
    solid flux density (1 - voidfraction) * Us."""
    n = xyz.shape[0]
    f4 = lambda a: np.ascontiguousarray(a, dtype=">f4").tobytes()
    with open(path, "wb") as f:
        f.write(f"# vtk DataFile Version 2.0\n{title}\nBINARY\nDATASET POLYDATA\n".encode())
        f.write(f"POINTS {n} float\n".encode() + f4(xyz))
        verts = np.column_stack([np.ones(n), np.arange(n)]).astype(">i4")
        f.write(f"\nVERTICES {n} {2 * n}\n".encode() + verts.tobytes())
        f.write(f"\nPOINT_DATA {n}\nVECTORS Us float\n".encode() + f4(us))
        f.write(b"\nSCALARS Us_mag float 1\nLOOKUP_TABLE default\n" + f4(np.linalg.norm(us, axis=1)))
        f.write(b"\nSCALARS voidfraction float 1\nLOOKUP_TABLE default\n" + f4(eps))
        flux = (1.0 - eps)[:, None] * us
        f.write(b"\nVECTORS sed_flux float\n" + f4(flux))
        f.write(b"\nSCALARS sed_flux_mag float 1\nLOOKUP_TABLE default\n" + f4(np.linalg.norm(flux, axis=1)))
        f.write(b"\nSCALARS neeta float 1\nLOOKUP_TABLE default\n" + f4(neeta))
        f.write(b"\nSCALARS height_above_bed float 1\nLOOKUP_TABLE default\n" + f4(dy))
        f.write(b"\n")


# --- vtk worker-side state, set once per process by the initializer ---
_vtk = None


def _init_vtk_worker(static):
    """The harmonic fill holds a SuperLU factorisation, which can't be pickled,
    so each worker builds its own from the shared mesh."""
    global _vtk
    _vtk = dict(static)
    n_footprint = static["footprint"].shape[0]
    n_total = static["mesh_xz"].shape[0]
    _vtk["fill_height"] = (build_harmonic_fill(static["simplices"], n_footprint, n_total)
                           if n_total > n_footprint else None)


def process_vtk(n, t, csv_path: Path):
    s = _vtk
    footprint, neighbours = s["footprint"], s["neighbours"]
    n_footprint = footprint.shape[0]
    n_fill = s["mesh_xz"].shape[0] - n_footprint
    t_str = fmt_time(t)                                    # display/title label only now
    out_path = s["out_dir"] / f"neeta_bed_{n}.vtk"         # sequential index (0,1,2,...) -- output name

    height, bed, has, q_sed, cells = compute_column_height(csv_path, s["footprint_lookup"], n_footprint)
    # Only fills genuine transient gaps now -- the far-field halo and the
    # pier hole are already excluded from `footprint`/`simplices`,
    # so this can no longer paint fake data into either of them.
    nanmask = np.isnan(height)
    if nanmask.any():
        valid_idx = np.where(~nanmask)[0]
        valid_tree = cKDTree(footprint[valid_idx])
        _, nn = valid_tree.query(footprint[nanmask], k=1)
        height[nanmask] = height[valid_idx[nn]]

    spiked = despike(height, neighbours)
    n_spikes = int(spiked.sum())

    # bed-cell values at corrected columns came from a wrong/missing cell ->
    # rebuild them from the neighbouring columns. q_sed integrates the whole
    # column, so it doesn't depend on the detected height and is kept as is.
    corrected = nanmask | spiked
    tau, us, eps = (fill_from_neighbours(bed[k], corrected, neighbours, footprint)
                    for k in ("tau", "Us", "voidfraction"))

    if WRITE_NEAR_BED:
        # measured from the corrected bed, so despiked columns don't shift the slab
        col = cells["col"]
        ok = col >= 0
        dy = np.full(col.shape[0], np.nan)
        dy[ok] = cells["xyz"][ok, 1] - height[col[ok]]
        sel = ok & (dy >= -NEAR_BED_BELOW) & (dy <= NEAR_BED_ABOVE)
        write_near_bed_vtk(s["out_dir"] / f"near_bed_{n}.vtk", cells["xyz"][sel], cells["us"][sel],
                           cells["eps"][sel], cells["neeta"][sel], dy[sel],
                           title=f"near-bed cells t={t_str}")

    if n_fill:
        height = np.r_[height, s["fill_height"](height)]
        tau = np.vstack([tau, np.zeros((n_fill, 3))])  # no bed shear inside the pier
        us = np.vstack([us, np.zeros((n_fill, 3))])    # no sediment inside the pier
        eps = np.vstack([eps, np.ones((n_fill, 1))])
        q_sed = np.vstack([q_sed, np.zeros((n_fill, 3))])
        corrected = np.r_[corrected, np.zeros(n_fill, dtype=bool)]

    mesh_xz = s["mesh_xz"]
    points = np.column_stack([mesh_xz[:, 0], height, mesh_xz[:, 1]])

    write_legacy_vtk_polydata(out_path, points, s["simplices"], tau, us, eps, q_sed, corrected, s["pier"],
                              title=f"neeta bed surface (raw) t={t_str}")
    return out_path.name, t_str, int(nanmask.sum()), n_spikes, has


def run_vtk(csv_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_timesteps(csv_dir)
    if not selected:
        print(f"No neeta_*.csv.gz files found in {csv_dir}", flush=True)
        return

    print("Building static footprint + Delaunay mesh (once)...", flush=True)
    footprint_full, cell_counts = load_footprint(csv_dir)
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
    if n_fill:
        print(f"  pier hole filled with {n_fill} points (x {fill_xz[:, 0].min():.1f}-{fill_xz[:, 0].max():.1f}, "
              f"z {fill_xz[:, 1].min():.1f}-{fill_xz[:, 1].max():.1f} cm)", flush=True)
    print(f"  {n_footprint} footprint columns, {simplices.shape[0]} triangles", flush=True)
    kdtree = cKDTree(footprint)  # despike/tau neighbours: real bed columns only
    _, neighbours = kdtree.query(footprint, k=SPIKE_K + 1)
    neighbours = neighbours[:, 1:]  # exclude self

    print(f"Processing all {len(selected)} timesteps found on disk "
          f"(t={selected[0][0]:g} to {selected[-1][0]:g}) with {N_WORKERS} workers...", flush=True)

    static = dict(out_dir=out_dir, footprint=footprint, footprint_lookup=footprint_lookup,
                  mesh_xz=mesh_xz, simplices=simplices, pier=pier, neighbours=neighbours)
    done_count = 0
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, initializer=_init_vtk_worker, initargs=(static,)
    ) as ex:
        futures = {ex.submit(process_vtk, n, t, csv_path): (n, t, csv_path)
                   for n, (t, csv_path) in enumerate(selected)}
        for fut in as_completed(futures):
            n, t, csv_path = futures[fut]
            try:
                name, t_str, n_gaps, n_spikes, has = fut.result()
            except Exception as e:
                print(f"  [ERROR] t={t:g} ({csv_path.name}): {e}", flush=True)
                continue
            missing = [field for field, ok in has.items() if not ok]
            if missing and n == 0:
                print(f"  WARNING: {csv_path.name} has no {', '.join(missing)} columns -> written as "
                      f"0 (voidfraction 1, so q_sed 0). Re-run the extract step first.", flush=True)
            done_count += 1
            if done_count % 20 == 0 or done_count == len(selected):
                print(f"  [{done_count}/{len(selected)}] {name} (t={t_str}, {n_gaps} gap-filled, {n_spikes} spikes removed)", flush=True)


# ---------------------------------------------------------------------------

STEPS = ("centres", "extract", "vtk")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfd", help="decomposed case directory (prompted for if omitted)")
    ap.add_argument("--steps", default=",".join(STEPS),
                    help=f"comma-separated subset of {','.join(STEPS)} (default: all)")
    ap.add_argument("--times", nargs="+", metavar="T",
                    help="extract only these time directories (default: all)")
    args = ap.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        ap.error(f"unknown step(s) {bad}; choose from {STEPS}")

    cfd = ask_case(args.cfd)
    n_procs = count_processors(cfd)
    csv_dir = cfd / "postProcessing" / "neeta_coords"
    vtk_dir = cfd / "vtk_raw"
    print(f"Case: {cfd} ({n_procs} processor dirs), steps: {', '.join(steps)}", flush=True)

    if "centres" in steps:
        write_cell_centres(cfd, n_procs)
    if "extract" in steps:
        run_extract(cfd, n_procs, csv_dir, args.times)
    if "vtk" in steps:
        run_vtk(csv_dir, vtk_dir)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
