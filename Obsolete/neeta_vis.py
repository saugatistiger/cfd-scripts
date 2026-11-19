# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:10:04 2026

@author: uqsshre9
"""

#!/usr/bin/env python3
"""Extract x,y,z (static mesh cell centres) + neeta value per cell, per timestep,
plus tau_vis (volVectorField) on the neeta=1 cells. tau_x/tau_y/tau_z/tau_mag are
written as 0 on every cell where neeta != 1.
Parallelized across timesteps using multiprocessing.
"""
import gzip
import re
import sys
from pathlib import Path
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

CFD = Path("../20260820/30_circle/SIM009_us8.896/CFD/")
OUT_DIR = CFD / "postProcessing" / "neeta_coords"
N_PROCS = 64
# Each worker holds a full 1.48M x 8 array + its CSV text (a few hundred MB),
# so os.cpu_count() (256 on ARCHER2) gets workers OOM-killed. Override with
# the N_WORKERS env var, e.g. N_WORKERS=64 inside a Slurm job.
N_WORKERS = int(os.environ.get("N_WORKERS", 16))

HEADER = "x,y,z,neeta,tau_x,tau_y,tau_z,tau_mag"

NUM_RE = re.compile(rb"-?[0-9.]+(?:[eE][+-]?[0-9]+)?")


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


def load_cell_centres():
    centres = []
    counts = []
    for p in range(N_PROCS):
        cpath = CFD / f"processor{p}" / "33.4" / "C"
        n = count_cells(cpath)
        c = read_internal_field(cpath, ncomp=3, ncells=n)
        centres.append(c)
        counts.append(n)
    return np.concatenate(centres, axis=0), counts


def list_timesteps():
    proc0 = CFD / "processor0"
    times = []
    for d in proc0.iterdir():
        if d.is_dir() and re.fullmatch(r"[0-9]+(\.[0-9]+)?", d.name):
            times.append(d.name)
    times.sort(key=float)
    return times


# --- worker-side globals, set once per process by the initializer ---
_xyz = None
_counts = None


def _init_worker(xyz, counts):
    global _xyz, _counts
    _xyz = xyz
    _counts = counts


def process_timestep(t: str):
    out_path = OUT_DIR / f"neeta_{t}.csv.gz"
    if out_path.exists():
        # Older files only have x,y,z,neeta -> regenerate those with tau_vis.
        with gzip.open(out_path, "rt") as f:
            if f.readline().strip() == HEADER:
                return t, "skipped"

    neeta_parts = []
    for p in range(N_PROCS):
        fpath = CFD / f"processor{p}" / t / "neeta"
        neeta_parts.append(read_internal_field(fpath, ncomp=1, ncells=_counts[p]))
    neeta = np.concatenate(neeta_parts, axis=0)

    if neeta.shape[0] != _xyz.shape[0]:
        raise ValueError(f"t={t}: cell count mismatch ({neeta.shape[0]} vs {_xyz.shape[0]})")

    tau_parts = []
    for p in range(N_PROCS):
        fpath = CFD / f"processor{p}" / t / "tau_vis"
        tau_parts.append(read_internal_field(fpath, ncomp=3, ncells=_counts[p]))
    tau = np.concatenate(tau_parts, axis=0)

    # Keep tau_vis only on the cells flagged neeta=1 (the bed interface).
    tau[neeta != 1] = 0.0
    tau_mag = np.linalg.norm(tau, axis=1)

    out = np.column_stack([_xyz, neeta, tau, tau_mag])
    with gzip.open(out_path, "wt") as f:
        f.write(HEADER + "\n")
        np.savetxt(f, out, delimiter=",", fmt="%.6g")

    return t, "done"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading static cell centres (once)...", flush=True)
    xyz, counts = load_cell_centres()
    print(f"  {xyz.shape[0]} cells total", flush=True)

    times = list_timesteps()
    print(f"Found {len(times)} timesteps", flush=True)

    only = sys.argv[1:] if len(sys.argv) > 1 else None
    todo = [t for t in times if not only or t in only]

    print(f"Processing {len(todo)} timesteps with {N_WORKERS} workers...", flush=True)

    done_count = 0
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, initializer=_init_worker, initargs=(xyz, counts)
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

    print("Done.", flush=True)


if __name__ == "__main__":
    print(os.getcwd())
    main()
