"""Scour depth time series from the bed-surface VTKs written by
build_neeta_vis_20261013.py.

For every vtk_raw/neeta_bed_{n}.vtk of a case, takes the lowest y-coordinate of
the bed surface (the deepest point of the scour hole) and writes one row per
timestep to <CFD>/scour_depth.csv:

    index, time, y_min, x_at_min, z_at_min, scour_depth

scour_depth = Y_REF - y_min. Y_REF is the undisturbed bed level, taken as the
median bed height of the first VTK (flat bed at t=0) unless set explicitly.
Pier-fill points (pier == 1) are interpolated, not real bed, so they are
excluded from the minimum.

A combined file with every case (long format, extra `case` column) is written
next to this script as scour_depth_all_cases.csv.
"""
import csv
import re
from pathlib import Path

import numpy as np

# Same cases / layout as build_neeta_vis_20261013.py
BASE = Path("../20260820")
CASES = [
    "40_circle/SIM009_us8.896/CFD",
]

Y_REF = None  # cm; None -> median bed height of the first VTK
EXCLUDE_PIER_FILL = True

HEADER = ["index", "time", "y_min", "x_at_min", "z_at_min", "scour_depth"]


def discover_vtks(vtk_dir: Path):
    """neeta_bed_{n}.vtk sorted by n (numeric, so 10 comes after 9)."""
    pat = re.compile(r"neeta_bed_(\d+)\.vtk$")
    pairs = []
    for p in vtk_dir.glob("neeta_bed_*.vtk"):
        m = pat.search(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def discover_times(csv_dir: Path):
    """Exact write times from the neeta_{t}.csv.gz files, in the same sorted
    order build_neeta_vis_20261013.py used to number the VTKs."""
    pat = re.compile(r"neeta_([0-9.]+)\.csv\.gz$")
    times = []
    for p in csv_dir.glob("neeta_*.csv.gz"):
        m = pat.search(p.name)
        if m:
            times.append(float(m.group(1)))
    return sorted(times)


def read_bed_vtk(path: Path):
    """Returns (time_from_title or None, points (n,3), pier mask (n,) or None)."""
    with open(path) as f:
        lines = f.read().splitlines()

    m = re.search(r"t=([0-9.eE+-]+)", lines[1])
    t_title = float(m.group(1)) if m else None

    i = next(k for k, s in enumerate(lines) if s.startswith("POINTS"))
    n = int(lines[i].split()[1])
    points = np.array(" ".join(lines[i + 1:i + 1 + n]).split(), dtype=float).reshape(n, 3)

    pier = None
    j = next((k for k, s in enumerate(lines) if s.startswith("SCALARS pier")), None)
    if j is not None:
        start = j + 2  # skip LOOKUP_TABLE line
        pier = np.array(lines[start:start + n], dtype=int).astype(bool)
    return t_title, points, pier


def process_case(cfd: Path):
    vtks = discover_vtks(cfd / "vtk_raw")
    if not vtks:
        print(f"  skipping: no neeta_bed_*.vtk in {cfd / 'vtk_raw'}", flush=True)
        return []

    # Prefer exact times from the source CSVs; the VTK title only keeps 1 decimal.
    times = discover_times(cfd / "postProcessing" / "neeta_coords")
    use_csv_times = len(times) == len(vtks) and [n for n, _ in vtks] == list(range(len(vtks)))
    if not use_csv_times:
        print(f"  note: {len(times)} CSV times vs {len(vtks)} VTKs -> using t= from VTK titles "
              f"(rounded to 0.1)", flush=True)

    rows = []
    y_ref = Y_REF
    for k, (n, path) in enumerate(vtks):
        t_title, points, pier = read_bed_vtk(path)
        bed = points[~pier] if (EXCLUDE_PIER_FILL and pier is not None) else points

        if y_ref is None:
            y_ref = float(np.median(bed[:, 1]))
            print(f"  Y_REF = {y_ref:g} (median bed height of {path.name})", flush=True)

        i_min = int(np.argmin(bed[:, 1]))
        x, y, z = bed[i_min]
        t = times[n] if use_csv_times else t_title
        rows.append([n, t, y, x, z, y_ref - y])

        if (k + 1) % 50 == 0 or k == len(vtks) - 1:
            print(f"  [{k + 1}/{len(vtks)}] t={t:g}  y_min={y:g}  depth={y_ref - y:g}", flush=True)

    out = cfd / "scour_depth.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"  wrote {out}", flush=True)
    return rows


if __name__ == "__main__":
    all_rows = []
    for case in CASES:
        print(f"\n=== {case} ===", flush=True)
        for r in process_case(BASE / case):
            all_rows.append([case] + r)

    if all_rows:
        out = Path(__file__).with_name("scour_depth_all_cases.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case"] + HEADER)
            w.writerows(all_rows)
        print(f"\nwrote {out}", flush=True)
