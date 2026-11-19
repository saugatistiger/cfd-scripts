"""Stack the per-run scour_depth.csv files of each case into one time series.

Each case was run in segments (CFD, then the restart CFD1, ...), and
extract_scour_depth_20261013.py writes one <segment>/scour_depth.csv per
segment. This script joins the segments of each case in ascending time and
writes one table per case next to this script:

    scour_depth_30_circle.csv, scour_depth_40_circle.csv,
    scour_depth_30_square.csv, scour_depth_40_square.csv

    index, time, y_min, x_at_min, z_at_min, scour_depth, segment

Segments are ordered by their first write time, not by folder name (e.g. in
these runs CFD1 covers t = 0..32.2 and CFD continues from t = 33.4).

Restart overlap: a later segment supersedes an earlier one from its first time
onward (a restart can begin at or before the previous segment's last write).

Reference level: extract_scour_depth_20261013.py takes Y_REF from the first
VTK *of each segment*, so a restart segment's depth is measured from an
already scoured bed. scour_depth is therefore recomputed for every row as
Y_REF - y_min, with Y_REF recovered from the earliest segment (flat bed) unless
set explicitly below.
"""
import csv
from pathlib import Path

import numpy as np

# Same layout as extract_scour_depth_20261013.py
BASE = Path("../20260820")
CASE_DIR = "SIM009_us8.896"
CASES = ["30_circle", "40_circle", "30_square", "40_square"]
SEGMENTS = ["CFD", "CFD1"]  # any order -- sorted by start time; add "CFD2", ... as needed

Y_REF = None  # cm; None -> Y_REF of the earliest segment (scour_depth + y_min of its first row)

HEADER = ["index", "time", "y_min", "x_at_min", "z_at_min", "scour_depth", "segment"]


def read_segment(path: Path):
    """Returns array of (time, y_min, x_at_min, z_at_min, scour_depth), time-sorted."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    data = np.array([[float(r["time"]), float(r["y_min"]), float(r["x_at_min"]),
                      float(r["z_at_min"]), float(r["scour_depth"])] for r in rows])
    if data.size == 0:
        return data.reshape(0, 5)
    return data[np.argsort(data[:, 0], kind="stable")]


def stack_case(case: str):
    segments = []
    for seg in SEGMENTS:
        path = BASE / case / CASE_DIR / seg / "scour_depth.csv"
        if not path.exists():
            print(f"  {seg}: no {path}, skipping", flush=True)
            continue
        data = read_segment(path)
        if len(data) == 0:
            print(f"  {seg}: empty, skipping", flush=True)
            continue
        print(f"  {seg}: {len(data)} rows, t = {data[0, 0]:g} .. {data[-1, 0]:g}", flush=True)
        segments.append((seg, data))

    if not segments:
        return None
    segments.sort(key=lambda s: s[1][0, 0])
    print(f"  time order: {' -> '.join(seg for seg, _ in segments)}", flush=True)

    y_ref = Y_REF
    if y_ref is None:
        first = segments[0][1][0]
        y_ref = first[4] + first[1]  # scour_depth + y_min = that segment's Y_REF
    print(f"  Y_REF = {y_ref:g}", flush=True)

    # Later segments override earlier ones from their first time onward.
    stacked = []
    for k, (seg, data) in enumerate(segments):
        if k + 1 < len(segments):
            t_next = segments[k + 1][1][0, 0]
            if t_next <= data[0, 0]:
                raise ValueError(
                    f"{case}: {seg} and {segments[k + 1][0]} both start at t={t_next:g} -- "
                    f"can't tell which run supersedes the other")
            keep = data[:, 0] < t_next - 1e-9
            if (~keep).any():
                print(f"  {seg}: {int((~keep).sum())} rows at t >= {t_next:g} superseded by "
                      f"{segments[k + 1][0]}", flush=True)
            data = data[keep]
        for t, y, x, z, _ in data:
            stacked.append([t, y, x, z, y_ref - y, seg])

    stacked.sort(key=lambda r: r[0])
    return [[i] + r for i, r in enumerate(stacked)]


if __name__ == "__main__":
    for case in CASES:
        print(f"\n=== {case} ===", flush=True)
        rows = stack_case(case)
        if rows is None:
            print("  no scour_depth.csv found for any segment", flush=True)
            continue
        out = Path(__file__).with_name(f"scour_depth_{case}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"  wrote {out.name}: {len(rows)} rows, t = {rows[0][1]:g} .. {rows[-1][1]:g}, "
              f"max depth = {max(r[5] for r in rows):g}", flush=True)
