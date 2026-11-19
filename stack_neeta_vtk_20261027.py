#!/usr/bin/env python3
"""Stack the bed-surface VTKs of a run's restart segments into one sequence.

A run is split over segment folders (CFD, CFD_11, CFD_22, ...), each with its
own vtk_raw/neeta_bed_{n}.vtk written by neeta_pipeline_20261027.py. The
folder names say nothing about the order, so segments are sorted by their
first write time.

Restart overlap: a later segment supersedes an earlier one from its first time
onward (a restart can begin at or before the previous segment's last write),
as in stack_scour_depth_20261013.py.

Output, in <run>/vtk_stacked/:
    neeta_bed_{i}.vtk       i = 0,1,2,... in time order; relative symlinks to the
                            segment files (--copy for real copies)
    neeta_bed.vtk.series    real times for ParaView -- open this file instead of
                            the .vtk group and the time slider shows t, not i
    stacked_times.csv       index, time, segment, source
    near_bed_{i}.vtk        near-bed cells with Us (same frames), and
    near_bed.vtk.series     its ParaView time file -- only if the segments have them

and the scour depth time series of the whole run (replaces
extract_scour_depth_20261013.py + stack_scour_depth_20261013.py), in <run>/:
    scour_depth_<case>.csv  index, time, y_min, x_at_min, z_at_min, scour_depth, segment
                            e.g. scour_depth_30_circle.csv

y_min is the lowest bed point of each frame (the deepest point of the scour
hole); pier-fill points (pier == 1) are interpolated, not real bed, so they are
excluded. scour_depth = Y_REF - y_min, with Y_REF the median bed height of the
first stacked frame (flat bed at t=0) unless given with --yref. Because the
frames are stacked first, every segment is measured from the same flat-bed
reference, not from its own (already scoured) first frame.

Times come from the segment's postProcessing/neeta_coords/neeta_{t}.csv.gz names
(exact), or from the t= in the VTK title (0.1 precision) if those don't match.

    python3 stack_neeta_vtk_20261027.py
    Path to run folder (holding CFD, CFD_11, ...) [<current dir>]: /work/e773/e773/ss1604/20260820/30_circle/SIM009_us8.896

Pressing Enter uses the current directory. Folders without a vtk_raw/
(DEM, DEM_BED, Initialize, ...) are ignored.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

OUT_NAME = "vtk_stacked"

EXCLUDE_PIER_FILL = True
DEPTH_HEADER = ["index", "time", "y_min", "x_at_min", "z_at_min", "scour_depth", "segment"]


def clean_path(s: str) -> Path:
    """Tolerates pasted quotes, a trailing '>' or '/', a sentence-ending '.',
    and ~. Empty -> current directory."""
    s = s.strip().strip("'\"").rstrip(">").strip()
    if not s:
        return Path.cwd()
    p = Path(os.path.expanduser(s))
    if not p.exists() and s.endswith(".") and Path(os.path.expanduser(s.rstrip("."))).exists():
        p = Path(os.path.expanduser(s.rstrip(".")))
    return p


def find_segments(run: Path):
    return sorted(d for d in run.iterdir()
                  if d.is_dir() and d.name != OUT_NAME and (d / "vtk_raw").is_dir())


def ask_run(given):
    while True:
        if given:
            run, given = clean_path(given), None
        else:
            try:
                run = clean_path(input(f"Path to run folder (holding CFD, CFD_11, ...) [{Path.cwd()}]: "))
            except EOFError:
                sys.exit("\nERROR: nothing to read the path from -- pass it with --run PATH")
        if not run.is_dir():
            print(f"  {run} is not a directory, try again", flush=True)
        elif not find_segments(run):
            print(f"  no sub-folder of {run} has a vtk_raw/, try again", flush=True)
        else:
            return run.resolve()


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
    order the pipeline used to number the VTKs."""
    pat = re.compile(r"neeta_([0-9.]+)\.csv\.gz$")
    times = []
    for p in csv_dir.glob("neeta_*.csv.gz"):
        m = pat.search(p.name)
        if m:
            times.append(float(m.group(1)))
    return sorted(times)


def title_time(path: Path):
    with open(path) as f:
        f.readline()
        m = re.search(r"t=([0-9.eE+-]+)", f.readline())
    return float(m.group(1)) if m else None


def read_segment(seg: Path):
    """Returns [(time, vtk_path), ...] sorted by time."""
    vtks = discover_vtks(seg / "vtk_raw")
    if not vtks:
        return []
    times = discover_times(seg / "postProcessing" / "neeta_coords")
    if len(times) == len(vtks) and [n for n, _ in vtks] == list(range(len(vtks))):
        pairs = [(times[n], p) for n, p in vtks]
    else:
        print(f"  {seg.name}: {len(times)} CSV times vs {len(vtks)} VTKs -> using t= from VTK titles "
              f"(rounded to 0.1)", flush=True)
        pairs = []
        for _, p in vtks:
            t = title_time(p)
            if t is None:
                print(f"  {seg.name}: no t= in {p.name}, skipped", flush=True)
                continue
            pairs.append((t, p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def stack(run: Path):
    segments = []
    for seg in find_segments(run):
        pairs = read_segment(seg)
        if not pairs:
            print(f"  {seg.name}: no neeta_bed_*.vtk, skipping", flush=True)
            continue
        print(f"  {seg.name:>10}: {len(pairs):5d} VTKs, t = {pairs[0][0]:g} .. {pairs[-1][0]:g}", flush=True)
        segments.append((seg.name, pairs))
    if not segments:
        return []

    segments.sort(key=lambda s: s[1][0][0])
    print(f"  time order: {' -> '.join(name for name, _ in segments)}", flush=True)

    # Later segments override earlier ones from their first time onward.
    stacked = []
    for k, (name, pairs) in enumerate(segments):
        if k + 1 < len(segments):
            nxt_name, nxt = segments[k + 1]
            t_next = nxt[0][0]
            if t_next <= pairs[0][0]:
                raise SystemExit(f"ERROR: {name} and {nxt_name} both start at t={t_next:g} -- "
                                 f"can't tell which run supersedes the other")
            keep = [(t, p) for t, p in pairs if t < t_next - 1e-9]
            if len(keep) < len(pairs):
                print(f"  {name}: {len(pairs) - len(keep)} VTKs at t >= {t_next:g} superseded by "
                      f"{nxt_name}", flush=True)
            pairs = keep
        stacked += [(t, name, p) for t, p in pairs]

    # gaps between segments are worth knowing about (missing restart output)
    for (t0, s0, _), (t1, s1, _) in zip(stacked, stacked[1:]):
        if s0 != s1:
            print(f"  join {s0} -> {s1}: t = {t0:g} -> {t1:g}", flush=True)
    return stacked


def write_output(run: Path, stacked, copy: bool):
    out = run / OUT_NAME
    out.mkdir(exist_ok=True)
    # clear the previous stack so a shorter re-stack leaves no stale frames
    for p in [*out.glob("neeta_bed_*.vtk"), *out.glob("near_bed_*.vtk"), out / "near_bed.vtk.series"]:
        if p.is_symlink() or p.exists():
            p.unlink()

    def place(src, dst):
        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(os.path.relpath(src, out))

    series, near_series = [], []
    with open(out / "stacked_times.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "time", "segment", "source"])
        for i, (t, seg, src) in enumerate(stacked):
            dst = out / f"neeta_bed_{i}.vtk"
            place(src, dst)
            series.append({"name": dst.name, "time": t})
            w.writerow([i, f"{t:g}", seg, os.path.relpath(src, run)])

            # near-bed Us slab written alongside it by the pipeline (same n)
            near = src.with_name(src.name.replace("neeta_bed_", "near_bed_"))
            if near.is_file():
                place(near, out / f"near_bed_{i}.vtk")
                near_series.append({"name": f"near_bed_{i}.vtk", "time": t})

    with open(out / "neeta_bed.vtk.series", "w") as f:
        json.dump({"file-series-version": "1.0", "files": series}, f, indent=1)
    if near_series:
        with open(out / "near_bed.vtk.series", "w") as f:
            json.dump({"file-series-version": "1.0", "files": near_series}, f, indent=1)
        if len(near_series) < len(series):
            print(f"  note: near_bed VTKs for only {len(near_series)} of {len(series)} frames "
                  f"(segments processed before Us was added?)", flush=True)
    return out


def read_bed_vtk(path: Path):
    """Returns (points (n,3), pier mask (n,) or None)."""
    with open(path) as f:
        lines = f.read().splitlines()

    i = next(k for k, s in enumerate(lines) if s.startswith("POINTS"))
    n = int(lines[i].split()[1])
    points = np.array(" ".join(lines[i + 1:i + 1 + n]).split(), dtype=float).reshape(n, 3)

    pier = None
    j = next((k for k, s in enumerate(lines) if s.startswith("SCALARS pier")), None)
    if j is not None:
        start = j + 2  # skip LOOKUP_TABLE line
        pier = np.array(lines[start:start + n], dtype=int).astype(bool)
    return points, pier


def scour_depth(stacked, y_ref):
    """One row per stacked frame: [index, time, y_min, x, z, depth, segment]."""
    if stacked[0][0] > 1e-6 and y_ref is None:
        print(f"  WARNING: first frame is t={stacked[0][0]:g}, not t=0 -- Y_REF from it is an "
              f"already scoured bed. Add the t=0 segment, or pass --yref.", flush=True)
    rows = []
    for i, (t, seg, src) in enumerate(stacked):
        points, pier = read_bed_vtk(src)
        bed = points[~pier] if (EXCLUDE_PIER_FILL and pier is not None) else points

        if y_ref is None:
            y_ref = float(np.median(bed[:, 1]))
            print(f"  Y_REF = {y_ref:g} (median bed height of frame 0, {seg}/{src.name})", flush=True)

        k = int(np.argmin(bed[:, 1]))
        x, y, z = bed[k]
        rows.append([i, t, y, x, z, y_ref - y, seg])

        if (i + 1) % 50 == 0 or i == len(stacked) - 1:
            print(f"  [{i + 1}/{len(stacked)}] t={t:g}  y_min={y:g}  depth={y_ref - y:g}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder holding the CFD segment folders (prompted for if omitted)")
    ap.add_argument("--copy", action="store_true", help="copy the VTKs instead of symlinking them")
    ap.add_argument("--yref", type=float, help="undisturbed bed level in cm (default: median of frame 0)")
    args = ap.parse_args()

    run = ask_run(args.run)
    print(f"Run: {run}", flush=True)
    stacked = stack(run)
    if not stacked:
        sys.exit("ERROR: no VTKs found in any segment")
    out = write_output(run, stacked, args.copy)
    print(f"Wrote {len(stacked)} frames, t = {stacked[0][0]:g} .. {stacked[-1][0]:g} -> {out}", flush=True)
    print(f"  ParaView: open {out / 'neeta_bed.vtk.series'}", flush=True)

    print("Scour depth...", flush=True)
    rows = scour_depth(stacked, args.yref)
    case = run.parent.name  # .../30_circle/SIM009_us8.896 -> 30_circle
    depth_csv = run / f"scour_depth_{case}.csv"
    with open(depth_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DEPTH_HEADER)
        w.writerows(rows)
    deepest = max(rows, key=lambda r: r[5])
    print(f"Wrote {depth_csv}: {len(rows)} rows, max depth = {deepest[5]:g} at t={deepest[1]:g}", flush=True)


if __name__ == "__main__":
    main()
