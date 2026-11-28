"""Build (or verify) the NeuroState epilepsy data package (format: ops repo shared/INTERFACES.md).

Steps, in order: labels.py (labels.csv, meta.csv) -> splits.py (splits/) -> preprocess.py with seizure_cut.py
(windows19/, windows10/, seizure/) -> meta.csv window counts -> PACKAGE_INFO.json -> MANIFEST.sha256.
One recording set (patient) per process, all CPU cores by default.

Build (PC1 only, needs the raw TUH files; --minutes and --missing have no default: team decisions):
    python -m experiments.epilepsy.package --out %NS_DATA% --work %NS_WORKSPACE%\\epilepsy\\work ^
        --minutes 5 --missing drop [--groups tuep_epi,tuep_ctrl,seizure] [--workers 10] [--expect]

Verify a copied package (every PC, after the SSD arrives):
    python -m experiments.epilepsy.package --verify <folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.epilepsy import labels as L
from experiments.epilepsy import preprocess as P
from experiments.epilepsy import seizure_cut as SC
from experiments.epilepsy import splits as S

DIAG_GROUPS = ["tuep_epi", "tuep_ctrl", "tueg_y", "tusz_y", "tuab_train", "tuab_eval", "nigeria", "guinea_bissau"]
ALL_GROUPS = DIAG_GROUPS + ["seizure"]       # "seizure" = stage 1 windows of the TUSZ splits
DEFAULT_SEIZURE_PTP_UV = 1000.0              # stage 1: only extreme artefacts (seizures often exceed 150 uV)


# ----------------------------------------------------------------------------------------------- manifest
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(folder):
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    with open(folder / "MANIFEST.sha256", "w", newline="\n") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(folder).as_posix()}\n")
    return len(files)


def verify(folder) -> list[str]:
    """Problems found (empty list = the copy matches its manifest exactly). No patient codes in the messages
    beyond relative file names, which the caller should not print into shared files."""
    folder = Path(folder)
    man = folder / "MANIFEST.sha256"
    if not man.is_file():
        return ["MANIFEST.sha256 missing"]
    listed = {}
    for line in man.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
    on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"}
    probs = [f"missing: {r}" for r in sorted(set(listed) - on_disk)]
    probs += [f"not in manifest: {r}" for r in sorted(on_disk - set(listed))]
    for rel in sorted(set(listed) & on_disk):
        if sha256(folder / rel) != listed[rel]:
            probs.append(f"checksum differs: {rel}")
    return probs


# ----------------------------------------------------------------------------------------------- workers
def _diag_task(task, loader=None):
    """Preprocess one patient and write windows19 / windows10. Returns aggregate stats (no arrays)."""
    st = P.Settings(**task["settings"])
    res = P.process_patient(task["recordings"], st, kind=task["kind"], loader=loader)
    out = Path(task["out"])
    if res.n_windows:
        if res.x is not None:
            p = out / "windows19" / task["group"] / f"{task['pid']}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, x=res.x, x_ms=res.x_ms, start_s=res.start_s)
        p = out / "windows10" / task["group"] / f"{task['pid']}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, x=res.x10, x_ms=res.x10_ms, start_s=res.start_s)
    return dict(pid=task["pid"], group=task["group"], n_windows=res.n_windows, sfreq_orig=res.sfreq_orig,
                stats=res.stats)


def _seizure_task(task, loader=None):
    x, y, start, st = P.seizure_windows(task["recordings"], task["missing"], task["ptp_uv"], loader=loader)
    if x is not None:
        p = Path(task["out"]) / "seizure" / f"{task['pid']}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, x=x, y=y, start_s=start)
    return dict(pid=task["pid"], group="seizure", n_windows=0 if x is None else len(y), stats=st)


def _run(fn, tasks, workers, loader=None, label=""):
    results, t0 = [], time.time()
    if workers <= 1 or loader is not None:
        for i, t in enumerate(tasks, 1):
            results.append(fn(t, loader))
            _progress(i, len(tasks), t0, label)
        return results
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            _progress(i, len(tasks), t0, label)
    return results


def _progress(i, n, t0, label):
    if i % 25 == 0 or i == n:
        el = time.time() - t0
        print(f"  {label} {i}/{n} patients ({el:.0f} s, ~{el / i * (n - i) / 60:.1f} min left)", flush=True)


# ----------------------------------------------------------------------------------------------- build
def _code_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20,
                           cwd=Path(__file__).resolve().parent)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_tasks(cohort, recs, splits, out, settings: P.Settings, groups, intervals=None,
                seizure_ptp_uv=DEFAULT_SEIZURE_PTP_UV):
    intervals = intervals or {}

    def seiz(path):
        if not str(path).lower().endswith(".edf"):
            return []
        return SC.merge(SC.intervals_for_edf(path) + intervals.get(os.path.normcase(str(path)), []))

    diag, stage1 = [], []
    d = recs[recs.role == "diag"]
    by_pid = {p: g for p, g in d.groupby("pid")}
    for c in cohort.itertuples():
        if c.group not in groups or c.group not in DIAG_GROUPS or c.pid not in by_pid:
            continue
        rs = by_pid[c.pid].sort_values(["ses", "tok"])
        diag.append(dict(pid=c.pid, group=c.group, out=str(out), settings=settings.init_args(),
                         kind="emotiv" if c.group in S.EXTERNAL else "edf",
                         recordings=[dict(path=r.path, seizures=seiz(r.path)) for r in rs.itertuples()]))
    if "seizure" in groups:
        keep = set().union(*(set(splits[k].pid) for k in ("tusz_train", "tusz_dev", "tusz_eval")))
        sz = recs[(recs.role == "seizure") & recs.pid.isin(keep)]
        for pid, rs in sz.groupby("pid"):
            rs = rs.sort_values(["ses", "tok"])
            stage1.append(dict(pid=pid, out=str(out), missing=settings.missing, ptp_uv=seizure_ptp_uv,
                               recordings=[dict(path=r.path, seizures=seiz(r.path)) for r in rs.itertuples()]))
    return diag, stage1


def aggregate(results):
    agg = defaultdict(Counter)
    for r in results:
        agg[r["group"]]["patients"] += 1
        agg[r["group"]]["patients_with_windows"] += int(r["n_windows"] > 0)
        for k, v in r["stats"].items():
            agg[r["group"]][k] += v
    return {g: dict(v) for g, v in sorted(agg.items())}


def run(root, out, work, settings: P.Settings, groups=ALL_GROUPS, workers=None, expect=False,
        intervals_csv=None, seizure_ptp_uv=DEFAULT_SEIZURE_PTP_UV, read_headers=True,
        cohort=None, recs=None, loader=None):
    """Build the package. cohort / recs / loader can be given directly (tests); otherwise labels.py builds them."""
    out, work = Path(out), Path(work)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: a package is written once into an empty folder.")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if cohort is None:
        cohort, recs, conflicts, info = L.build(root)
        print(L.summary(cohort, recs, info))
    else:
        conflicts, info = pd.DataFrame(columns=["pid", "group", "issue"]), {}
    meta = L.make_meta(cohort, recs, read_headers=read_headers)
    L.write(cohort, recs, conflicts, meta, out, work)
    splits = S.make_splits(cohort)
    S.check(splits, cohort)
    S.write(splits, out)
    found = S.counts(splits, cohort)
    mismatches = S.compare(found) if expect else []
    if expect:
        print(f"--expect: {len(mismatches)} mismatch(es) with shared/PLAN.md")
        for m in mismatches:
            print("  " + m)

    intervals = SC.read_interval_table(intervals_csv) if intervals_csv else {}
    diag, stage1 = build_tasks(cohort, recs, splits, out, settings, groups, intervals, seizure_ptp_uv)
    workers = workers or os.cpu_count() or 1
    t1 = time.time()
    res_d = _run(_diag_task, diag, workers, loader, "windows")
    t2 = time.time()
    res_s = _run(_seizure_task, stage1, workers, loader, "seizure")
    t3 = time.time()

    nwin = {r["pid"]: r["n_windows"] for r in res_d}
    sf = {r["pid"]: r["sfreq_orig"] for r in res_d if r["sfreq_orig"]}
    meta["n_windows"] = meta.pid.map(nwin)
    meta["minutes"] = meta.pid.map(lambda p: round(nwin[p] * P.WIN_S / 60, 2) if p in nwin else None)
    meta["sfreq_orig"] = [sf.get(p, v) for p, v in zip(meta.pid, meta.sfreq_orig)]
    meta.sort_values("pid").to_csv(out / "meta.csv", index=False)

    stats = dict(windows=aggregate(res_d), seizure=aggregate(res_s))
    pinfo = dict(fake=False, code_commit=_code_commit(), date=datetime.now().isoformat(timespec="seconds"),
                 groups_built=list(groups), counts_per_group=cohort.group.value_counts().to_dict(),
                 split_counts=found, expect_mismatches=mismatches, preprocessing=settings.as_dict(),
                 seizure_windows=dict(ptp_uv=seizure_ptp_uv, min_overlap=0.5, flat_uv=P.DEFAULT_FLAT_UV),
                 seizure_margin_s=settings.seizure_margin_s, stage1_intervals_file=bool(intervals_csv),
                 stats=stats, workers=workers,
                 seconds=dict(labels_splits=round(t1 - t0, 1), windows=round(t2 - t1, 1), seizure=round(t3 - t2, 1)),
                 n_label_conflicts=int(len(conflicts)))
    with open(out / "PACKAGE_INFO.json", "w") as f:
        json.dump(pinfo, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    n = write_manifest(out)
    print(f"package written: {out} ({n} files in MANIFEST.sha256, {time.time() - t0:.0f} s)")
    return pinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", metavar="FOLDER", help="check a copied package against its MANIFEST.sha256")
    ap.add_argument("--root", default=r"G:\NeuroState")
    ap.add_argument("--out", help="new, empty package folder (NS_DATA)")
    ap.add_argument("--work", help="local working folder (cohort, recordings, conflicts)")
    ap.add_argument("--minutes", type=float, help="clean minutes per patient (required for a build)")
    ap.add_argument("--missing", choices=["drop", "interpolate"], help="files missing an electrode (required)")
    ap.add_argument("--groups", default=",".join(ALL_GROUPS), help=f"comma list of {ALL_GROUPS}")
    ap.add_argument("--ptp-uv", type=float, default=P.DEFAULT_PTP_UV)
    ap.add_argument("--flat-uv", type=float, default=P.DEFAULT_FLAT_UV)
    ap.add_argument("--seizure-ptp-uv", type=float, default=DEFAULT_SEIZURE_PTP_UV)
    ap.add_argument("--intervals", help="CSV path,start_s,stop_s of extra seizure intervals (stage 1 detector)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--expect", action="store_true")
    a = ap.parse_args()
    if a.verify:
        probs = verify(a.verify)
        if probs:
            print(f"VERIFY FAILED: {len(probs)} problem(s)")
            for p in probs[:50]:
                print("  " + p)
            sys.exit(1)
        n = sum(1 for _ in open(Path(a.verify) / "MANIFEST.sha256"))
        print(f"verify OK: {n} files match MANIFEST.sha256")
        return
    if not (a.out and a.work and a.minutes and a.missing):
        ap.error("a build needs --out, --work, --minutes and --missing")
    for p in (a.out, a.work):
        L._guard(p, a.root)
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    bad = set(groups) - set(ALL_GROUPS)
    if bad:
        ap.error(f"unknown groups: {sorted(bad)}")
    st = P.Settings(minutes=a.minutes, missing=a.missing, ptp_uv=a.ptp_uv, flat_uv=a.flat_uv)
    run(a.root, a.out, a.work, st, groups, a.workers, a.expect, a.intervals, a.seizure_ptp_uv)


if __name__ == "__main__":
    main()
