"""Seizure removal: drop every window that overlaps a seizure +/- 60 s.

Seizure intervals come from the TUH `.csv_bi` files next to each EDF (TUSZ, TUEP), or later from the stage 1
detector (TUEG Y) as a CSV with columns `path, start_s, stop_s`. preprocess.py calls `window_mask()` per recording
before the minute cap, so a patient still gets the same number of clean minutes.

Report (aggregate only):
    python -m experiments.epilepsy.seizure_cut --recordings <work>/recordings.csv [--intervals detector.csv]
prints, per group, the recordings and patients with seizures and the minutes inside seizure +/- 60 s.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

MARGIN_S = 60.0


def read_csv_bi(path) -> list[tuple[float, float]]:
    """TUH .csv_bi (term-based): rows `channel,start_time,stop_time,label,confidence`; every label other than
    'bckg' is a seizure. Returns merged (start, stop) intervals in seconds."""
    ivs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = [r for r in f if r.strip() and not r.lstrip().startswith("#")]
    for r in csv.reader(rows[1:] if rows and rows[0].lower().startswith("channel") else rows):
        if len(r) >= 4 and r[3].strip().lower() != "bckg":
            try:
                ivs.append((float(r[1]), float(r[2])))
            except ValueError:
                continue
    return merge(ivs)


def merge(ivs):
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def intervals_for_edf(edf_path) -> list[tuple[float, float]]:
    """Seizures of one recording from the sibling .csv_bi, or [] when there is none."""
    p = os.path.splitext(str(edf_path))[0] + ".csv_bi"
    return read_csv_bi(p) if os.path.exists(p) else []


def read_interval_table(path) -> dict[str, list[tuple[float, float]]]:
    """CSV with columns path, start_s, stop_s (e.g. stage 1 detector output) -> {normalised path: intervals}."""
    d = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[os.path.normcase(r["path"])].append((float(r["start_s"]), float(r["stop_s"])))
    return {k: merge(v) for k, v in d.items()}


def window_mask(start_s, intervals, margin_s=MARGIN_S, win_s=4.0) -> np.ndarray:
    """True for every window [start, start + win_s) that overlaps any seizure widened by margin_s on both sides."""
    start = np.asarray(start_s, dtype=float)
    m = np.zeros(len(start), bool)
    for s, e in intervals:
        m |= (start < e + margin_s) & (start + win_s > s - margin_s)
    return m


def window_overlap(start_s, intervals, win_s=4.0) -> np.ndarray:
    """Share of each window inside a seizure (0-1). Intervals should be merged."""
    start = np.asarray(start_s, dtype=float)
    ov = np.zeros(len(start))
    for s, e in intervals:
        ov += np.clip(np.minimum(start + win_s, e) - np.maximum(start, s), 0, None)
    return np.clip(ov / win_s, 0, 1)


def removed_seconds(intervals, duration_s, margin_s=MARGIN_S) -> float:
    """Seconds of a recording inside seizure +/- margin (clipped to the recording)."""
    widened = merge([(max(0.0, s - margin_s), min(duration_s, e + margin_s)) for s, e in intervals])
    return float(sum(max(0.0, e - s) for s, e in widened))


def report(recordings_csv, intervals_csv=None, margin_s=MARGIN_S):
    """Aggregate per group: recordings / patients with seizures, minutes removed. Reads only the label files."""
    extra = read_interval_table(intervals_csv) if intervals_csv else {}
    agg = defaultdict(lambda: dict(recordings=0, recordings_with_seizure=0, patients=set(), patients_with_seizure=set(),
                                   seizures=0, seizure_min=0.0, removed_min=0.0))
    with open(recordings_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("role", "diag") != "diag":
                continue
            a = agg[r["group"]]
            a["recordings"] += 1
            a["patients"].add(r["pid"])
            ivs = merge(intervals_for_edf(r["path"]) + extra.get(os.path.normcase(r["path"]), [])) \
                if r["path"].lower().endswith(".edf") else []
            if ivs:
                dur = float(r.get("duration_s") or max(e for _, e in ivs))
                a["recordings_with_seizure"] += 1
                a["patients_with_seizure"].add(r["pid"])
                a["seizures"] += len(ivs)
                a["seizure_min"] += sum(e - s for s, e in ivs) / 60
                a["removed_min"] += removed_seconds(ivs, dur, margin_s) / 60
    out = {}
    for g, a in sorted(agg.items()):
        out[g] = dict(recordings=a["recordings"], recordings_with_seizure=a["recordings_with_seizure"],
                      patients=len(a["patients"]), patients_with_seizure=len(a["patients_with_seizure"]),
                      seizures=a["seizures"], seizure_minutes=round(a["seizure_min"], 1),
                      minutes_removed_with_margin=round(a["removed_min"], 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", required=True, help="recordings.csv written by labels.py (local, never in git)")
    ap.add_argument("--intervals", help="optional CSV path,start_s,stop_s (stage 1 detector)")
    ap.add_argument("--margin", type=float, default=MARGIN_S)
    a = ap.parse_args()
    res = report(a.recordings, a.intervals, a.margin)
    print(f"{'group':<14}{'recs':>7}{'recs w/ seiz':>13}{'patients':>9}{'pts w/ seiz':>12}{'seizures':>9}"
          f"{'seiz min':>9}{'removed min':>12}")
    for g, d in res.items():
        print(f"{g:<14}{d['recordings']:>7}{d['recordings_with_seizure']:>13}{d['patients']:>9}"
              f"{d['patients_with_seizure']:>12}{d['seizures']:>9}{d['seizure_minutes']:>9}"
              f"{d['minutes_removed_with_margin']:>12}")


if __name__ == "__main__":
    main()
