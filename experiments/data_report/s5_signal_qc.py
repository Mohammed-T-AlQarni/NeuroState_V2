"""Step 5: signal quality on a STRATIFIED SAMPLE of EDFs, read with MNE (read-only).
Strata = dataset x montage folder; every stratum gets at least --min-per-stratum files (or all it has),
the rest is allocated in proportion to size, up to --n files (default 400, >= 200 required by the brief).
For each file: open with MNE, read three 20 s windows (at 10%, 50%, 90% of the length) and the last second, compute per-channel SD and
peak amplitude of the 19 standard electrodes, compare MNE length with the header.

Private output: cache/signal_qc_files.csv. Aggregate output: outputs/s5_signal_qc.json
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict

import numpy as np

import common as C

FLAT_UV = 0.5      # channel SD below this (microvolts) = flat
HIGH_SD_UV = 200   # channel SD above this = very noisy / artefact
EXTREME_UV = 2000  # any sample beyond +-2 mV = extreme amplitude


def sample(rows, n_target, min_per, seed):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["dataset"], r["montage"])].append(r)
    rng = random.Random(seed)
    alloc = {k: min(len(v), min_per) for k, v in strata.items()}
    left = n_target - sum(alloc.values())
    total = sum(len(v) for v in strata.values())
    if left > 0:
        for k, v in strata.items():
            alloc[k] = min(len(v), alloc[k] + int(round(left * len(v) / total)))
    picked = []
    for k, v in sorted(strata.items()):
        picked += rng.sample(v, alloc[k])
    return picked, {f"{a} | {b}": dict(available=len(strata[(a, b)]), sampled=alloc[(a, b)]) for a, b in sorted(strata)}


def qc_file(r, window_s=20.0):
    import mne
    out = dict(dataset=r["dataset"], path=r["path"], montage=r["montage"], ok=0, error="")
    t0 = time.time()
    try:
        raw = mne.io.read_raw_edf(r["path"], preload=False, verbose="error")
    except Exception as e:
        out["error"] = f"open {type(e).__name__}: {str(e)[:80]}"
        return out
    sf = raw.info["sfreq"]
    n = raw.n_times
    out["mne_duration_s"] = round(n / sf, 3)
    out["header_duration_s"] = r["duration_s"]
    try:
        out["duration_mismatch_s"] = round(abs(n / sf - float(r["duration_s"])), 3)
    except ValueError:
        out["duration_mismatch_s"] = ""
    names = {C.norm_label(ch): ch for ch in raw.ch_names}
    picks = []
    for s in C.STD19:
        for alt in ([s] if isinstance(s, str) else s):
            if alt in names:
                picks.append(names[alt])
                break
    out["n_std_channels"] = len(picks)
    if not picks:
        out["error"] = "no standard channels"
        return out
    try:
        w = int(min(n, window_s * sf))
        xs = []
        for frac in (0.1, 0.5, 0.9):   # three windows: near the start, middle, near the end
            start = int(min(max(0, n * frac - w / 2), n - w))
            xs.append(raw.get_data(picks=picks, start=start, stop=start + w))
        x = np.concatenate(xs, axis=1) * 1e6
        last = raw.get_data(picks=picks[:1], start=max(0, n - int(sf)), stop=n)
        out["last_second_ok"] = int(last.shape[1] > 0 and np.isfinite(last).all())
    except Exception as e:
        out["error"] = f"read {type(e).__name__}: {str(e)[:80]}"
        return out
    pk = np.abs(x).max(axis=1)
    sd = np.median([xi.std(axis=1) * 1e6 for xi in xs], axis=0)   # SD per window, median over windows
    out.update(ok=1, sfreq=sf, window_s=round(3 * w / sf, 1), n_flat=int((sd < FLAT_UV).sum()),
               n_high_sd=int((sd > HIGH_SD_UV).sum()), n_extreme=int((pk > EXTREME_UV).sum()),
               n_nonfinite=int((~np.isfinite(x)).any(axis=1).sum()),
               median_sd_uv=round(float(np.median(sd)), 2), max_peak_uv=round(float(pk.max()), 1),
               seconds=round(time.time() - t0, 2))
    return out


def agg(rows):
    ok = [r for r in rows if r["ok"] == 1]
    res = dict(n_sampled=len(rows), n_read_ok=len(ok), n_failed=len(rows) - len(ok),
               errors=C.top(Counter(r["error"].split(":")[0] for r in rows if r["ok"] != 1), 8))
    if not ok:
        return res
    nch = sum(r["n_std_channels"] for r in ok)
    res.update(
        files_with_flat_channel=sum(1 for r in ok if r["n_flat"] > 0),
        pct_channels_flat=round(100 * sum(r["n_flat"] for r in ok) / nch, 2),
        files_all_channels_flat=sum(1 for r in ok if r["n_flat"] == r["n_std_channels"]),
        files_with_high_sd_channel=sum(1 for r in ok if r["n_high_sd"] > 0),
        files_with_extreme_amplitude=sum(1 for r in ok if r["n_extreme"] > 0),
        files_with_nonfinite=sum(1 for r in ok if r["n_nonfinite"] > 0),
        median_channel_sd_uv=C.dist([r["median_sd_uv"] for r in ok]),
        max_peak_uv=C.dist([r["max_peak_uv"] for r in ok]),
        files_duration_mismatch_gt1s=sum(1 for r in ok if r["duration_mismatch_s"] != "" and r["duration_mismatch_s"] > 1),
        files_last_second_unreadable=sum(1 for r in ok if not r.get("last_second_ok", 1)),
        files_missing_some_std_channels=sum(1 for r in ok if r["n_std_channels"] < 19),
        sfreq=C.top(Counter(r["sfreq"] for r in ok), 8),
        seconds_per_file=C.dist([r["seconds"] for r in ok]),
    )
    for k in ("median_channel_sd_uv", "max_peak_uv", "seconds_per_file"):
        res[k].pop("sum", None)
    return res


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 5 signal QC (sample)")
    hdr = [r for r in C.read_csv(P["cache"] / "edf_headers.csv") if r["ok"] == "1" and r["dataset"] in C.MAIN]
    picked, plan = sample(hdr, args.n, args.min_per_stratum, args.seed)
    log(f"  sampling {len(picked)} of {len(hdr)} readable EDFs from {len(plan)} strata (seed {args.seed})")
    rows, t0 = [], time.time()
    for i, r in enumerate(picked, 1):
        rows.append(qc_file(r))
        C.progress(i, len(picked), t0, 25, "QC")
    fields = ["dataset", "path", "montage", "ok", "error", "sfreq", "mne_duration_s", "header_duration_s", "duration_mismatch_s",
              "n_std_channels", "window_s", "n_flat", "n_high_sd", "n_extreme", "n_nonfinite", "median_sd_uv", "max_peak_uv",
              "last_second_ok", "seconds"]
    C.write_csv(P["cache"] / "signal_qc_files.csv", rows, fields)
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    res = {k: agg(v) for k, v in by.items()}
    res["ALL"] = agg(rows)
    C.write_json(P["out"] / "s5_signal_qc.json", dict(
        by_dataset=res, strata=plan, n_sample=len(rows), seed=args.seed,
        thresholds=dict(flat_sd_uv=FLAT_UV, high_sd_uv=HIGH_SD_UV, extreme_peak_uv=EXTREME_UV, window_s="3 x 20 s at 10/50/90% of the recording"),
        source=f"stratified random sample of {len(rows)} EDFs (seed {args.seed})", seconds=round(time.time() - t0, 1)))
    log(f"STEP 5 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--min-per-stratum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    main(ap.parse_args())
