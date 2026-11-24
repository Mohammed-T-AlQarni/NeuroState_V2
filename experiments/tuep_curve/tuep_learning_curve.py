"""Small learning-curve test on TUEP: does a microstate model get better with more training patients?

Read-only on the data. Per patient: 1 recording, 3 clean minutes (no annotated seizure within 60 s),
prepared as in prep.py (19 channels, average reference, 2-20 Hz, 128 Hz). Microstate maps are fitted
with the repo's microstates.py on the TRAINING patients of each fold only; 24 microstate parameters +
logistic regression. Patient-level stratified 5-fold CV x 3 repeats; inside each fold the model is
trained on 20, 40, 80 or all 160 training patients (balanced), and always tested on the same 40
held-out patients.

Usage:
  python tuep_learning_curve.py --root G:\\NeuroState --cache <cache dir> --out <results dir>
Outputs: <out>/curve_folds.csv, curve_summary.json, learning_curve.png (no patient IDs).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                    # experiments/ (microstates.py)
sys.path.insert(0, str(HERE.parent / "data_report"))    # common.py (label normalisation)
import microstates as MS  # noqa: E402
import common as CM       # noqa: E402

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
SFREQ = 128.0
SEG_START, SEG_LEN, PAD, SEIZ_MARGIN = 30.0, 180.0, 10.0, 60.0
SIZES = [10, 20, 40, 80]   # patients PER CLASS in the training subset (total 20, 40, 80, 160)


def choose_recordings(root):
    cache = Path(root) / "Claude_Workspace" / "cache" / "eeg_data_report"
    man = [r for r in CM.read_csv(cache / "manifest_edf.csv") if r["dataset"] in ("TUEP_epilepsy", "TUEP_no_epilepsy")]
    hdr = {r["path"]: r for r in CM.read_csv(cache / "edf_headers.csv")}
    seiz = defaultdict(list)
    for e in CM.read_csv(cache / "seizure_events.csv"):
        seiz[e["path"]].append((float(e["start"]), float(e["stop"])))
    by_pat = defaultdict(list)
    for r in man:
        h = hdr.get(r["path"])
        if h and h["ok"] == "1" and h["duration_s"]:
            by_pat[(r["pid"], r["dataset"])].append((r["ses"], r["tok"], r["path"], float(h["duration_s"])))
    chosen = []
    for (pid, ds), recs in sorted(by_pat.items()):
        for ses, tok, path, dur in sorted(recs):
            a, b = SEG_START, SEG_START + SEG_LEN
            if dur < b + PAD:
                continue
            if any(s - SEIZ_MARGIN < b and e + SEIZ_MARGIN > a for s, e in seiz[path]):
                continue
            chosen.append(dict(pid=pid, label=1 if ds == "TUEP_epilepsy" else 0, path=path))
            break
    return chosen


def prepare(rec, cache_dir):
    import mne
    out = Path(cache_dir) / f"{rec['pid']}.npz"
    if out.exists():
        return np.load(out)["ms"]
    raw = mne.io.read_raw_edf(rec["path"], preload=False, verbose="ERROR")
    raw.crop(SEG_START - PAD, SEG_START + SEG_LEN + PAD)
    pick = {}
    for ch in raw.ch_names:
        n = CM.norm_label(ch)
        n = ALIAS.get(n.capitalize() if len(n) == 2 else n, n)
        for c in CHANNELS:
            if n.upper() == c.upper() and c not in pick:
                pick[c] = ch
    if len(pick) < len(CHANNELS):
        return None
    raw.pick([pick[c] for c in CHANNELS]).load_data(verbose="ERROR")
    raw.rename_channels({pick[c]: c for c in CHANNELS})
    raw.set_channel_types({c: "eeg" for c in CHANNELS}, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(2.0, 20.0, method="fir", phase="zero", verbose="ERROR")
    raw.resample(SFREQ, verbose="ERROR")
    x = (raw.get_data() * 1e6).astype(np.float32)
    p = int(PAD * SFREQ)
    x = x[:, p:p + int(SEG_LEN * SFREQ)]
    np.savez_compressed(out, ms=x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"G:\NeuroState")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--subsets", type=int, default=3, help="random training subsets per size (smaller sizes)")
    ap.add_argument("--ptp", type=float, default=0, help="drop 1-s epochs with any channel peak-to-peak above this (uV); 0 = off")
    ap.add_argument("--min-clean", type=float, default=90, help="seconds of clean EEG a patient needs (with --ptp)")
    ap.add_argument("--smooth", type=int, default=0, help="minimum segment length in samples (0 = no smoothing)")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    recs = choose_recordings(a.root)
    data, labels, peaks = [], [], []
    rng = np.random.default_rng(2026)
    for i, r in enumerate(recs):
        x = prepare(r, a.cache)
        if x is None:
            continue
        if a.ptp:
            fs = int(SFREQ)
            ep = x[:, : (x.shape[1] // fs) * fs].reshape(x.shape[0], -1, fs)
            keep = (ep.max(2) - ep.min(2)).max(0) <= a.ptp
            if keep.sum() < a.min_clean:
                continue
            x = ep[:, keep, :].reshape(x.shape[0], -1)
        data.append(x); labels.append(r["label"]); peaks.append(MS.gfp_peaks(x, max_peaks=500, rng=rng))
        if (i + 1) % 25 == 0:
            print(f"  prepared {i + 1}/{len(recs)}  ({time.time() - t0:.0f} s)", flush=True)
    y = np.array(labels)
    print(f"patients used: {len(y)} (epilepsy {int(y.sum())}, no epilepsy {int((1 - y).sum())}); "
          f"{SEG_LEN / 60:.0f} min each", flush=True)

    rows = []
    for rep in range(a.repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            r2 = np.random.default_rng(1000 * rep + fold)
            pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
            for n in SIZES:
                n_sub = 1 if n >= min(len(pos), len(neg)) else a.subsets
                for s in range(n_sub):
                    sub = np.r_[r2.choice(pos, min(n, len(pos)), replace=False), r2.choice(neg, min(n, len(neg)), replace=False)]
                    maps, gev = MS.fit_maps([peaks[i] for i in sub], CHANNELS, SFREQ, seed=42, n_init=10)
                    def feats(idx):
                        out = []
                        for i in idx:
                            lab = MS.corr_sequence(data[i], maps).argmax(0)
                            if a.smooth:
                                lab = MS.reject_short_segments(lab, data[i], a.smooth)
                            out.append(MS.features(lab, SFREQ))
                        return np.array(out)
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
                    clf.fit(feats(sub), y[sub])
                    p = clf.predict_proba(feats(te))[:, 1]
                    rows.append(dict(repeat=rep, fold=fold, n_train=2 * n, subset=s, gev=round(gev, 3),
                                     auc=roc_auc_score(y[te], p), bacc=balanced_accuracy_score(y[te], p >= 0.5)))
            print(f"  repeat {rep} fold {fold} done ({time.time() - t0:.0f} s)", flush=True)

    with open(Path(a.out) / "curve_folds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {}
    for n in SIZES:
        # average subsets within a fold first, so every fold counts once
        per_fold = defaultdict(list)
        for r in rows:
            if r["n_train"] == 2 * n:
                per_fold[(r["repeat"], r["fold"])].append(r)
        auc = np.array([np.mean([r["auc"] for r in v]) for v in per_fold.values()])
        bacc = np.array([np.mean([r["bacc"] for r in v]) for v in per_fold.values()])
        summary[2 * n] = dict(folds=len(auc), auc_mean=float(auc.mean()), auc_sd=float(auc.std(ddof=1)),
                              auc_se=float(auc.std(ddof=1) / np.sqrt(len(auc))),
                              bacc_mean=float(bacc.mean()), bacc_sd=float(bacc.std(ddof=1)),
                              bacc_se=float(bacc.std(ddof=1) / np.sqrt(len(bacc))))
    # paired test: smallest vs largest training size, per fold
    def fold_vals(n, key):
        d = defaultdict(list)
        for r in rows:
            if r["n_train"] == n:
                d[(r["repeat"], r["fold"])].append(r[key])
        return {k: np.mean(v) for k, v in d.items()}
    small, large = fold_vals(2 * SIZES[0], "auc"), fold_vals(2 * SIZES[-1], "auc")
    diffs = np.array([large[k] - small[k] for k in large])
    from scipy.stats import wilcoxon
    stat = wilcoxon(diffs) if np.any(diffs != 0) else None
    out = dict(design=dict(patients=int(len(y)), epilepsy=int(y.sum()), no_epilepsy=int((1 - y).sum()),
                           minutes_per_patient=SEG_LEN / 60, cv="stratified 5-fold x %d, patient level" % a.repeats,
                           model="24 microstate parameters + logistic regression; maps fitted on training patients only",
                           training_sizes=[2 * n for n in SIZES], subsets_per_small_size=a.subsets,
                           artifact_ptp_uV=a.ptp, min_clean_s=a.min_clean if a.ptp else None, smoothing_samples=a.smooth),
               curve=summary,
               largest_vs_smallest_auc=dict(mean_gain=float(diffs.mean()), folds_improved=int((diffs > 0).sum()),
                                            folds=int(len(diffs)), wilcoxon_p=(float(stat.pvalue) if stat else None)),
               seconds=round(time.time() - t0))
    with open(Path(a.out) / "curve_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
