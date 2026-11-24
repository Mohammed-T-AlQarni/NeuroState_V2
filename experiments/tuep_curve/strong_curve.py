"""Stronger TUEP learning-curve test, step 2: same folds for every method (aggregate output only).

Data: clean resting-like EEG from prep_clean.py (up to 10 min per patient, 2-s epochs, 19 ch, 1-40 Hz, 128 Hz).
Methods:
  metadata     - header facts only (sampling rate, reference, year, channel count): a shortcut check.
  spectral     - log relative band power (delta/theta/alpha/beta) per channel + logistic regression.
  microstates  - repo microstates.py on 2-20 Hz data, maps fitted on training patients only, 24 parameters + LR.
  eegnet       - EEGNet-8,2 on 2-s windows (GPU), patient score = mean window probability.
Targets: epilepsy (main question), age (positive control: older vs younger half), ref (montage shortcut check).
Patient-level stratified 5-fold x 3 repeats; training subsets of 20 / 40 / 80 / all patients (balanced),
always scored on the same held-out fold.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import microstates as MS  # noqa: E402

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
SFREQ = 128
SIZES = [10, 20, 40, 80]   # per class


def load(cache):
    sel = list(csv.DictReader(open(Path(cache) / "selection.csv", encoding="utf-8")))
    X = [np.load(Path(cache) / f"{r['pid']}.npz")["x"] for r in sel]
    return sel, X


def targets(sel, name):
    if name == "epilepsy":
        return np.array([int(r["label"]) for r in sel]), np.ones(len(sel), bool)
    if name == "age":
        age = np.array([float(r["age"]) if r["age"] not in ("", None) else np.nan for r in sel])
        ok = np.isfinite(age)
        return (age >= np.nanmedian(age)).astype(int), ok
    if name == "ref":
        return np.array([int("LE" in r["ref"]) for r in sel]), np.ones(len(sel), bool)
    raise ValueError(name)


def matched(sel, max_age=5, max_year=2):
    """1:1 match each no-epilepsy patient to an epilepsy patient of similar age and recording year (greedy, seeded)."""
    g = np.random.default_rng(7)
    age = lambda r: float(r["age"]) if r["age"] else np.nan
    pos = [i for i, r in enumerate(sel) if r["label"] == "1"]
    neg = [i for i, r in enumerate(sel) if r["label"] == "0"]
    used, keep = set(), []
    for j in g.permutation(neg):
        best = None
        for i in pos:
            if i in used:
                continue
            da, dy = abs(age(sel[i]) - age(sel[j])), abs(float(sel[i]["year"]) - float(sel[j]["year"]))
            if da <= max_age and dy <= max_year and (best is None or da + dy < best[0]):
                best = (da + dy, i)
        if best:
            used.add(best[1]); keep += [int(j), best[1]]
    return sorted(keep)


def splits(y, repeats, subsets):
    out = []
    for rep in range(repeats):
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y)):
            r2 = np.random.default_rng(1000 * rep + fold)
            pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
            m = min(len(pos), len(neg))
            for n in SIZES + ["all"]:
                if n == "all":
                    out.append((rep, fold, "all", 0, tr, te)); continue
                if n >= m:
                    continue
                for s in range(subsets):
                    out.append((rep, fold, 2 * n, s, np.r_[r2.choice(pos, n, replace=False), r2.choice(neg, n, replace=False)], te))
    return out


# ---------------- features ----------------
def spectral_feats(X):
    f = np.fft.rfftfreq(256, 1 / SFREQ)
    bands = [(1, 4), (4, 8), (8, 13), (13, 30)]
    F = []
    for x in X:
        p = (np.abs(np.fft.rfft(x.astype(np.float32) * np.hanning(256), axis=2)) ** 2).mean(0)   # 19 x freqs
        tot = p[:, (f >= 1) & (f <= 30)].sum(1)
        F.append(np.log(np.concatenate([p[:, (f >= a) & (f < b)].sum(1) / tot for a, b in bands])))
    return np.array(F)


def lr_eval(F, y, sp):
    rows = []
    for rep, fold, n, s, tr, te in sp:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(F[tr], y[tr])
        rows.append((rep, fold, n, s, te, clf.predict_proba(F[te])[:, 1]))
    return rows


def microstate_eval(X, y, sp, smooth=3):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [2, 20], btype="band", fs=SFREQ, output="sos")
    D = [np.concatenate(list(sosfiltfilt(sos, x.astype(np.float32), axis=2)), axis=1) for x in X]   # 19 x T
    rng = np.random.default_rng(0)
    peaks = [MS.gfp_peaks(d, max_peaks=1000, rng=rng) for d in D]
    rows, t0 = [], time.time()
    for k, (rep, fold, n, s, tr, te) in enumerate(sp):
        maps, gev = MS.fit_maps([peaks[i] for i in tr], CHANNELS, SFREQ, seed=42, n_init=10)
        def feats(idx):
            out = []
            for i in idx:
                lab = MS.corr_sequence(D[i], maps).argmax(0)
                if smooth:
                    lab = MS.reject_short_segments(lab, D[i], smooth)
                out.append(MS.features(lab, SFREQ))
            return np.array(out)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(feats(tr), y[tr])
        rows.append((rep, fold, n, s, te, clf.predict_proba(feats(te))[:, 1], gev))
        if (k + 1) % 10 == 0:
            print(f"  microstates {k + 1}/{len(sp)} ({time.time() - t0:.0f} s)", flush=True)
    return rows


def eegnet_eval(X, y, sp, steps=800, batch=128, seed=0):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

    # per-patient, per-channel standardisation (removes gain/montage scale differences)
    Z = []
    for x in X:
        x = x.astype(np.float32)
        mu, sd = x.mean((0, 2), keepdims=True), x.std((0, 2), keepdims=True) + 1e-6
        Z.append(torch.tensor(np.clip((x - mu) / sd, -8, 8), dtype=torch.float16, device=dev))

    class EEGNet(nn.Module):
        def __init__(s, C=19, T=256, F1=8, D=2, F2=16, k=64, p=0.25):
            super().__init__()
            s.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1),
                                 nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
                                 nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
            s.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                 nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                 nn.AvgPool2d((1, 8)), nn.Dropout(p))
            with torch.no_grad():
                nf = s.b2(s.b1(torch.zeros(1, 1, C, T))).numel()
            s.fc = nn.Linear(nf, 1)

        def forward(s, x):
            return s.fc(s.b2(s.b1(x.unsqueeze(1))).flatten(1)).squeeze(1)

    rows, t0 = [], time.time()
    for k, (rep, fold, n, s, tr, te) in enumerate(sp):
        torch.manual_seed(seed + k)
        g = np.random.default_rng(seed + k)
        net = EEGNet().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        net.train()
        for step in range(steps):
            pats = np.r_[g.choice(pos, batch // 2), g.choice(neg, batch // 2)]
            xb = torch.stack([Z[i][g.integers(len(Z[i]))] for i in pats]).float()
            xb = xb * torch.empty(len(pats), 1, 1, device=dev).uniform_(0.8, 1.2) + 0.05 * torch.randn_like(xb)
            yb = torch.tensor(y[pats], dtype=torch.float32, device=dev)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():   # EEGNet max-norm on the spatial filter
                w = net.b1[2].weight; w.data = torch.renorm(w.data, 2, 0, 1.0)
        net.eval(); p = []
        with torch.no_grad():
            for i in te:
                p.append(torch.sigmoid(torch.cat([net(Z[i][j:j + 512].float()) for j in range(0, len(Z[i]), 512)])).mean().item())
        rows.append((rep, fold, n, s, te, np.array(p)))
        if (k + 1) % 10 == 0:
            print(f"  eegnet {k + 1}/{len(sp)} ({time.time() - t0:.0f} s)", flush=True)
    return rows


def metadata_feats(sel):
    fs_vals = sorted({r["fs"] for r in sel})
    F = []
    for r in sel:
        F.append([float(r["year"] or 0), int("LE" in r["ref"]), float(r["n_signals"] or 0)] + [int(r["fs"] == v) for v in fs_vals])
    return np.array(F, float)


# ---------------- summary ----------------
def summarise(rows, y, method, target, extra):
    per = defaultdict(list)
    recs = []
    for r in rows:
        rep, fold, n, s, te, p = r[:6]
        auc, bacc = roc_auc_score(y[te], p), balanced_accuracy_score(y[te], p >= 0.5)
        per[(n, rep, fold)].append((auc, bacc))
        recs.append(dict(repeat=rep, fold=fold, n_train=n, subset=s, auc=round(auc, 4), bacc=round(bacc, 4),
                         **({"gev": round(r[6], 3)} if len(r) > 6 else {})))
    sizes = sorted({k[0] for k in per if k[0] != "all"}) + ["all"]
    curve = {}
    for n in sizes:
        v = np.array([np.mean(per[k], 0) for k in per if k[0] == n])
        curve[str(n)] = dict(folds=len(v), auc_mean=float(v[:, 0].mean()), auc_se=float(v[:, 0].std(ddof=1) / np.sqrt(len(v))),
                             bacc_mean=float(v[:, 1].mean()), bacc_se=float(v[:, 1].std(ddof=1) / np.sqrt(len(v))))
    res = dict(method=method, target=target, curve=curve, **extra)
    if len(sizes) > 1:
        from scipy.stats import wilcoxon
        small = {k[1:]: np.mean(per[k], 0)[0] for k in per if k[0] == sizes[0]}
        large = {k[1:]: np.mean(per[k], 0)[0] for k in per if k[0] == "all"}
        d = np.array([large[k] - small[k] for k in large if k in small])
        res["all_vs_smallest_auc"] = dict(mean_gain=float(d.mean()), folds_improved=int((d > 0).sum()), folds=len(d),
                                          wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else None)
    return res, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True, choices=["metadata", "spectral", "microstates", "eegnet"])
    ap.add_argument("--target", default="epilepsy", choices=["epilepsy", "age", "ref"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--subsets", type=int, default=2)
    ap.add_argument("--full-only", action="store_true", help="only train on all training patients")
    ap.add_argument("--matched", action="store_true", help="1:1 age- and year-matched epilepsy / no-epilepsy cohort")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sel, X = load(a.cache)
    if a.matched:
        keep = matched(sel)
        sel, X = [sel[i] for i in keep], [X[i] for i in keep]
    y, ok = targets(sel, a.target)
    idx = np.where(ok)[0]
    sel, X, y = [sel[i] for i in idx], [X[i] for i in idx], y[idx]
    sp = splits(y, a.repeats, a.subsets)
    if a.full_only:
        sp = [s for s in sp if s[2] == "all"]
    print(f"{a.method} / {a.target}: {len(y)} patients ({int(y.sum())} vs {int((1 - y).sum())}), {len(sp)} fits", flush=True)
    extra = dict(patients=len(y), positives=int(y.sum()),
                 minutes_median=float(np.median([float(r["minutes"]) for r in sel])))
    if a.method == "metadata":
        rows = lr_eval(metadata_feats(sel), y, sp)
    elif a.method == "spectral":
        rows = lr_eval(spectral_feats(X), y, sp)
    elif a.method == "microstates":
        rows = microstate_eval(X, y, sp)
        extra["gev_median"] = float(np.median([r[6] for r in rows]))
    else:
        rows = eegnet_eval(X, y, sp)
    extra["seconds"] = round(time.time() - t0)
    res, recs = summarise(rows, y, a.method, a.target, extra)
    tag = f"{a.method}_{a.target}" + ("_matched" if a.matched else "")
    res["cohort"] = "age- and year-matched 1:1" if a.matched else "all clean patients"
    with open(out / f"folds_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    (out / f"summary_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
