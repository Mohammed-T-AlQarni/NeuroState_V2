"""Sanity checks for the TUEP learning-curve run (aggregate output only).
1. Microstate fit quality (GEV) and whether the 24 features vary.
2. Does the same feature pipeline detect a KNOWN difference? Target: recording reference (-REF vs -LE montage),
   which is not about epilepsy but is a real property of the files.
3. Spectral baseline on the same 3-minute segments: relative band power (2-4, 4-8, 8-13, 13-20 Hz) per channel
   + logistic regression, same folds and training sizes, epilepsy vs no epilepsy.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "data_report")); sys.path.insert(0, str(HERE))
import microstates as MS  # noqa
import tuep_learning_curve as T  # noqa

recs = T.choose_recordings(r"G:\NeuroState")
hdr = {r["path"]: r for r in csv.DictReader(open(r"G:\NeuroState\Claude_Workspace\cache\eeg_data_report\edf_headers.csv", encoding="utf-8"))}
data, y, ref = [], [], []
for r in recs:
    x = T.prepare(r, r"G:\NeuroState\Claude_Workspace\cache\tuep_curve")
    if x is None:
        continue
    data.append(x); y.append(r["label"]); ref.append(1 if "LE" in hdr[r["path"]]["ref_suffixes"] else 0)
y, ref = np.array(y), np.array(ref)
rng = np.random.default_rng(0)
peaks = [MS.gfp_peaks(x, 500, rng) for x in data]

maps, gev = MS.fit_maps(peaks, T.CHANNELS, T.SFREQ, seed=42, n_init=10)
F = np.array([MS.features(MS.corr_sequence(x, maps).argmax(0), T.SFREQ) for x in data])
print(f"1. GEV of maps fitted on all 200 patients: {gev:.3f}")
print("   feature SD across patients (first 12):", np.round(F[:, :12].std(0), 3).tolist())
print("   mean coverage by group  epilepsy:", np.round(F[y == 1, :4].mean(0), 3).tolist(), " no-epilepsy:", np.round(F[y == 0, :4].mean(0), 3).tolist())


def cv_auc(X, target, sizes=(10, 20, 40, 80), repeats=3, fit_maps=False):
    out = defaultdict(list)
    for rep in range(repeats):
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X if X is not None else np.zeros(len(target)), target)):
            r2 = np.random.default_rng(1000 * rep + fold)
            pos, neg = tr[target[tr] == 1], tr[target[tr] == 0]
            for n in sizes:
                if n > min(len(pos), len(neg)):
                    continue
                vals = []
                for s in range(1 if n >= min(len(pos), len(neg)) else 3):
                    sub = np.r_[r2.choice(pos, n, replace=False), r2.choice(neg, n, replace=False)]
                    if fit_maps:
                        m, _ = MS.fit_maps([peaks[i] for i in sub], T.CHANNELS, T.SFREQ, seed=42, n_init=10)
                        f = lambda idx: np.array([MS.features(MS.corr_sequence(data[i], m).argmax(0), T.SFREQ) for i in idx])
                        Xtr, Xte = f(sub), f(te)
                    else:
                        Xtr, Xte = X[sub], X[te]
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr, target[sub])
                    vals.append(roc_auc_score(target[te], clf.predict_proba(Xte)[:, 1]))
                out[2 * n].append(np.mean(vals))
    return {k: (round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 3)) for k, v in sorted(out.items())}


print(f"2. Known-difference check: microstate features predicting reference montage (LE {int(ref.sum())} vs REF {int((1 - ref).sum())})")
print("   AUC by training size (mean, SE):", cv_auc(None, ref, sizes=tuple(n for n in (10, 20, 40, 80) if n <= min(ref.sum(), (1 - ref).sum()) * 0.8), fit_maps=True))

bands = [(2, 4), (4, 8), (8, 13), (13, 20)]
B = []
for x in data:
    f, p = welch(x, fs=T.SFREQ, nperseg=512, axis=1)
    tot = p[:, (f >= 2) & (f <= 20)].sum(1)
    B.append(np.concatenate([p[:, (f >= a) & (f < b)].sum(1) / tot for a, b in bands]))
B = np.log(np.array(B))
print("3. Spectral baseline, epilepsy vs no epilepsy, AUC by training size (mean, SE):", cv_auc(B, y))
