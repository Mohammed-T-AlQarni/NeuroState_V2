"""How much of the 3-minute segments is artifact? (aggregate output only)"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "data_report")); sys.path.insert(0, str(HERE))
import tuep_learning_curve as T  # noqa

recs = T.choose_recordings(r"G:\NeuroState")
X = [T.prepare(r, r"G:\NeuroState\Claude_Workspace\cache\tuep_curve") for r in recs]
X = [x for x in X if x is not None]
sd = np.array([x.std(1) for x in X])                  # patients x channels (uV)
print("median channel SD (uV):", round(float(np.median(sd)), 1), " 95th pct:", round(float(np.percentile(sd, 95)), 1), " max:", round(float(sd.max()), 1))
ratio = sd.max(1) / np.median(sd, 1)
print("patients with one channel > 3x their median channel SD:", int((ratio > 3).sum()), "of", len(X))
ch_dom = np.bincount(sd.argmax(1), minlength=19)
print("channel with the largest SD, count per channel:", dict(zip(T.CHANNELS, ch_dom.tolist())))
fs = int(T.SFREQ)
bad = []
for x in X:
    ep = x[:, : (x.shape[1] // fs) * fs].reshape(19, -1, fs)
    ptp = ep.max(2) - ep.min(2)
    bad.append((ptp.max(0) > 150).mean())
bad = np.array(bad)
print("share of 1-s epochs with any channel peak-to-peak > 150 uV: median", round(float(np.median(bad)), 3), " mean", round(float(bad.mean()), 3),
      " patients with > 50% bad:", int((bad > 0.5).sum()))
