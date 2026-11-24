"""Stronger TUEP test, step 1: select clean, resting-like EEG per patient (read-only on the data).

Per patient (TUEP epilepsy / no epilepsy), routine-length recordings only (8-75 min), in session order:
  * 19 channels, average reference, 1-40 Hz band-pass, 128 Hz.
  * skip the first 60 s and last 30 s; skip annotated seizures (+-60 s) and photic stimulation (from the PHOTIC channel).
  * 2-s epochs; drop any epoch with a channel peak-to-peak > 150 uV, a flat channel (SD < 0.5 uV) or strong muscle
    (20-40 Hz share > 35 % on any channel).
  * keep the epochs that look most like eyes-closed rest: posterior alpha share (O1, O2, P3, P4; 8-13 Hz of 2-20 Hz)
    at or above that recording's median.
  * stop at MAX_MIN minutes per patient (same cap for both groups); patients with < MIN_MIN minutes are dropped.
Output: one npz per patient in --cache (epochs x 19 x 256, float16, uV) + selection.csv with the patient's group,
age, sex, montage, sampling rate and year (no IDs leave the workspace; nothing here goes into a report).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data_report"))
import common as CM  # noqa: E402

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
SFREQ, EP = 128, 256
POST = [CHANNELS.index(c) for c in ("O1", "O2", "P3", "P4")]
MAX_MIN, MIN_MIN = 10.0, 4.0


def recordings(root):
    cache = Path(root) / "Claude_Workspace" / "cache" / "eeg_data_report"
    man = [r for r in CM.read_csv(cache / "manifest_edf.csv") if r["dataset"] in ("TUEP_epilepsy", "TUEP_no_epilepsy")]
    hdr = {r["path"]: r for r in CM.read_csv(cache / "edf_headers.csv")}
    seiz = defaultdict(list)
    for e in CM.read_csv(cache / "seizure_events.csv"):
        seiz[e["path"]].append((float(e["start"]), float(e["stop"])))
    by_pat = defaultdict(list)
    for r in man:
        h = hdr.get(r["path"])
        if h and h["ok"] == "1" and h["duration_s"] and 8 * 60 <= float(h["duration_s"]) <= 75 * 60:
            by_pat[(r["pid"], r["dataset"])].append((r["ses"], r["tok"], r["path"], h))
    return {k: sorted(v, key=lambda t: (t[0], t[1])) for k, v in sorted(by_pat.items())}, seiz


def load(path):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    pick, photic = {}, None
    for ch in raw.ch_names:
        n = CM.norm_label(ch)
        if "PHOTIC" in n and photic is None:
            photic = ch
        n = ALIAS.get(n.capitalize() if len(n) == 2 else n, n)
        for c in CHANNELS:
            if n.upper() == c.upper() and c not in pick:
                pick[c] = ch
    if len(pick) < len(CHANNELS):
        return None, None
    fs0 = raw.info["sfreq"]
    ph = None
    if photic is not None:
        ph = raw.copy().pick([photic]).load_data(verbose="ERROR").get_data()[0]
    raw.pick([pick[c] for c in CHANNELS]).load_data(verbose="ERROR")
    raw.rename_channels({pick[c]: c for c in CHANNELS})
    raw.set_channel_types({c: "eeg" for c in CHANNELS}, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if fs0 > 130:
        raw.notch_filter([50.0, 60.0], verbose="ERROR")
    raw.filter(1.0, 40.0, method="fir", phase="zero", verbose="ERROR")
    raw.resample(SFREQ, verbose="ERROR")
    x = (raw.get_data() * 1e6).astype(np.float32)
    bad_ph = np.zeros(x.shape[1] // EP, bool)
    if ph is not None and np.ptp(ph) > 0:
        # photic pulses: 2-s windows whose PHOTIC-channel range is a large share of its overall range
        step = int(round(fs0 * EP / SFREQ))
        nwin = min(len(ph) // step, len(bad_ph))
        r = np.array([np.ptp(ph[i * step:(i + 1) * step]) for i in range(nwin)])
        on = r > 0.3 * np.percentile(r, 99.5) if np.percentile(r, 99.5) > 5 * (np.median(r) + 1e-12) else np.zeros(nwin, bool)
        on = np.convolve(on.astype(int), np.ones(5), "same") > 0       # +-4 s around stimulation
        bad_ph[:nwin] = on
    return x, bad_ph


def clean_epochs(x, bad, skip):
    n = x.shape[1] // EP
    ep = x[:, : n * EP].reshape(19, n, EP).transpose(1, 0, 2)          # n x 19 x 256
    ok = ~bad[:n] & ~skip[:n]
    ptp = ep.max(2) - ep.min(2)
    ok &= ptp.max(1) <= 150
    ok &= ep.std(2).min(1) >= 0.5
    f = np.fft.rfftfreq(EP, 1 / SFREQ)
    p = np.abs(np.fft.rfft(ep * np.hanning(EP), axis=2)) ** 2
    tot = p[:, :, (f >= 1) & (f <= 40)].sum(2) + 1e-9
    ok &= (p[:, :, (f >= 20) & (f <= 40)].sum(2) / tot).max(1) <= 0.35
    a = p[:, POST][:, :, (f >= 8) & (f < 13)].sum((1, 2)) / (p[:, POST][:, :, (f >= 2) & (f < 20)].sum((1, 2)) + 1e-9)
    if ok.sum() == 0:
        return ep[:0]
    ok &= a >= np.median(a[ok])
    return ep[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"G:\NeuroState")
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    out = Path(a.cache); out.mkdir(parents=True, exist_ok=True)
    pats, seiz = recordings(a.root)
    rows, t0 = [], time.time()
    for k, ((pid, ds), recs) in enumerate(pats.items()):
        keep, used, h0 = [], 0, None
        for ses, tok, path, h in recs:
            if sum(len(e) for e in keep) * 2 / 60 >= MAX_MIN:
                break
            x, bad = load(path)
            if x is None:
                continue
            n = x.shape[1] // EP
            t = (np.arange(n) * EP + EP / 2) / SFREQ
            skip = (t < 60) | (t > n * EP / SFREQ - 30)
            for s, e in seiz[path]:
                skip |= (t > s - 60) & (t < e + 60)
            ep = clean_epochs(x, bad, skip)
            if len(ep):
                keep.append(ep); used += 1
                h0 = h0 or h
        ep = np.concatenate(keep) if keep else np.zeros((0, 19, EP), np.float32)
        ep = ep[: int(MAX_MIN * 30)]
        minutes = len(ep) * 2 / 60
        if minutes >= MIN_MIN:
            np.savez_compressed(out / f"{pid}.npz", x=ep.astype(np.float16))
            rows.append(dict(pid=pid, label=int(ds == "TUEP_epilepsy"), minutes=round(minutes, 2), recordings=used,
                             age=h0["age"], sex=h0["sex"], ref=h0["ref_suffixes"], fs=h0["fs_eeg"],
                             year=h0["ses_year"] or h0["header_year"], n_signals=h0["n_signals"]))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(pats)} patients, kept {len(rows)}  ({time.time() - t0:.0f} s)", flush=True)
    with open(out / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    y = np.array([r["label"] for r in rows]); m = np.array([r["minutes"] for r in rows])
    print(f"kept {len(rows)} patients (epilepsy {int(y.sum())}, no epilepsy {int((1 - y).sum())}); "
          f"minutes median {np.median(m):.1f} (epilepsy {np.median(m[y == 1]):.1f}, no epilepsy {np.median(m[y == 0]):.1f}); "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
