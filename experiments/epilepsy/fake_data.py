"""Write a complete FAKE data package in the NS_DATA format (NeuroState_Epilepsy_Ops shared/INTERFACES.md).

For writing and testing code on PCs without the real data. Every signal is random (pink noise plus a 10 Hz
rhythm; seizure windows get a 3 Hz rhythm on top), every patient code is fake (`fake0001`, ...).

Usage (from the repo root):
    python -m experiments.epilepsy.fake_data --out <folder> --seed 0

The folder must be empty or new. The script refuses to write into %NS_DATA% (the real package).

Layout written:
    MANIFEST.sha256, PACKAGE_INFO.json, labels.csv, meta.csv,
    splits/t1_r{0,1,2}.csv, splits/t1_r{r}_f{f}_val.csv, splits/t2.csv, t3_heldout.csv, t3_tuep.csv,
    splits/tusz_train.csv, tusz_dev.csv, tusz_eval.csv, run_B_extra.csv, run_C_extra.csv,
    windows19/<group>/<pid>.npz, windows10/<group>/<pid>.npz, seizure/<pid>.npz

Choices the interface leaves open (made here, to be confirmed by the director):
    - labels.csv has columns pid, group, label and lists the diagnosis patients only; the stage 1 seizure
      patients (group tusz_seiz) appear in meta.csv, the tusz_* split files and seizure/.
    - Nigeria and Guinea-Bissau have only 10-channel windows (their headset has no full 19-channel montage).
    - Windows are in microvolts (float16 cannot hold EEG in volts), as in experiments/prep.py.
    - t1 folds: StratifiedGroupKFold(5, shuffle=True, random_state=100*r) by patient;
      validation: stratified 15% of the training folds, random_state=100*r+f.
"""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SFREQ = 128
WIN = 512  # 4 s at 128 Hz
CH19 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
CH10 = ["F7", "F3", "F4", "F8", "T3", "T4", "T5", "T6", "O1", "O2"]
IDX10 = [CH19.index(c) for c in CH10]
BAND_X, BAND_MS = (1.0, 40.0), (2.0, 20.0)
N_REPEATS, N_FOLDS, VAL_FRAC = 3, 5, 0.15

# group -> (label, source, number of fake patients)
GROUPS = {
    "tuep_epi": ("epilepsy", "TUEP", 12),
    "tuep_ctrl": ("no_epilepsy", "TUEP", 12),
    "tueg_y": ("epilepsy", "TUEG", 6),
    "tusz_y": ("epilepsy", "TUSZ", 6),
    "tuab_train": ("normal_eeg", "TUAB", 10),
    "tuab_eval": ("normal_eeg", "TUAB", 4),
    "tusz_seiz": (None, "TUSZ", 6),           # stage 1 only: seizure / background windows
    "nigeria": (None, "Nigeria", 4),          # half epilepsy, half control
    "guinea_bissau": (None, "Guinea-Bissau", 4),
}
TUH_DIAG = ["tuep_epi", "tuep_ctrl", "tueg_y", "tusz_y", "tuab_train", "tuab_eval"]
EXTERNAL = ["nigeria", "guinea_bissau"]
LABELS_COLS = ["pid", "group", "label"]
META_COLS = ["pid", "group", "source", "n_windows", "minutes", "age", "sex", "year", "montage", "sfreq_orig"]


def _pink(rng, n_ch, n):
    """1/f noise per channel via spectral shaping."""
    spec = rng.standard_normal((n_ch, n // 2 + 1)) + 1j * rng.standard_normal((n_ch, n // 2 + 1))
    f = np.fft.rfftfreq(n, 1 / SFREQ)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n=n, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _signal(rng, n_ch, n_win, seizure_mask=None):
    """Continuous fake EEG in microvolts, average-referenced: pink noise + 10 Hz rhythm (+ 3 Hz in seizures)."""
    n = n_win * WIN
    t = np.arange(n) / SFREQ
    x = 15.0 * _pink(rng, n_ch, n)
    gain = rng.uniform(0.3, 1.0, size=(n_ch, 1))
    x += 10.0 * gain * np.sin(2 * np.pi * rng.uniform(9, 11) * t + rng.uniform(0, 2 * np.pi))
    if seizure_mask is not None and seizure_mask.any():
        s = np.repeat(seizure_mask.astype(float), WIN)
        x += 60.0 * gain * s * np.sin(2 * np.pi * 3.0 * t)
    return x - x.mean(axis=0, keepdims=True)


def _bands(x):
    out = []
    for lo, hi in (BAND_X, BAND_MS):
        sos = butter(4, [lo, hi], btype="band", fs=SFREQ, output="sos")
        out.append(sosfiltfilt(sos, x, axis=1))
    return out


def _windows(x):
    n_ch, n = x.shape
    return x.reshape(n_ch, n // WIN, WIN).transpose(1, 0, 2).astype(np.float16)


def _save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def make_patients(seed):
    """Fake patient table: pid, group, label, source (pids fake0001 ... in a shuffled order)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, (label, source, n) in GROUPS.items():
        for i in range(n):
            lab = label if label is not None else (None if g == "tusz_seiz" else ("epilepsy" if i < n // 2 else "no_epilepsy"))
            rows.append({"group": g, "label": lab, "source": source})
    order = rng.permutation(len(rows))
    df = pd.DataFrame([rows[i] for i in order])
    df.insert(0, "pid", [f"fake{i + 1:04d}" for i in range(len(df))])
    return df


def make_splits(pat, seed):
    """All split files as DataFrames, keyed by file name (without .csv)."""
    rng = np.random.default_rng(seed + 1)
    s = {}
    tuep = pat[pat.group.isin(["tuep_epi", "tuep_ctrl"])].sort_values("pid").reset_index(drop=True)
    y = (tuep.label == "epilepsy").astype(int).to_numpy()
    for r in range(N_REPEATS):
        fold = np.full(len(tuep), -1)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=100 * r)
        for f, (_, te) in enumerate(sgkf.split(tuep.pid, y, groups=tuep.pid)):
            fold[te] = f
        s[f"t1_r{r}"] = pd.DataFrame({"pid": tuep.pid, "fold": fold, "label": tuep.label})
        for f in range(N_FOLDS):
            tr = tuep[fold != f]
            _, val = train_test_split(tr, test_size=VAL_FRAC, stratify=tr.label, random_state=100 * r + f)
            s[f"t1_r{r}_f{f}_val"] = val[["pid", "label"]].sort_values("pid").reset_index(drop=True)

    def pids(g):
        return sorted(pat.pid[pat.group == g])

    s["t2"] = pd.DataFrame({"pid": pids("tuab_eval")})
    # T3: epilepsy patients whose EEG was read as normal. Held out: one TUEG Y + one TUSZ Y (fake choice).
    held = [rng.choice(pids("tueg_y")), rng.choice(pids("tusz_y"))]
    s["t3_heldout"] = pd.DataFrame({"pid": sorted(held)})
    s["t3_tuep"] = pd.DataFrame({"pid": sorted(rng.choice(pids("tuep_epi"), size=4, replace=False))})
    seiz = list(rng.permutation(pids("tusz_seiz")))
    s["tusz_train"] = pd.DataFrame({"pid": sorted(seiz[:4])})
    s["tusz_dev"] = pd.DataFrame({"pid": sorted(seiz[4:5])})
    s["tusz_eval"] = pd.DataFrame({"pid": sorted(seiz[5:])})
    extra_epi = pat[pat.group.isin(["tueg_y", "tusz_y"]) & ~pat.pid.isin(held)]
    s["run_B_extra"] = extra_epi[["pid", "label"]].sort_values("pid").reset_index(drop=True)
    extra_c = pd.concat([extra_epi, pat[pat.group == "tuab_train"]])
    s["run_C_extra"] = extra_c[["pid", "label"]].sort_values("pid").reset_index(drop=True)
    return s


def write_package(out, seed=0):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty. Give a new or empty folder.")
    ns_data = os.environ.get("NS_DATA")
    if ns_data and out.resolve() == Path(ns_data).resolve():
        raise SystemExit("Refusing to write fake data into NS_DATA (the real package folder).")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pat = make_patients(seed)
    meta = []
    for row in pat.itertuples():
        n_win = int(rng.integers(30, 61))  # 2-4 minutes of windows
        start = (np.arange(n_win) * WIN / SFREQ + float(rng.integers(0, 60))).astype(np.float32)
        info = {"pid": row.pid, "group": row.group, "source": row.source, "n_windows": n_win,
                "minutes": round(n_win * 4 / 60, 2), "age": int(rng.integers(18, 80)),
                "sex": str(rng.choice(["F", "M"])), "year": int(rng.integers(2002, 2018)),
                "montage": "emotiv_14" if row.group in EXTERNAL else str(rng.choice(["01_tcp_ar", "02_tcp_le"])),
                "sfreq_orig": int(rng.choice([250, 256, 400, 512])) if row.group not in EXTERNAL else 128}
        meta.append(info)
        if row.group == "tusz_seiz":
            mask = np.zeros(n_win, bool)
            k = int(rng.integers(3, 8))
            s0 = int(rng.integers(0, n_win - k))
            mask[s0:s0 + k] = True
            x, _ = _bands(_signal(rng, 19, n_win, mask))
            _save_npz(out / "seizure" / f"{row.pid}.npz", x=_windows(x), y=mask.astype(np.int8), start_s=start)
        elif row.group in EXTERNAL:
            x, x_ms = _bands(_signal(rng, 10, n_win))
            _save_npz(out / "windows10" / row.group / f"{row.pid}.npz", x=_windows(x), x_ms=_windows(x_ms), start_s=start)
        else:
            x, x_ms = _bands(_signal(rng, 19, n_win))
            w, w_ms = _windows(x), _windows(x_ms)
            _save_npz(out / "windows19" / row.group / f"{row.pid}.npz", x=w, x_ms=w_ms, start_s=start)
            _save_npz(out / "windows10" / row.group / f"{row.pid}.npz", x=w[:, IDX10], x_ms=w_ms[:, IDX10], start_s=start)

    labels = pat[pat.group != "tusz_seiz"][LABELS_COLS].sort_values("pid")
    labels.to_csv(out / "labels.csv", index=False)
    pd.DataFrame(meta)[META_COLS].sort_values("pid").to_csv(out / "meta.csv", index=False)
    (out / "splits").mkdir(exist_ok=True)
    for name, df in make_splits(pat, seed).items():
        df.to_csv(out / "splits" / f"{name}.csv", index=False)

    counts = pat.group.value_counts().reindex(list(GROUPS)).astype(int).to_dict()
    info = {"fake": True, "seed": seed, "code_commit": _code_commit(),
            "date": datetime.now().isoformat(timespec="seconds"), "counts_per_group": counts,
            "n_patients": int(len(pat)),
            "preprocessing": {"sfreq": SFREQ, "window_s": 4, "window_samples": WIN, "overlap": 0,
                              "reference": "average", "channels19": CH19, "channels10": CH10,
                              "band_x_hz": list(BAND_X), "band_x_ms_hz": list(BAND_MS),
                              "dtype": "float16", "units": "uV"}}
    with open(out / "PACKAGE_INFO.json", "w") as f:
        json.dump(info, f, indent=2)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    with open(out / "MANIFEST.sha256", "w", newline="\n") as f:
        for p in files:
            f.write(f"{_sha256(p)}  {p.relative_to(out).as_posix()}\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="new or empty folder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = write_package(args.out, args.seed)
    n = sum(1 for _ in open(out / "MANIFEST.sha256"))
    print(f"fake package written: {out} ({n} files in MANIFEST.sha256)")


if __name__ == "__main__":
    main()
