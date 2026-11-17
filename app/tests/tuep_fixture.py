"""A SYNTHETIC tree shaped like the TUEP v3.1.0 release (layout from its AAREADME.txt), for testing the inventory and
preparation tools before the real data arrive. Signals are random noise plus a 10 Hz rhythm; nothing here is real.
Assumptions to re-check on the real release are marked ASSUMED."""
from __future__ import annotations

import string
from pathlib import Path

import mne
import numpy as np
import pandas as pd

STD = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]


def pid(i: int) -> str:  # TUH-style 8-letter patient ids (ASSUMED format)
    a = string.ascii_lowercase
    return "aaaaaa" + a[i // 26] + a[i % 26]


def write_edf(path: Path, ref="REF", seconds=40.0, sfreq=250.0, seed=0):
    rng = np.random.default_rng(seed)
    ch = [f"EEG {c}-{ref}" for c in STD] + [f"EEG A1-{ref}", "EKG1-REF"]
    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    data = 25e-6 * np.sin(2 * np.pi * 10 * t)[None] * rng.uniform(-1, 1, (len(ch), 1)) + 10e-6 * rng.standard_normal((len(ch), n))
    raw = mne.io.RawArray(data, mne.create_info(ch, sfreq, "eeg"), verbose="ERROR")
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True, verbose="ERROR")


def write_csv_bi(edf: Path, duration: float, seizures: list[tuple[float, float]]):
    """TUSZ-style term annotations (ASSUMED exact header lines; the parser only relies on '#' comments and the
    'channel,start_time,stop_time,label,confidence' header)."""
    lines = ["# version = csv_v1.0.0", f"# bname = {edf.stem}", f"# duration = {duration:.4f} secs", "#",
             "channel,start_time,stop_time,label,confidence"]
    t = 0.0
    for a, b in seizures:
        lines += [f"TERM,{t:.4f},{a:.4f},bckg,1.0000", f"TERM,{a:.4f},{b:.4f},seiz,1.0000"]
        t = b
    lines.append(f"TERM,{t:.4f},{duration:.4f},bckg,1.0000")
    edf.with_suffix(".csv_bi").write_text("\n".join(lines) + "\n")


def build(root: Path, n_e=10, n_n=10, seconds=40.0) -> dict:
    """Epilepsy patients: 2 sessions x 2 files; no-epilepsy: 1 session x 1 file. Patient 0 has a seizure 10-18 s in its
    first file. Patient 1 records with linked ears (02_tcp_le, '-LE' names)."""
    root = Path(root)
    meta, common, unique, k = [], [], [], 0
    for i in range(n_e + n_n):
        p = pid(i)
        epi = i < n_e
        cls = "00_epilepsy" if epi else "01_no_epilepsy"
        for s in range(2 if epi else 1):
            ses = f"s{s + 1:03d}_{2010 + s}"
            mont, ref = ("02_tcp_le", "LE") if i == 1 else ("01_tcp_ar", "REF")
            for t in range(2 if epi else 1):
                edf = root / cls / p / ses / mont / f"{p}_s{s + 1:03d}_t{t:03d}.edf"
                write_edf(edf, ref=ref, seconds=seconds, seed=k)
                k += 1
                write_csv_bi(edf, seconds, [(10.0, 18.0)] if (i == 0 and s == 0 and t == 0) else [])
            if i == n_e + n_n - 1:  # one session without metadata
                pass
            else:
                meta.append({"Subject ID": p, "Session": s + 1, "Diagnosis": "epilepsy" if epi else "no epilepsy",
                             "Medications": "levetiracetam" if epi else "none"})
            (common if i % 4 == 0 else unique).append(f"{cls}/{p}/{ses}")
    meta.append({"Subject ID": "zzzzzzzz", "Session": 1, "Diagnosis": "?", "Medications": "?"})  # metadata without EEG
    docs = root / "DOCS"
    docs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta).to_excel(docs / "metadata_v00r.xlsx", sheet_name="sessions", index=False)
    (docs / "sessions_common_with_tusz.list").write_text("\n".join(common) + "\n")
    (docs / "sessions_unique_to_tuep.list").write_text("\n".join(unique) + "\n")
    return {"patients": n_e + n_n, "files_e": n_e * 4, "files_n": n_n, "sessions_e": n_e * 2, "sessions_n": n_n}
