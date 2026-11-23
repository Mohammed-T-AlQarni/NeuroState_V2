"""Build a small FAKE copy of the G:\\NeuroState layout (synthetic signals, invented IDs) for testing.
Usage: python tests/make_fake_neurostate.py <out_root>
"""
import gzip
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
R = random.Random(0)
STD = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ"]


def edf(path, dur=120, fs=250, suffix="REF", extra=("EKG1", "PHOTIC"), sex="M", age=40, year=2010, flat=False, extreme=False, truncate=False, drop=()):
    labels = [f"EEG {c}-{suffix}" for c in STD if c not in drop] + [f"EEG {c}-{suffix}" if c != "PHOTIC" else "PHOTIC-REF" for c in extra]
    ns = len(labels)
    nrec = int(dur)
    h = b""
    def f(s, w):
        return str(s)[:w].ljust(w).encode("latin-1")
    pid = path.stem.split("_")[0]
    h += f("0", 8) + f(f"{pid} {sex} 01-JAN-1970 {pid} Age:{age}", 80) + f(f"Startdate 01-JAN-{year} X X X", 80)
    h += f(f"01.01.{year % 100:02d}", 8) + f("10.00.00", 8) + f(256 + 256 * ns, 8) + f("", 44) + f(nrec, 8) + f(1, 8) + f(ns, 4)
    for w, vals in ((16, labels), (80, ["Unknown"] * ns), (8, ["uV"] * ns), (8, ["-3200"] * ns), (8, ["3200"] * ns),
                    (8, ["-32768"] * ns), (8, ["32767"] * ns), (80, ["HP:0.000 Hz LP:0.0 Hz N:0.0"] * ns), (8, [fs] * ns), (32, [""] * ns)):
        h += b"".join(f(v, w) for v in vals)
    t = np.arange(nrec * fs) / fs
    data = np.zeros((ns, nrec * fs))
    for i in range(ns):
        data[i] = 30 * np.sin(2 * np.pi * 10 * t + i) + rng.normal(0, 10, t.size)
    if flat:
        data[3] = 0
    if extreme:
        data[5, 1000:1100] = 3100
    dig = np.clip(data / 3200 * 32767, -32768, 32767).astype("<i2")
    body = b"".join(dig[:, r * fs:(r + 1) * fs].tobytes() for r in range(nrec))
    if truncate:
        body = body[: len(body) // 2 + 7]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(h + body)


def sidecars(path, dur, seizures):
    chans = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8"]
    head = f"# version = csv_v1.0.0\n# bname = {path.stem}\n# duration = {dur:.2f} secs\n# montage_file = x\n#\n"
    rows, bi = [], []
    for c in chans:
        tcur = 0.0
        for s, e, typ in seizures:
            if s > tcur:
                rows.append(f"{c},{tcur:.4f},{s:.4f},bckg,1.0000")
            rows.append(f"{c},{s:.4f},{e:.4f},{typ},1.0000")
            tcur = e
        rows.append(f"{c},{tcur:.4f},{dur:.4f},bckg,1.0000")
    tcur = 0.0
    for s, e, typ in seizures:
        if s > tcur:
            bi.append(f"TERM,{tcur:.4f},{s:.4f},bckg,1.0000")
        bi.append(f"TERM,{s:.4f},{e:.4f},seiz,1.0000")
        tcur = e
    bi.append(f"TERM,{tcur:.4f},{dur:.4f},bckg,1.0000")
    path.with_suffix(".csv").write_text(head + "channel,start_time,stop_time,label,confidence\n" + "\n".join(rows) + "\n")
    path.with_suffix(".csv_bi").write_text(head + "channel,start_time,stop_time,label,confidence\n" + "\n".join(bi) + "\n")


def pid(i, pre):
    s = ""
    n = i
    for _ in range(4):
        s = chr(97 + n % 26) + s
        n //= 26
    return pre + s


def recordings(base, pids, montages, label=True, seiz_prob=0.3, nses=(1, 3), ntok=(1, 2), year0=2003, three=False):
    out = []
    for p in pids:
        for si in range(1, R.randint(*nses) + 1):
            yr = year0 + R.randint(0, 12)
            mont = R.choice(montages)
            for ti in range(R.randint(*ntok)):
                sub = [f"{R.randint(0, 9):03d}"] if three else []
                path = base.joinpath(*sub, p, f"s{si:03d}_{yr}", mont, f"{p}_s{si:03d}_t{ti:03d}.edf")
                dur = R.choice([45, 120, 180, 300])
                fs = 400 if mont == "03_tcp_ar_a" else R.choice([250, 256])
                edf(path, dur=dur, fs=fs, suffix="LE" if "le" in mont else "REF", sex=R.choice("MF"), age=R.randint(1, 90), year=yr,
                    drop=("A1", "A2") if mont == "03_tcp_ar_a" else ())
                if label:
                    sz = []
                    if R.random() < seiz_prob and dur > 60:
                        s = R.uniform(5, dur - 40)
                        sz.append((s, s + R.uniform(8, 35), R.choice(["fnsz", "gnsz", "cpsz", "absz", "tcsz"])))
                    sidecars(path, dur, sz)
                out.append(path)
    return out


def main(root):
    root = Path(root)
    tusz = root / "NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/edf"
    M = ["01_tcp_ar", "02_tcp_le", "03_tcp_ar_a"]
    tr = [pid(i, "aaaa") for i in range(8)]
    recordings(tusz / "train", tr, M, seiz_prob=0.5)
    recordings(tusz / "dev", [pid(i, "aaab") for i in range(3)], M, seiz_prob=0.5)
    recordings(tusz / "eval", [pid(i, "aaac") for i in range(3)], M, seiz_prob=0.5)
    tuep = root / "TUH/TUEP"
    epi = tr[:2] + [pid(i, "aaad") for i in range(6)]
    recordings(tuep / "00_epilepsy", epi, M, seiz_prob=0.1, nses=(1, 4), ntok=(1, 3))
    recordings(tuep / "01_no_epilepsy", [pid(i, "aaae") for i in range(5)], M, seiz_prob=0.0, nses=(1, 2), year0=2008)
    # a corrupt / truncated file
    edf(tuep / "00_epilepsy/aaadaaab/s009_2010/01_tcp_ar/aaadaaab_s009_t000.edf", truncate=True)
    sidecars(tuep / "00_epilepsy/aaadaaab/s009_2010/01_tcp_ar/aaadaaab_s009_t000.edf", 120, [])
    p = tuep / "01_no_epilepsy/aaaeaaaa/s005_2011/02_tcp_le/aaaeaaaa_s005_t000.edf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"garbage")
    edf(tuep / "00_epilepsy/aaadaaac/s008_2012/01_tcp_ar/aaadaaac_s008_t000.edf", flat=True, extreme=True)
    sidecars(tuep / "00_epilepsy/aaadaaac/s008_2012/01_tcp_ar/aaadaaac_s008_t000.edf", 120, [])
    docs = tuep / "DOCS"
    docs.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, grp in ((tuep / "00_epilepsy", "epilepsy"), (tuep / "01_no_epilepsy", "no_epilepsy")):
        for e in d.rglob("*.edf"):
            rows.append({"File Name": e.stem, "Class": grp, "Age": R.randint(1, 90), "Gender": R.choice("MF"),
                         "Medications": R.choice(["Keppra, Dilantin", "Lamictal", "", "none"]),
                         "Clinical History": R.choice(["normal EEG, syncope episode", "history of seizures", "headache", "spells, ? PNES", ""])})
    pd.DataFrame(rows).to_excel(docs / "metadata_v00.xlsx", index=False)
    pd.DataFrame(rows).to_excel(docs / "metadata_v00r.xlsx", index=False)
    (docs / "sessions_common_with_tusz.list").write_text(f"{tr[0]}_s001\n{tr[1]}_s001\n")
    (docs / "01_tcp_ar_montage.txt").write_text("montage = 0, FP1-F7: EEG FP1-REF -- EEG F7-REF\n")
    add = root / "TUH/TUH_Additional"
    tdocs = add / "TUSZ_v2.0.6/DOCS"
    tdocs.mkdir(parents=True, exist_ok=True)
    meta = []
    for e in tusz.rglob("*.edf"):
        meta.append([e.stem, R.choice("YN"), R.choice("YN"), R.choice("YN"), R.choice(["", "migraine", "stroke", "TBI"]), R.choice(["Keppra", "", "Depakote"]), R.choice(["", "HTN"])])
    with pd.ExcelWriter(tdocs / "metadata_v06r.xlsx") as w:
        hdr = pd.DataFrame([["TUSZ metadata v06r", None, None, None, None, None, None], [None] * 7,
                            ["File", "Epilepsy", "Electrographic seizure", "Clinical seizure", "Other Diagnosis", "Medications", "Comorbidities"]])
        pd.concat([hdr, pd.DataFrame(meta)]).to_excel(w, sheet_name="TUSZ", index=False, header=False)
    pd.DataFrame([["fnsz", "focal non-specific seizure"], ["gnsz", "generalized non-specific seizure"], ["absz", "absence seizure"]],
                 columns=["Code", "Description"]).to_excel(tdocs / "seizures_types_v02.xlsx", index=False)
    recordings(add / "TUSZ_v2.0.6/edf/train", tr[:1], M, seiz_prob=0.5)
    ty = recordings(add / "TUEG_v2.0.2/epilepsy_Y/edf", [pid(i, "aaaf") for i in range(4)] + [tr[2]], M, label=False, three=True)
    recordings(add / "TUEG_v2.0.2/edf", [pid(i, "aaag") for i in range(10)] + [pid(0, "aaaf")], M, label=False, three=True, ntok=(1, 1))
    inv = root / "Claude_Workspace/inventory"
    inv.mkdir(parents=True, exist_ok=True)
    (inv / "tueg_epilepsy_Y_edfs.txt").write_text("\n".join(str(p) for p in ty) + "\n")
    ext = root / "External_Datasets"
    for name, n in (("EEGs_Guinea-Bissau", 6), ("EEGs_Nigeria", 5)):
        (ext / name).mkdir(parents=True, exist_ok=True)
        ids = []
        for i in range(1, n + 1):
            cols = ["", "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "COUNTER", "INTERPOLATED", "timestamp"]
            nr = 128 * R.randint(200, 400)
            x = 4200 + rng.normal(0, 15, (nr, 14))
            ts = 1.5e9 + np.arange(nr) / 128
            df = pd.DataFrame(np.column_stack([np.arange(nr), x, np.arange(nr) % 128, np.zeros(nr), ts]), columns=cols)
            with gzip.open(ext / name / f"signal-{i}.csv.gz", "wt") as f:
                df.to_csv(f, index=False)
            ids.append(i)
        mdf = pd.DataFrame({"subject.id": ids, "Group": [R.choice(["Epilepsy", "Control"]) for _ in ids],
                            "Eyes.condition": ["closed-3min-then-open-2min"] * n,
                            "remarks": [R.choice(["", "convulsions since childhood", "noisy"]) for _ in ids], "recordedPeriod": [300] * n})
        mdf.to_csv(ext / f"metadata_{'guineabissau' if 'Guinea' in name else 'nigeria'}.csv", index=False)
    print("fake tree at", root)


if __name__ == "__main__":
    main(sys.argv[1])
