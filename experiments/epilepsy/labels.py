"""Label tables for the NeuroState epilepsy study: labels.csv and meta.csv (format: ops repo shared/INTERFACES.md).

Sources (read-only, under --root, default G:\\NeuroState):
    TUEP v3.1.0           TUH/TUEP/00_epilepsy, 01_no_epilepsy (patient folders)
    TUEG Epilepsy = Y     TUH/TUH_Additional/TUEG_v2.0.2/epilepsy_Y/edf (the files flagged Y in metadata v12 / v19)
    TUSZ v2.0.6           NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/edf/{train,dev,eval};
                          Epilepsy = Y in any session of metadata_v06r.xlsx (sheet TUSZ)
    TUAB v3.0.1 normal    Claude_Workspace/inventory/tuab_normal_eeg_manifest.csv (train / eval split) and
                          the TUAB listing (every normal recording, for T3)
    TUEP metadata         TUH/TUEP/DOCS/metadata_v00r.xlsx (only for the conflict check)
    seizure times         the .csv_bi next to each TUSZ / TUEP EDF (seizure_cut.py)
    Nigeria, G.-Bissau    External_Datasets/metadata_*.csv and EEGs_*/ signal CSVs

Rules: every patient once. Group priority: tuep_epi, tuep_ctrl, tueg_y, tusz_y, tuab_train / tuab_eval, tusz_seiz
(TUSZ patients in no diagnosis group: stage 1 only). A patient in an epilepsy group is never normal_eeg. Label
conflicts are flagged, never resolved, in the local file label_conflicts.csv.

Outputs:
    <out>/labels.csv, <out>/meta.csv                      (package files: patient codes, never in git)
    <work>/cohort.csv, recordings.csv, label_conflicts.csv (local working files, never in git)
The console shows aggregate counts only.

Usage:
    python -m experiments.epilepsy.labels --out %NS_DATA% --work %NS_WORKSPACE%\\epilepsy\\work
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

GROUP_LABEL = {"tuep_epi": "epilepsy", "tuep_ctrl": "no_epilepsy", "tueg_y": "epilepsy", "tusz_y": "epilepsy",
               "tuab_train": "normal_eeg", "tuab_eval": "normal_eeg", "tusz_seiz": None}
GROUP_SOURCE = {"tuep_epi": "TUEP", "tuep_ctrl": "TUEP", "tueg_y": "TUEG", "tusz_y": "TUSZ", "tuab_train": "TUAB",
                "tuab_eval": "TUAB", "tusz_seiz": "TUSZ", "nigeria": "Nigeria", "guinea_bissau": "Guinea-Bissau"}
GROUPS = list(GROUP_SOURCE)
EPILEPSY_GROUPS = ("tuep_epi", "tueg_y", "tusz_y")
LABELS_COLS = ["pid", "group", "label"]
META_COLS = ["pid", "group", "source", "n_windows", "minutes", "age", "sex", "year", "montage", "sfreq_orig"]
COHORT_COLS = ["pid", "group", "label", "source", "tusz_split", "tuab_split", "has_tuab_normal", "n_recordings"]
REC_COLS = ["pid", "group", "role", "split", "ses", "tok", "year", "montage", "path"]

REL = dict(
    tuep="TUH/TUEP",
    tuep_meta="TUH/TUEP/DOCS/metadata_v00r.xlsx",
    tueg_y="TUH/TUH_Additional/TUEG_v2.0.2/epilepsy_Y/edf",
    tusz="NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/edf",
    tusz_meta="TUH/TUH_Additional/TUSZ_v2.0.6/DOCS/metadata_v06r.xlsx",
    tuab_manifest="Claude_Workspace/inventory/tuab_normal_eeg_manifest.csv",
    tuab_listing="Claude_Workspace/inventory/tuab_v301_ls_lR.txt",
    ext="External_Datasets",
)
TUSZ_SPLITS = ("train", "dev", "eval")
RE_EDF = re.compile(r"^(?P<pid>[a-z]{8})_(?P<ses>s\d{3})_(?P<tok>t\d{3})\.edf$", re.I)
RE_SES = re.compile(r"^s\d{3}_(\d{4})$", re.I)
RE_MONTAGE = re.compile(r"^\d{2}_tcp_[a-z_]+$", re.I)
RE_PID = re.compile(r"(?<![a-z])([a-z]{8})(?![a-z])")
EPILEPSY_WORDS = re.compile(r"epilep|seizure disorder", re.I)


# ----------------------------------------------------------------------------------------------- readers
def scan_edfs(folder) -> list[dict]:
    """Every EDF below folder with TUH naming -> pid, ses, tok, year (from sNNN_YYYY), montage, path."""
    rows = []
    for dp, _, fns in os.walk(folder):
        parts = Path(dp).parts
        year = next((int(m.group(1)) for p in parts if (m := RE_SES.match(p))), None)
        montage = next((p.lower() for p in parts if RE_MONTAGE.match(p)), "")
        for fn in fns:
            m = RE_EDF.match(fn)
            if m:
                rows.append(dict(pid=m["pid"].lower(), ses=m["ses"].lower(), tok=m["tok"].lower(), year=year,
                                 montage=montage, path=os.path.join(dp, fn)))
    return rows


def tuep_patients(root) -> dict[str, str]:
    out = {}
    for sub, grp in (("00_epilepsy", "tuep_epi"), ("01_no_epilepsy", "tuep_ctrl")):
        d = Path(root) / REL["tuep"] / sub
        for p in sorted(d.iterdir()) if d.is_dir() else []:
            if p.is_dir() and re.fullmatch(r"[a-z]{8}", p.name, re.I):
                out.setdefault(p.name.lower(), grp)
    return out


def read_metadata_sheet(path, sheet) -> pd.DataFrame | None:
    """TUH metadata workbook -> DataFrame with pid, epilepsy (Y/N/...), other (free text). Finds the header row."""
    if not Path(path).is_file():
        return None
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    hdr = next((i for i in range(min(10, len(raw)))
                if any(str(v).strip().lower() == "epilepsy" for v in raw.iloc[i].tolist())), None)
    if hdr is None:
        return None
    cols = [str(c).strip().lower() for c in raw.iloc[hdr]]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    idc = next(c for c in cols if "subject" in c)
    other = next((c for c in cols if c.startswith("other diagnosis")), None)
    pid = df[idc].astype(str).str.lower().map(lambda s: (m.group(1) if (m := RE_PID.search(s)) else None))
    return pd.DataFrame({"pid": pid, "epilepsy": df["epilepsy"].astype(str).str.strip().str.upper(),
                         "other": df[other].fillna("").astype(str) if other else ""}).dropna(subset=["pid"])


def tuab_listing_normal(path) -> dict[str, set]:
    """TUAB v3.0.1 `ls -lR` listing -> {pid: {splits}} for every recording under */normal/."""
    out = defaultdict(set)
    if not Path(path).is_file():
        return out
    cur = None
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.endswith(":"):
                cur = line[:-1]
            elif cur and line.lower().endswith(".edf") and "/edf/" in cur:
                bits = cur.split("/edf/", 1)[1].split("/")
                m = RE_EDF.match(line.split()[-1])
                if m and len(bits) > 1 and bits[1] == "normal":
                    out[m["pid"].lower()].add(bits[0])
    return out


def external_patients(root) -> tuple[list[dict], list[dict]]:
    """Nigeria / Guinea-Bissau: one patient per subject (first session with a signal file).
    Patient codes: ng#### / gb#### (from the dataset's subject number)."""
    base = Path(root) / REL["ext"]
    pats, recs = [], []
    files = {p.name.lower(): p for p in base.rglob("*.csv.gz")} if base.is_dir() else {}
    specs = [("nigeria", "metadata_nigeria.csv", "ng", lambda r: r.get("csv.file", "")),
             ("guinea_bissau", "metadata_guineabissau.csv", "gb", lambda r: f"signal-{r['subject.id']}.csv.gz")]
    for grp, meta, prefix, fname in specs:
        p = base / meta
        if not p.is_file():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        rows.sort(key=lambda r: (int(re.sub(r"\D", "", r["subject.id"]) or 0), int(r.get("session.id") or 1)))
        seen = set()
        for r in rows:
            sid = int(re.sub(r"\D", "", r["subject.id"]) or 0)
            f = files.get(fname(r).lower())
            if sid in seen or f is None:
                continue
            seen.add(sid)
            g = r["Group"].strip().lower()
            label = "epilepsy" if g.startswith("epilep") else "no_epilepsy" if g.startswith("control") else None
            if label is None:
                continue
            pid = f"{prefix}{sid:04d}"
            pats.append(dict(pid=pid, group=grp, label=label))
            recs.append(dict(pid=pid, group=grp, role="diag", split="", ses="s001", tok="t000", year=None,
                             montage="emotiv_14", path=str(f)))
    return pats, recs


# ----------------------------------------------------------------------------------------------- EDF header
def edf_header(path) -> dict:
    """age, sex, year, sfreq of the EEG channels from the EDF header (read-only, header bytes only)."""
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256].decode("latin-1").strip() or 0)
        r = f.read(ns * 256)
    dec = lambda b: b.decode("latin-1").strip()
    patient, startdate = dec(h[8:88]), dec(h[168:176])
    recdur = float(dec(h[244:252]) or 0) or 1.0
    labels = [dec(r[i * 16:(i + 1) * 16]) for i in range(ns)]
    off = ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    nsamp = [int(float(dec(r[off + i * 8: off + (i + 1) * 8]) or 0)) for i in range(ns)]
    eeg = [n / recdur for l, n in zip(labels, nsamp) if l.upper().startswith("EEG")]
    toks = patient.split()
    m = re.search(r"Age:\s*(\d{1,3})", patient, re.I)
    y = re.match(r"^\d\d\.\d\d\.(\d\d)$", startdate)
    return dict(age=int(m.group(1)) if m else None,
                sex=toks[1].upper() if len(toks) > 1 and toks[1].upper() in ("M", "F") else None,
                year=(1900 + int(y.group(1)) if int(y.group(1)) >= 85 else 2000 + int(y.group(1))) if y else None,
                sfreq=max(set(eeg), key=eeg.count) if eeg else None)


# ----------------------------------------------------------------------------------------------- build
def build(root, read_headers=True):
    """-> (cohort DataFrame, recordings DataFrame, conflicts DataFrame, info dict with aggregate counts)."""
    root = Path(root)
    info = {}
    tuep = tuep_patients(root)
    tueg_recs = scan_edfs(root / REL["tueg_y"])
    tueg_y = {r["pid"] for r in tueg_recs}
    tusz_recs = {s: scan_edfs(root / REL["tusz"] / s) for s in TUSZ_SPLITS}
    tusz_split = {}
    for s in TUSZ_SPLITS:
        for r in tusz_recs[s]:
            tusz_split.setdefault(r["pid"], s)
    split_sets = [{r["pid"] for r in tusz_recs[s]} for s in TUSZ_SPLITS]
    info["tusz_patients_in_two_splits"] = sum(sum(p in s for s in split_sets) > 1 for p in tusz_split)
    tusz_meta = read_metadata_sheet(root / REL["tusz_meta"], "TUSZ")
    tuep_meta = read_metadata_sheet(root / REL["tuep_meta"], "TUEP")
    if tusz_meta is None:
        raise SystemExit(f"TUSZ metadata not found or unreadable: {REL['tusz_meta']}")
    tusz_y_any = set(tusz_meta.pid[tusz_meta.epilepsy == "Y"]) & set(tusz_split)
    tusz_y = tusz_y_any - set(tuep) - tueg_y
    info["tusz_y_in_tuep"] = len(tusz_y_any & set(tuep))
    info["tusz_y_in_tueg_y"] = len(tusz_y_any & tueg_y)

    man = pd.read_csv(root / REL["tuab_manifest"], dtype=str)
    tuab_split = {}
    tuab_files = defaultdict(list)
    for r in man.itertuples():
        tuab_split.setdefault(r.patient_id, set()).add(r.tuab_split)
        tuab_files[r.patient_id].append(r.local_edf_path)
    listing_normal = tuab_listing_normal(root / REL["tuab_listing"])
    has_tuab_normal = set(listing_normal) | set(tuab_split)

    # group assignment by priority
    group = {}
    for p, g in tuep.items():
        group[p] = g
    for p in sorted(tueg_y):
        group.setdefault(p, "tueg_y")
    for p in sorted(tusz_y):
        group.setdefault(p, "tusz_y")
    removed_from_tuab = 0
    for p, splits in sorted(tuab_split.items()):
        if p in group:
            removed_from_tuab += 1
            continue
        group[p] = "tuab_eval" if splits == {"eval"} else "tuab_train" if splits == {"train"} else "tuab_conflict"
    info["tuab_manifest_patients_already_in_other_group"] = removed_from_tuab
    for p in sorted(tusz_split):
        group.setdefault(p, "tusz_seiz")

    # conflicts (flagged, never resolved)
    conflicts = []
    meta_y = set()
    for name, m in (("TUEP metadata_v00r", tuep_meta), ("TUSZ metadata_v06r", tusz_meta)):
        if m is None:
            continue
        for p in sorted(set(m.pid[m.epilepsy == "Y"])):
            meta_y.add(p)
            if group.get(p) == "tuep_ctrl":
                conflicts.append(dict(pid=p, group="tuep_ctrl", issue=f"Epilepsy = Y in {name}"))
        for p, txt in zip(m.pid, m.other):
            if group.get(p) in ("tuab_train", "tuab_eval") and EPILEPSY_WORDS.search(txt or ""):
                conflicts.append(dict(pid=p, group=group[p], issue=f"free text in {name} mentions epilepsy"))
    for p in sorted(p for p, g in group.items() if g in ("tuab_train", "tuab_eval") and p in meta_y):
        conflicts.append(dict(pid=p, group=group[p], issue="Epilepsy = Y in a metadata sheet"))
    for p in sorted(p for p, g in group.items() if g == "tuab_conflict"):
        conflicts.append(dict(pid=p, group=g, issue="TUAB normal in both train and eval split"))
    for p in sorted(p for p, g in group.items() if g == "tuep_ctrl" and p in tueg_y):
        conflicts.append(dict(pid=p, group=g, issue="TUEP control in the TUEG Epilepsy = Y folder"))
    conflicts = pd.DataFrame(conflicts, columns=["pid", "group", "issue"]).drop_duplicates()

    # recordings per patient: from the patient's group source only ("diag"); TUSZ files also as "seizure"
    recs = []
    tuep_root = root / REL["tuep"]
    for sub, g in (("00_epilepsy", "tuep_epi"), ("01_no_epilepsy", "tuep_ctrl")):
        for r in scan_edfs(tuep_root / sub):
            if group.get(r["pid"]) == g:
                recs.append(dict(r, group=g, role="diag", split=""))
    for r in tueg_recs:
        if group.get(r["pid"]) == "tueg_y":
            recs.append(dict(r, group="tueg_y", role="diag", split=""))
    for s in TUSZ_SPLITS:
        for r in tusz_recs[s]:
            g = group.get(r["pid"])
            if g == "tusz_y":
                recs.append(dict(r, group=g, role="diag", split=s))
            if g not in ("tuep_epi", "tuep_ctrl"):
                recs.append(dict(r, group=g, role="seizure", split=s))
    for p, paths in tuab_files.items():
        g = group.get(p)
        if g in ("tuab_train", "tuab_eval"):
            for path in paths:
                m = RE_EDF.match(os.path.basename(path))
                parts = Path(path).parts
                recs.append(dict(pid=p, group=g, role="diag", split=g.split("_")[1], ses=m["ses"] if m else "",
                                 tok=m["tok"] if m else "", path=path,
                                 year=next((int(x.group(1)) for q in parts if (x := RE_SES.match(q))), None),
                                 montage=next((q.lower() for q in parts if RE_MONTAGE.match(q)), "")))
    ext_pats, ext_recs = external_patients(root)
    recs += ext_recs
    recs = pd.DataFrame(recs, columns=REC_COLS).sort_values(["pid", "role", "ses", "tok"]).reset_index(drop=True)

    rows = []
    for p, g in sorted(group.items()):
        rows.append(dict(pid=p, group=g, label=GROUP_LABEL.get(g), source=GROUP_SOURCE.get(g, ""),
                         tusz_split=tusz_split.get(p, ""), tuab_split="|".join(sorted(tuab_split.get(p, []))),
                         has_tuab_normal=p in has_tuab_normal))
    for e in ext_pats:
        rows.append(dict(e, source=GROUP_SOURCE[e["group"]], tusz_split="", tuab_split="", has_tuab_normal=False))
    cohort = pd.DataFrame(rows)
    n_rec = recs[recs.role == "diag"].groupby("pid").size()
    cohort["n_recordings"] = cohort.pid.map(n_rec).fillna(0).astype(int)
    cohort = cohort[COHORT_COLS]
    if not cohort.pid.is_unique:
        raise AssertionError("a patient appears twice in the cohort")
    info["patients_without_recordings"] = dict(Counter(cohort.group[(cohort.n_recordings == 0) & (cohort.group != "tusz_seiz")]))
    info["conflicts"] = dict(Counter(conflicts.issue)) if len(conflicts) else {}
    return cohort, recs, conflicts, info


def make_meta(cohort, recs, read_headers=True) -> pd.DataFrame:
    """meta.csv rows: age / sex / year / sfreq from the first recording's EDF header; n_windows / minutes are
    filled in by package.py after preprocessing."""
    first = recs.sort_values(["pid", "role", "ses", "tok"]).groupby("pid").first()
    rows = []
    for c in cohort.itertuples():
        r = first.loc[c.pid] if c.pid in first.index else None
        h = {}
        if read_headers and r is not None and str(r.path).lower().endswith(".edf"):
            try:
                h = edf_header(r.path)
            except (OSError, ValueError):
                h = {}
        year = (r.year if r is not None and pd.notna(r.year) else None) or h.get("year")
        rows.append(dict(pid=c.pid, group=c.group, source=c.source, n_windows="", minutes="",
                         age=h.get("age"), sex=h.get("sex"), year=year,
                         montage=(r.montage if r is not None else ""), sfreq_orig=h.get("sfreq")))
    return pd.DataFrame(rows, columns=META_COLS)


def write(cohort, recs, conflicts, meta, out, work):
    out, work = Path(out), Path(work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    lab = cohort[cohort.label.notna() & (cohort.group != "tusz_seiz")][LABELS_COLS].sort_values("pid")
    lab.to_csv(out / "labels.csv", index=False)
    meta.sort_values("pid").to_csv(out / "meta.csv", index=False)
    cohort.to_csv(work / "cohort.csv", index=False)
    recs.to_csv(work / "recordings.csv", index=False)
    conflicts.to_csv(work / "label_conflicts.csv", index=False)


def summary(cohort, recs, info) -> str:
    lines = [f"{'group':<15}{'patients':>9}{'recordings':>11}"]
    d = recs[recs.role == "diag"].groupby("group").size()
    for g, n in cohort.group.value_counts().reindex(GROUPS + ["tuab_conflict"]).dropna().items():
        lines.append(f"{g:<15}{int(n):>9}{int(d.get(g, 0)):>11}")
    lab = cohort[cohort.group != "tusz_seiz"].label.value_counts()
    lines.append("labels: " + ", ".join(f"{k} {v}" for k, v in lab.items()))
    for k, v in info.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _guard(path, root):
    """Refuse to write inside the raw-data root, except in its Claude_Workspace."""
    p, r = Path(path).resolve(), Path(root).resolve()
    if (p == r or r in p.parents) and not (r / "Claude_Workspace" == p or (r / "Claude_Workspace") in p.parents):
        raise SystemExit(f"Refusing to write inside the raw data: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"G:\NeuroState")
    ap.add_argument("--out", required=True, help="package folder (labels.csv, meta.csv)")
    ap.add_argument("--work", required=True, help="local working folder (cohort, recordings, conflicts)")
    ap.add_argument("--no-headers", action="store_true", help="skip reading EDF headers (age, sex, sfreq)")
    a = ap.parse_args()
    for p in (a.out, a.work):
        _guard(p, a.root)
    cohort, recs, conflicts, info = build(a.root)
    meta = make_meta(cohort, recs, read_headers=not a.no_headers)
    write(cohort, recs, conflicts, meta, a.out, a.work)
    print(summary(cohort, recs, info))
    print(f"label conflicts flagged: {len(conflicts)} (details in {Path(a.work) / 'label_conflicts.csv'}, local only)")


if __name__ == "__main__":
    main()
