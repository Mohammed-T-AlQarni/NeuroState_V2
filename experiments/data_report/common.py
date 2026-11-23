"""Shared settings and helpers for the EEG data report.

Read-only rule: nothing here opens a data file for writing. Every write goes through
`out_path()` / `write_json()` / `write_csv()`, which refuse any target outside
<root>/Claude_Workspace.
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = r"G:\NeuroState"

# key, path relative to root, family, role, short description
DATASETS = [
    ("TUSZ_train", "NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/edf/train", "TUSZ", "labelled", "TUH Seizure Corpus v2.0.6, train split"),
    ("TUSZ_dev", "NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/edf/dev", "TUSZ", "labelled", "TUH Seizure Corpus v2.0.6, dev split"),
    ("TUSZ_eval", "NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/edf/eval", "TUSZ", "labelled", "TUH Seizure Corpus v2.0.6, eval split"),
    ("TUEP_epilepsy", "TUH/TUEP/00_epilepsy", "TUEP", "labelled", "TUH Epilepsy Corpus v3.1.0, 00_epilepsy"),
    ("TUEP_no_epilepsy", "TUH/TUEP/01_no_epilepsy", "TUEP", "labelled", "TUH Epilepsy Corpus v3.1.0, 01_no_epilepsy"),
    ("TUEG_epilepsy_Y", "TUH/TUH_Additional/TUEG_v2.0.2/epilepsy_Y/edf", "TUEG", "candidates", "TUEG v2.0.2 subset flagged Epilepsy=Y in report metadata"),
    ("TUEG_partial", "TUH/TUH_Additional/TUEG_v2.0.2/edf", "TUEG", "unlabelled", "TUEG v2.0.2 partial download"),
    ("TUSZ_incomplete_copy", "TUH/TUH_Additional/TUSZ_v2.0.6/edf", "TUSZ", "ignore", "Incomplete duplicate of TUSZ (ignored)"),
]
DS = {d[0]: dict(key=d[0], rel=d[1], family=d[2], role=d[3], desc=d[4]) for d in DATASETS}
LABELLED = ["TUSZ_train", "TUSZ_dev", "TUSZ_eval", "TUEP_epilepsy", "TUEP_no_epilepsy"]
MAIN = LABELLED + ["TUEG_epilepsy_Y", "TUEG_partial"]   # datasets analysed in full
EXTERNAL_REL = "External_Datasets"

# Folders searched for metadata / documentation spreadsheets
DOC_DIRS = [
    "TUH/TUEP",
    "TUH/TUH_Additional/TUSZ_v2.0.6/DOCS",
    "NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/DOCS",
    "TUH/TUH_Additional/TUEG_v2.0.2/DOCS",
]

# TUH naming
RE_PID = re.compile(r"^[a-z]{8}$")
RE_EDF = re.compile(r"^(?P<pid>[a-z]{8})_(?P<ses>s\d{3})_(?P<tok>t\d{3})$", re.I)
RE_SES_DIR = re.compile(r"^(?P<ses>s\d{3})(?:_(?P<year>\d{4}))?$", re.I)
RE_MONTAGE_DIR = re.compile(r"^\d{2}_tcp_[a-z_]+$", re.I)
RE_ANY_PID = re.compile(r"(?<![a-z])([a-z]{8})(?:_s\d{3})?(?![a-z])")

STD19 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", ("T3", "T7"), "C3", "CZ", "C4", ("T4", "T8"),
         ("T5", "P7"), "P3", "PZ", "P4", ("T6", "P8"), "O1", "O2"]
STD19_NAMES = [s if isinstance(s, str) else "/".join(s) for s in STD19]

_ROOT: Path | None = None
_WS: Path | None = None


def init(root: str | os.PathLike) -> dict:
    global _ROOT, _WS
    _ROOT = Path(root)
    _WS = _ROOT / "Claude_Workspace"
    return paths()


def paths() -> dict:
    ws = _WS
    return dict(root=_ROOT, ws=ws, reports=ws / "reports", scripts=ws / "reports" / "scripts",
                out=ws / "reports" / "scripts" / "outputs",
                cache=ws / "cache" / "eeg_data_report")


def ds_path(key: str) -> Path:
    return _ROOT.joinpath(*DS[key]["rel"].split("/"))


def rel_path(*parts) -> Path:
    p = _ROOT
    for part in parts:
        p = p.joinpath(*str(part).split("/"))
    return p


def out_path(p: str | os.PathLike) -> Path:
    """Return p after checking it lies inside Claude_Workspace (the only place we write)."""
    p = Path(p).resolve()
    ws = _WS.resolve()
    if p != ws and ws not in p.parents:
        raise RuntimeError(f"Refusing to write outside Claude_Workspace: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _jsonable(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


def write_json(p, obj):
    p = out_path(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_jsonable, ensure_ascii=False)
    return p


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_csv(p, rows: list[dict], fields: list[str] | None = None):
    p = out_path(p)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def read_csv(p) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ext_of(name: str) -> str:
    n = name.lower()
    if n.endswith(".csv_bi"):
        return ".csv_bi"
    for double in (".csv.gz", ".edf.gz", ".tar.gz", ".txt.gz"):
        if n.endswith(double):
            return double
    e = os.path.splitext(n)[1]
    return e if e else "(none)"


def name_pattern(name: str) -> str:
    """Turn a file/folder name into a pattern without identifiers, e.g. aaaaaaac_s001_t000 -> <pid>_s###_t###."""
    s = name
    if RE_MONTAGE_DIR.match(s):
        return "<montage>"
    s = re.sub(r"(?<![a-z])[a-z]{8}(?![a-z])", "<pid>", s)
    s = re.sub(r"s\d{3}_\d{4}", "s###_YYYY", s)
    s = re.sub(r"\d", "#", s)
    s = re.sub(r"#+", lambda m: "#" * min(len(m.group()), 6), s)
    return s


def dist(values) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return dict(n=0)
    def q(p):
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)
    return dict(n=len(v), min=v[0], p5=q(0.05), p25=q(0.25), median=q(0.5), p75=q(0.75), p95=q(0.95),
                max=v[-1], mean=statistics.fmean(v), sum=sum(v))


def top(counter: Counter, n=30, min_count=1) -> list:
    return [[k, c] for k, c in counter.most_common(n) if c >= min_count]


def norm_label(lbl: str) -> str:
    """'EEG FP1-REF' -> 'FP1'; 'EEG T3-LE' -> 'T3'; 'EEG EKG1-REF' -> 'EKG1'."""
    s = lbl.strip().upper()
    for pre in ("EEG ", "POL ", "EMG ", "ECG ", "EOG "):
        if s.startswith(pre):
            s = s[len(pre):]
    s = re.sub(r"-(REF|LE|AVG|AR|M1|M2|A1|A2)$", "", s)
    return s.strip()


def ref_suffix(lbl: str) -> str:
    s = lbl.strip().upper()
    m = re.search(r"-(REF|LE|AVG|AR)$", s)
    return m.group(1) if m else "none"


def channel_category(n: str) -> str:
    if n in {"FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "T7", "C3", "CZ", "C4", "T4", "T8",
             "T5", "P7", "P3", "PZ", "P4", "T6", "P8", "O1", "O2"}:
        return "standard 10-20"
    if n in {"A1", "A2", "M1", "M2"}:
        return "ear/mastoid (A1/A2)"
    if re.match(r"^(EKG|ECG)", n):
        return "EKG/ECG"
    if re.match(r"^EMG", n):
        return "EMG"
    if re.match(r"^(EOG|LOC|ROC|LUC|RLC|LLC|RUC)", n):
        return "EOG/eye"
    if "PHOTIC" in n or n in {"PHO", "PH"}:
        return "photic"
    if n in {"T1", "T2", "SP1", "SP2", "PG1", "PG2", "OZ", "FPZ", "POZ", "NZ"} or re.match(r"^(F|C|P|T|O|AF|FC|CP|PO|FT|TP)\d{1,2}$", n):
        return "other EEG electrode"
    if n in {"EDF ANNOTATIONS"}:
        return "EDF annotations"
    if re.match(r"^(RESP|SAO2|SPO2|PULSE|CO2|EtCO2|DC\d+)", n, re.I):
        return "respiration/oximetry/DC"
    if n in {"IBI", "BURSTS", "SUPPR"}:
        return "derived (IBI/BURSTS/SUPPR)"
    if re.match(r"^\d+$", n):
        return "numbered channel"
    return "other"


def std19_present(names: set[str]) -> list[bool]:
    out = []
    for s in STD19:
        out.append(s in names if isinstance(s, str) else any(x in names for x in s))
    return out


def parse_edf_header(path: str | os.PathLike) -> dict:
    """Read only the EDF/BDF header (256 + ns*256 bytes). Never writes."""
    path = str(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h = f.read(256)
        if len(h) < 256:
            raise ValueError("file shorter than 256-byte header")
        bdf = h[0] == 255
        dec = lambda b: b.decode("latin-1").strip()
        ver = dec(h[1:8]) if bdf else dec(h[0:8])
        patient, recording = dec(h[8:88]), dec(h[88:168])
        startdate, starttime = dec(h[168:176]), dec(h[176:184])
        nbytes = int(dec(h[184:192]) or 0)
        reserved = dec(h[192:236])
        nrec = int(float(dec(h[236:244]) or -1))
        recdur = float(dec(h[244:252]) or 0)
        ns = int(dec(h[252:256]) or 0)
        if ns <= 0 or ns > 1024:
            raise ValueError(f"implausible number of signals {ns}")
        r = f.read(ns * 256)
        if len(r) < ns * 256:
            raise ValueError("signal header truncated")
    def field(off, width):
        return [dec(r[off + i * width: off + (i + 1) * width]) for i in range(ns)]
    o = 0
    labels = field(o, 16); o += 16 * ns
    transd = field(o, 80); o += 80 * ns
    dims = field(o, 8); o += 8 * ns
    pmin = field(o, 8); o += 8 * ns
    pmax = field(o, 8); o += 8 * ns
    dmin = field(o, 8); o += 8 * ns
    dmax = field(o, 8); o += 8 * ns
    prefilt = field(o, 80); o += 80 * ns
    nsamp = [int(float(x or 0)) for x in field(o, 8)]
    fmt = "BDF" if bdf else ("EDF+D" if reserved.startswith("EDF+D") else "EDF+C" if reserved.startswith("EDF+C") else "EDF")
    bps = 3 if bdf else 2
    rec_bytes = sum(nsamp) * bps
    data_bytes = size - 256 - ns * 256
    nrec_from_size = data_bytes / rec_bytes if rec_bytes else 0
    expected = 256 + ns * 256 + (nrec * rec_bytes if nrec > 0 else 0)
    return dict(fmt=fmt, version=ver, patient=patient, recording=recording, startdate=startdate,
                starttime=starttime, header_bytes=nbytes, reserved=reserved, n_records=nrec,
                record_dur=recdur, n_signals=ns, labels=labels, transducers=transd, dims=dims,
                pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax, prefilter=prefilt, nsamp=nsamp,
                size=size, expected_size=expected, size_diff=size - expected if nrec > 0 else None,
                n_records_from_size=nrec_from_size)


def header_year(startdate: str) -> int | None:
    m = re.match(r"^\d\d\.\d\d\.(\d\d)$", startdate or "")
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 85 else 2000 + yy  # EDF spec clipping year 1985


def parse_patient_field(patient: str) -> dict:
    """EDF+ 'local patient' field: code sex birthdate name [extra]. TUH adds 'Age:NN'."""
    toks = patient.split()
    sex = None
    if len(toks) >= 2 and toks[1].upper() in ("M", "F", "X"):
        sex = toks[1].upper()
    m = re.search(r"Age:\s*(\d{1,3})", patient, re.I)
    age = int(m.group(1)) if m else None
    return dict(sex=sex, age=age)


SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "site-packages", ".ipynb_checkpoints",
             "$RECYCLE.BIN", "System Volume Information"}


def iter_tree(root, skipped=None):
    """Yield (dirpath, subdir_names, [(file_name, size)]) using os.scandir (sizes come free on Windows). Read-only."""
    stack = [str(root)]
    while stack:
        dp = stack.pop()
        dirs, files = [], []
        try:
            with os.scandir(dp) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name in SKIP_DIRS:
                                if skipped is not None:
                                    skipped.append(1)
                            else:
                                dirs.append(e.name)
                        elif e.is_file(follow_symlinks=False):
                            files.append((e.name, e.stat(follow_symlinks=False).st_size))
                    except OSError:
                        continue
        except OSError:
            continue
        yield dp, dirs, files
        stack.extend(os.path.join(dp, d) for d in sorted(dirs, reverse=True))


class Log:
    def __init__(self, path):
        self.path = out_path(path)
        self.t0 = time.time()
        self.f = open(self.path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = f"[{time.time() - self.t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def progress(i, n, t0, every=500, label=""):
    if i % every == 0 or i == n:
        el = time.time() - t0
        rate = i / el if el > 0 else 0
        eta = (n - i) / rate if rate > 0 else 0
        print(f"   {label} {i}/{n}  ({rate:.0f}/s, ~{eta/60:.1f} min left)", flush=True)
