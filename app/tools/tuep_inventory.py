"""Recording inventory for the TUH EEG Epilepsy Corpus (TUEP), before any training.

  experiments/.venv/bin/python app/tools/tuep_inventory.py --root <path to tuh_eeg_epilepsy/v3.1.0> \
      [--out data/tuep/inventory] [--no-hash] [--overlap NAME=ids.txt ...] \
      [--meta-sheet S --meta-patient-col P --meta-session-col Q]

What it reads (layout from the official AAREADME.txt of v3.1.0; checked again on the real data):
  0?_<class>/<patient>/<session>/<montage>/<patient>_<session>_<token>.edf   00 = epilepsy, 01 = no epilepsy
  the same stem with .csv (event annotations) and .csv_bi (bckg / seiz)
  DOCS/metadata_v00r.xlsx                         session-level clinical metadata (columns discovered here)
  DOCS/sessions_common_with_tusz.list, DOCS/sessions_unique_to_tuep.list

What it writes (under --out, which defaults to the git-ignored data/ folder: TUH data and metadata are under a data
use agreement and must never be committed):
  inventory_recordings.csv   one row per EDF: ids, label, montage, header facts, channel check, seizures, TUSZ flag,
                             hashes, processing status
  inventory_sessions.csv     one row per session, with every metadata column linked (prefixed meta_)
  inventory_patients.csv     one row per patient: label, sessions, files, hours, seizures, overlap flags
  inventory_checks.json      counts against the README, label conflicts, duplicates, metadata linkage, confounds
  inventory_summary.md       the same, for reading

The label of a patient is its class folder. The spreadsheet is linked and cross-checked, never used to relabel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# AAREADME.txt v3.1.0, "BASIC STATISTICS"
EXPECTED = {"epilepsy": {"patients": 100, "sessions": 530, "files": 2244},
            "no_epilepsy": {"patients": 100, "sessions": 168, "files": 564}}
CLASS_CODE = {"00": ("epilepsy", "E"), "01": ("no_epilepsy", "N")}
PATH_RE = re.compile(r"^(?P<cls>0[01])_[^/]+/(?P<patient>[^/]+)/(?P<session>[^/]+)/(?P<montage>[^/]+)/(?P<file>[^/]+\.edf)$",
                     re.IGNORECASE)
SESSION_RE = re.compile(r"^(?P<sid>s\d{3})(?:_(?P<year>\d{4}))?", re.IGNORECASE)
TOKEN_RE = re.compile(r"_(?P<token>t\d{3})$", re.IGNORECASE)


# ----------------------------------------------------------------------------------------------- parsing
def parse_path(rel: str) -> dict | None:
    m = PATH_RE.match(rel.replace("\\", "/"))
    if not m:
        return None
    label, code = CLASS_CODE[m["cls"]]
    s = SESSION_RE.match(m["session"])
    stem = m["file"][:-4]
    t = TOKEN_RE.search(stem)
    return {"recording_id": stem, "path": rel.replace("\\", "/"), "class_folder": rel.split("/")[0], "label": label,
            "group": code, "patient_id": m["patient"], "session_folder": m["session"],
            "session_id": (s["sid"].lower() if s else m["session"]),
            "session_year": (int(s["year"]) if s and s["year"] else None),
            "montage_config": m["montage"], "token": (t["token"].lower() if t else None)}


def read_seizures(edf: Path) -> tuple[list[tuple[float, float]], str]:
    """Seizure intervals (s from file start) from <stem>.csv_bi ('seiz' rows), else from <stem>.csv (any label that
    is not 'bckg', union over channels). Format as in TUSZ: '#' comment lines, then a 'channel,start_time,...' header."""
    for ext, is_bi in ((".csv_bi", True), (".csv", False)):
        f = edf.with_suffix(ext)
        if not f.exists():
            continue
        rows, header = [], None
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cells = [c.strip() for c in line.split(",")]
            if header is None:
                header = [c.lower() for c in cells]
                continue
            r = dict(zip(header, cells))
            lab = r.get("label", "").lower()
            if (is_bi and lab == "seiz") or (not is_bi and lab and lab != "bckg"):
                rows.append((float(r["start_time"]), float(r["stop_time"])))
        rows.sort()
        merged: list[tuple[float, float]] = []
        for a, b in rows:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged, ext[1:]
    return [], "none"


def hashes(path: Path) -> tuple[str, str]:
    """SHA-256 of the whole file, and of the data records only (bytes after the EDF header): the second matches the
    same signal even when the header was edited (TUH re-wrote headers in v2.0.1)."""
    full, data = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as f:
        head = f.read(256)
        n = int(head[184:192].decode("ascii", "replace").strip() or 256)
        rest_head = f.read(max(n - 256, 0))
        full.update(head), full.update(rest_head)
        for chunk in iter(lambda: f.read(1 << 20), b""):
            full.update(chunk), data.update(chunk)
    return full.hexdigest(), data.hexdigest()


def read_session_list(path: Path, known: set[str]) -> set[tuple[str, str]]:
    """(patient, session) pairs named on each line of a DOCS *.list file, whatever the exact line format."""
    out = set()
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        low = line.strip().lower()
        pats = [p for p in re.split(r"[^a-z0-9]+", low) if p in known]
        ses = re.findall(r"(?<![a-z0-9])(s\d{3})", low)
        if pats and ses:
            out.add((pats[0], ses[0]))
    return out


def read_ids(path: Path, known: set[str]) -> set[str]:
    txt = Path(path).read_text(errors="replace").lower()
    toks = set(re.split(r"[^a-z0-9]+", txt))
    return toks & known if known else toks - {""}


# ----------------------------------------------------------------------------------------------- metadata
def _norm(v) -> str:
    return "" if pd.isna(v) else str(v).strip().lower()


def link_metadata(xlsx: Path, sessions: pd.DataFrame, sheet=None, patient_col=None, session_col=None):
    """Join the session-level spreadsheet to the inventory's (patient_id, session_id). The key columns are found by
    how many of their values match the inventory, unless given. Returns (session table with meta_* columns, report)."""
    report = {"file": str(xlsx), "found": xlsx.exists()}
    if not xlsx.exists():
        return sessions.assign(meta_linked=False), report
    book = pd.read_excel(xlsx, sheet_name=None, dtype=object)
    report["sheets"] = {k: [str(c) for c in v.columns] for k, v in book.items()}
    pats = set(sessions.patient_id.str.lower())
    pairs = set(zip(sessions.patient_id.str.lower(), sessions.session_id.str.lower()))

    def as_session(v):
        s = _norm(v)
        m = re.search(r"(?<![a-z0-9])(s\d{3})", s)
        if m:
            return m.group(1)
        return f"s{int(float(s)):03d}" if re.fullmatch(r"\d+(\.0+)?", s) else None

    best = None
    for name, df in book.items():
        if sheet and name != sheet:
            continue
        cols = list(df.columns)
        combo = {}
        for c in cols:  # one column holding both ids (e.g. 'aaaaaanr_s001' or a path)
            vals = [_norm(v) for v in df[c]]
            hits = 0
            for v in vals:
                toks = re.split(r"[^a-z0-9]+", v)
                p = next((t for t in toks if t in pats), None)
                s = re.search(r"(?<![a-z0-9])(s\d{3})", v)
                hits += bool(p and s and (p, s.group(1)) in pairs)
            combo[c] = hits
        pc = patient_col or max(cols, key=lambda c: sum(_norm(v) in pats for v in df[c]))
        sc = session_col or max([c for c in cols if c != pc] or cols, key=lambda c: sum(as_session(v) is not None for v in df[c]))
        sep_hits = sum((_norm(p), as_session(s)) in pairs for p, s in zip(df[pc], df[sc]))
        cc = max(combo, key=combo.get) if combo else None
        cand = ("combined", cc, None, combo.get(cc, 0)) if cc and combo[cc] > sep_hits and not (patient_col or session_col) \
            else ("separate", pc, sc, sep_hits)
        if best is None or cand[3] > best[1][3]:
            best = (name, cand)
    name, (mode, c1, c2, hits) = best
    df = book[name].copy()
    if mode == "combined":
        keys = []
        for v in df[c1]:
            v = _norm(v)
            p = next((t for t in re.split(r"[^a-z0-9]+", v) if t in pats), None)
            s = re.search(r"(?<![a-z0-9])(s\d{3})", v)
            keys.append((p, s.group(1) if s else None))
    else:
        keys = [(_norm(p), as_session(s)) for p, s in zip(df[c1], df[c2])]
    df["_key"] = keys
    dup = df[df.duplicated("_key", keep=False) & df["_key"].map(lambda k: k in pairs)]
    meta = df.drop_duplicates("_key", keep="first").set_index("_key")
    meta.columns = [f"meta_{c}" for c in meta.columns]
    out = sessions.copy()
    key = list(zip(out.patient_id.str.lower(), out.session_id.str.lower()))
    out["meta_linked"] = [k in meta.index for k in key]
    out = pd.concat([out.reset_index(drop=True), meta.reindex(key).reset_index(drop=True)], axis=1)
    unmatched_meta = [k for k in meta.index if k not in pairs]
    report.update({"sheet": name, "key_mode": mode, "key_columns": [str(c) for c in (c1, c2) if c is not None],
                   "rows": int(len(df)), "sessions_linked": int(out.meta_linked.sum()), "sessions_total": int(len(out)),
                   "sessions_without_metadata": int((~out.meta_linked).sum()),
                   "metadata_rows_without_eeg": len(unmatched_meta), "examples_metadata_without_eeg": [list(map(str, k)) for k in unmatched_meta[:10]],
                   "duplicate_metadata_keys": int(dup["_key"].nunique())})
    # values of diagnosis-like columns by class folder, for review only (never used to relabel)
    review = {}
    for c in meta.columns:
        if re.search(r"diagnos|epilep|impression|seiz", c, re.I):
            review[c] = {str(lab): {str(k): int(v) for k, v in g[c].dropna().astype(str).str.strip().value_counts().head(8).items()}
                         for lab, g in out.groupby("label")}
    report["review_by_class"] = review
    return out, report


# ----------------------------------------------------------------------------------------------- build
def build(root: Path, out: Path, do_hash=True, overlaps: dict[str, Path] | None = None, meta_args=None, status_csv=None) -> dict:
    from app import pipeline_adapter as pa
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, unparsed = [], []
    for edf in sorted(p for p in root.rglob("*") if p.suffix.lower() == ".edf"):
        rel = edf.relative_to(root).as_posix()
        r = parse_path(rel)
        if r is None:
            unparsed.append(rel)
            continue
        chk = pa.check_file(str(edf))
        sz, src = read_seizures(edf)
        r.update({"bytes": edf.stat().st_size, "edf_type": chk.get("edf_type"), "sfreq": chk.get("sfreq"),
                  "duration_s": chk.get("duration_s"), "n_channels": chk.get("n_channels"),
                  "standard_found": 19 - len(chk.get("missing", list(range(19)))) if "missing" in chk else None,
                  "missing_channels": " ".join(chk.get("missing", [])), "compatible": bool(chk["ok"]),
                  "problems": " | ".join(p["message"] for p in chk.get("problems", [])),
                  "annotation_file": src, "n_seizures": len(sz), "seizure_s": round(sum(b - a for a, b in sz), 3),
                  "seizure_intervals": ";".join(f"{a:.3f}-{b:.3f}" for a, b in sz)})
        if do_hash:
            r["sha256"], r["data_sha256"] = hashes(edf)
        rows.append(r)
    rec = pd.DataFrame(rows)
    if rec.empty:
        raise SystemExit(f"No EDF file under {root} matches the TUEP layout 0?_*/<patient>/<session>/<montage>/*.edf "
                         f"({len(unparsed)} other EDF files found).")
    known = set(rec.patient_id.str.lower())
    docs = root / "DOCS"
    common = read_session_list(docs / "sessions_common_with_tusz.list", known)
    unique = read_session_list(docs / "sessions_unique_to_tuep.list", known)
    have_lists = (docs / "sessions_common_with_tusz.list").exists() or (docs / "sessions_unique_to_tuep.list").exists()
    key = list(zip(rec.patient_id.str.lower(), rec.session_id))
    rec["in_tusz"] = [("yes" if k in common else "no" if k in unique else "unlisted") if have_lists else "unknown" for k in key]
    for name, path in (overlaps or {}).items():
        ids = read_ids(path, known)
        rec[f"overlap_{name}"] = rec.patient_id.str.lower().isin(ids)
    rec["status"], rec["status_reason"], rec["prepared_s"] = "pending", "", 0.0
    st_path = Path(status_csv) if status_csv else out.parent / "prepared" / "status.csv"
    if st_path.exists():
        st = pd.read_csv(st_path).set_index("recording_path")
        for c in ("status", "status_reason", "prepared_s"):
            rec[c] = rec.path.map(st[c]).fillna(rec[c])

    # sessions and metadata
    ses = rec.groupby(["patient_id", "session_id", "session_folder", "label"], as_index=False).agg(
        n_files=("path", "size"), duration_s=("duration_s", "sum"), seizure_s=("seizure_s", "sum"),
        in_tusz=("in_tusz", "first"), montages=("montage_config", lambda s: " ".join(sorted(set(s)))))
    ma = meta_args or {}
    ses, meta_report = link_metadata(docs / "metadata_v00r.xlsx", ses, ma.get("sheet"), ma.get("patient_col"), ma.get("session_col"))
    ses_cols = ["patient_id", "session_id", "session_folder"]
    rec = rec.merge(ses[ses_cols + ["meta_linked"]], on=ses_cols, how="left")

    # patients
    pat = rec.groupby("patient_id").agg(
        labels=("label", lambda s: " ".join(sorted(set(s)))), group=("group", lambda s: "".join(sorted(set(s)))),
        n_sessions=("session_id", "nunique"), n_files=("path", "size"), hours=("duration_s", lambda s: s.sum() / 3600),
        n_compatible=("compatible", "sum"), n_seizure_files=("n_seizures", lambda s: int((s > 0).sum())),
        seizure_s=("seizure_s", "sum"), sessions_in_tusz=("in_tusz", lambda s: int((s == "yes").sum())),
        sfreqs=("sfreq", lambda s: " ".join(str(int(x)) if pd.notna(x) else "?" for x in sorted(set(s)))),
        montages=("montage_config", lambda s: " ".join(sorted(set(s)))),
        prepared_s=("prepared_s", "sum")).reset_index()
    pat["label_conflict"] = pat["labels"].str.contains(" ")
    for c in [c for c in rec.columns if c.startswith("overlap_")]:
        pat[c] = pat.patient_id.map(rec.groupby("patient_id")[c].any())

    # checks
    checks = {"root": str(root), "edf_files": int(len(rec)), "edf_files_outside_layout": unparsed[:50],
              "n_edf_outside_layout": len(unparsed), "expected_from_readme": EXPECTED, "observed": {}}
    for lab in ("epilepsy", "no_epilepsy"):
        r = rec[rec.label == lab]
        checks["observed"][lab] = {"patients": int(r.patient_id.nunique()), "sessions": int(r.groupby(["patient_id", "session_id"]).ngroups),
                                   "files": int(len(r))}
    checks["counts_match_readme"] = checks["observed"] == EXPECTED
    checks["label_conflicts"] = pat.loc[pat.label_conflict, "patient_id"].tolist()
    if do_hash:
        def dup_groups(col):
            g = rec.groupby(col).filter(lambda d: len(d) > 1).groupby(col)
            return [{"hash": h[:16], "files": d.path.tolist(), "patients": sorted(set(d.patient_id)),
                     "cross_patient": d.patient_id.nunique() > 1} for h, d in g]
        checks["duplicate_files"] = dup_groups("sha256")
        checks["duplicate_signals_different_header"] = [d for d in dup_groups("data_sha256")
                                                        if d["hash"] not in {x["hash"] for x in checks["duplicate_files"]}]
    checks["incompatible_files"] = int((~rec.compatible).sum())
    checks["incompatible_reasons"] = Counter(p.split(" | ")[0] for p in rec.loc[~rec.compatible, "problems"]).most_common(10)
    checks["tusz_lists_found"] = have_lists
    checks["sessions_by_tusz_flag"] = ses.in_tusz.value_counts().to_dict()
    checks["metadata"] = meta_report
    checks["confounds_by_class"] = confounds(rec, pat)
    rec.to_csv(out / "inventory_recordings.csv", index=False)
    ses.to_csv(out / "inventory_sessions.csv", index=False)
    pat.to_csv(out / "inventory_patients.csv", index=False)
    (out / "inventory_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    (out / "inventory_summary.md").write_text(summary_md(checks))
    return checks


def confounds(rec: pd.DataFrame, pat: pd.DataFrame) -> dict:
    """Recording properties that differ between the classes could let a model separate them without the EEG
    pattern of interest. Reported for review before training."""
    out = {}
    for lab, r in rec.groupby("label"):
        p = pat[pat.labels == lab]
        out[lab] = {"patients": int(len(p)), "hours_total": round(float(r.duration_s.sum() / 3600), 1),
                    "hours_per_patient_median": round(float(p.hours.median()), 2),
                    "sessions_per_patient_mean": round(float(p.n_sessions.mean()), 2),
                    "sampling_rates": {str(int(k)) if pd.notna(k) else "?": int(v) for k, v in r.sfreq.value_counts().items()},
                    "montage_configs": r.montage_config.value_counts().to_dict(),
                    "session_years": [int(r.session_year.min()), int(r.session_year.max())] if r.session_year.notna().any() else None,
                    "files_with_seizures": int((r.n_seizures > 0).sum()), "seizure_hours": round(float(r.seizure_s.sum() / 3600), 2),
                    "compatible_files_pct": round(float(100 * r.compatible.mean()), 1)}
    return out


def summary_md(c: dict) -> str:
    o, e = c["observed"], c["expected_from_readme"]
    L = ["# TUEP recording inventory", "", f"Root: `{c['root']}` · {c['edf_files']} EDF files in the TUEP layout"
         + (f", {c['n_edf_outside_layout']} outside it" if c["n_edf_outside_layout"] else "") + ".", "",
         "| Class | Patients | Sessions | Files | README patients / sessions / files |", "|---|---|---|---|---|"]
    for lab in ("epilepsy", "no_epilepsy"):
        L.append(f"| {lab} | {o[lab]['patients']} | {o[lab]['sessions']} | {o[lab]['files']} | "
                 f"{e[lab]['patients']} / {e[lab]['sessions']} / {e[lab]['files']} |")
    L += ["", f"Counts match the README: **{c['counts_match_readme']}**.",
          f"Patients under both class folders: **{len(c['label_conflicts'])}** {c['label_conflicts'][:10]}"]
    if "duplicate_files" in c:
        xp = [d for d in c["duplicate_files"] + c["duplicate_signals_different_header"] if d["cross_patient"]]
        L.append(f"Duplicate files: {len(c['duplicate_files'])} groups; same signal with a different header: "
                 f"{len(c['duplicate_signals_different_header'])} groups; **across patients: {len(xp)}**.")
    L.append(f"Incompatible files: {c['incompatible_files']} {c['incompatible_reasons']}")
    L.append(f"TUSZ lists found: {c['tusz_lists_found']} · sessions by flag: {c['sessions_by_tusz_flag']}")
    m = c["metadata"]
    if m.get("found"):
        L += ["", "## Metadata spreadsheet", f"Sheet `{m.get('sheet')}`, key {m.get('key_mode')} {m.get('key_columns')}: "
              f"{m.get('sessions_linked')}/{m.get('sessions_total')} sessions linked; {m.get('metadata_rows_without_eeg')} metadata rows "
              f"without EEG; {m.get('duplicate_metadata_keys')} duplicated keys.", "", "Columns per sheet:"]
        L += [f"- `{s}`: {', '.join(cols)}" for s, cols in m.get("sheets", {}).items()]
    else:
        L += ["", "Metadata spreadsheet not found at DOCS/metadata_v00r.xlsx."]
    L += ["", "## Possible confounds by class (review before training)", "```", json.dumps(c["confounds_by_class"], indent=1), "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="the TUEP release folder (contains 00_*, 01_*, DOCS)")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "tuep" / "inventory")
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 (faster; disables the duplicate checks)")
    ap.add_argument("--overlap", action="append", default=[], metavar="NAME=PATH",
                    help="a file listing TUH patient ids of another source (e.g. TUSZ); flags those patients")
    ap.add_argument("--meta-sheet"), ap.add_argument("--meta-patient-col"), ap.add_argument("--meta-session-col")
    ap.add_argument("--status", type=Path, help="status.csv from tuep_prepare.py (default: <out>/../prepared/status.csv)")
    a = ap.parse_args(argv)
    ov = dict(x.split("=", 1) for x in a.overlap)
    c = build(a.root, a.out, not a.no_hash, {k: Path(v) for k, v in ov.items()},
              {"sheet": a.meta_sheet, "patient_col": a.meta_patient_col, "session_col": a.meta_session_col}, a.status)
    print((a.out / "inventory_summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
