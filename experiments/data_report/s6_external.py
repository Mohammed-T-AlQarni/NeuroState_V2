"""Step 6: External_Datasets (Guinea-Bissau and Nigeria). Reads ALL .gz files (streamed, read-only):
inner format, columns/channels, rows, duration; metadata CSVs: fields, % missing, group counts.
Amplitude check on a sample. Aggregate output: outputs/s6_external.json
"""
from __future__ import annotations

import argparse
import gzip
import io
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import common as C
from s4_metadata import KEYWORDS, cat_counts, col_kind, keyword_columns, may_list_values, missing_mask

EMOTIV14 = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def which(path_str):
    s = path_str.lower()
    if "guinea" in s or "bissau" in s:
        return "Guinea-Bissau"
    if "nigeria" in s:
        return "Nigeria"
    return "other"


def scan_gz(p):
    out = dict(ok=0)
    try:
        with gzip.open(p, "rb") as f:
            head = f.read(4096)
            if head[:8] == b"0       ":
                ns = int(head[252:256].decode("latin-1").strip())
                out.update(ok=1, inner="EDF", n_signals=ns)
                return out
            text_head = head.decode("utf-8", "replace")
            first = text_head.splitlines()[0]
            cols = [c.strip().strip('"') for c in first.split(",")]
            f.seek(0)
            n_lines, last, second = 0, b"", b""
            for chunk_line in f:
                n_lines += 1
                if n_lines == 2:
                    second = chunk_line
                last = chunk_line
        out.update(ok=1, inner="CSV", columns=cols, n_rows=n_lines - 1)
        tcol = next((i for i, c in enumerate(cols) if c.lower() in ("timestamp", "time", "time_stamp")), None)
        if tcol is not None and n_lines > 2:
            try:
                t1 = float(second.decode().split(",")[tcol])
                t2 = float(last.decode().split(",")[tcol])
                if t2 > t1:
                    out["fs_from_timestamp"] = (n_lines - 2) / (t2 - t1)
                    out["timestamp_span_s"] = t2 - t1
            except Exception:
                pass
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def amp_sample(p, fs=128, seconds=60):
    df = pd.read_csv(p, compression="gzip", nrows=int(fs * seconds * 3))
    cols = [c for c in df.columns if str(c).strip().strip('"').upper() in EMOTIV14]
    if not cols:
        return None
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    mid = len(x) // 2
    w = x[max(0, mid - fs * seconds // 2): mid + fs * seconds // 2]
    w = w - np.nanmean(w, axis=0)
    sd = np.nanstd(w, axis=0)
    return dict(median_sd=float(np.nanmedian(sd)), n_flat=int((sd < 0.5).sum()), n_chan=len(cols),
                max_abs=float(np.nanmax(np.abs(w))))


def summarize_meta(p):
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    out = dict(file=p.name, n_rows=len(df), columns=[])
    kinds = {c: col_kind(df[c], c) for c in df.columns}
    for c in df.columns:
        e = dict(name=c, kind=kinds[c], pct_missing=round(float(missing_mask(df[c]).mean() * 100), 1))
        if kinds[c] == "categorical" and may_list_values(c, df[c]):
            e["counts"] = cat_counts(df[c])
        elif kinds[c] == "numeric":
            d = C.dist(pd.to_numeric(df[c], errors="coerce").dropna().tolist())
            d.pop("sum", None)
            e["dist"] = d
        else:
            e["n_distinct"] = int(df[c].dropna().astype(str).nunique())
        out["columns"].append(e)
    gcol = next((c for c in df.columns if c.lower() in ("group", "diagnosis", "class", "label")), None)
    if gcol:
        textcols = [c for c in keyword_columns(df, kinds, None) if c != gcol]
        out["group_column"] = gcol
        out["group_counts"] = cat_counts(df[gcol])
        kw = {}
        for g, sub in df.groupby(df[gcol].astype(str).str.strip()):
            txt = sub[textcols].fillna("").astype(str).agg(" ".join, axis=1).str.lower() if textcols else pd.Series(dtype=str)
            kw[g] = {k: int(txt.map(lambda t: bool(re.search(rx, t))).sum()) for k, rx in KEYWORDS.items()}
            kw[g] = {k: v for k, v in kw[g].items() if v}
        out["keyword_rows_by_group"] = kw
        out["keyword_text_columns"] = textcols
    return out


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 6 external datasets")
    root = C.rel_path(C.EXTERNAL_REL)
    if not root.is_dir():
        C.write_json(P["out"] / "s6_external.json", dict(found=False))
        log("  External_Datasets not found")
        return
    files = [p for p in root.rglob("*") if p.is_file()]
    res = defaultdict(lambda: dict(n_gz=0, gz_bytes=0, inner=Counter(), errors=0, column_sets=Counter(), rows=[],
                                   fs=[], span=[], other_files=Counter()))
    t0 = time.time()
    gzs = [p for p in files if p.name.lower().endswith(".gz")]
    for i, p in enumerate(gzs, 1):
        d = res[which(str(p.relative_to(root)))]
        d["n_gz"] += 1
        d["gz_bytes"] += p.stat().st_size
        s = scan_gz(p)
        if not s["ok"]:
            d["errors"] += 1
            continue
        d["inner"][s["inner"]] += 1
        if s["inner"] == "CSV":
            d["column_sets"]["|".join(s["columns"])] += 1
            d["rows"].append(s["n_rows"])
            if "fs_from_timestamp" in s:
                d["fs"].append(s["fs_from_timestamp"])
                d["span"].append(s["timestamp_span_s"])
        C.progress(i, len(gzs), t0, 25, "gz")
    for p in files:
        if not p.name.lower().endswith(".gz"):
            res[which(str(p.relative_to(root)))]["other_files"][C.ext_of(p.name)] += 1
    rng = random.Random(args.seed)
    out = {}
    for k, d in res.items():
        mine = [p for p in gzs if which(str(p.relative_to(root))) == k]
        amps = []
        for p in rng.sample(mine, min(args.amp_sample, len(mine))):
            try:
                a = amp_sample(p)
                if a:
                    amps.append(a)
            except Exception:
                pass
        fs_med = float(np.median(d["fs"])) if d["fs"] else None
        fs_use = fs_med or 128.0
        cs = d["column_sets"].most_common(3)
        out[k] = dict(
            n_gz=d["n_gz"], gz_bytes=d["gz_bytes"], inner_format=dict(d["inner"]), unreadable=d["errors"],
            other_files=dict(d["other_files"]), n_distinct_column_sets=len(d["column_sets"]),
            main_columns=cs[0][0].split("|") if cs else [], main_columns_files=cs[0][1] if cs else 0,
            eeg_channels_found=[c for c in (cs[0][0].split("|") if cs else []) if c.strip('"').upper() in EMOTIV14],
            rows=C.dist(d["rows"]), fs_from_timestamp=C.dist(d["fs"]),
            duration_s=C.dist([r / fs_use for r in d["rows"]]), duration_basis=("timestamps" if fs_med else "assumed 128 Hz"),
            amplitude_sample=dict(n=len(amps), median_channel_sd=C.dist([a["median_sd"] for a in amps]),
                                  files_with_flat_channel=sum(1 for a in amps if a["n_flat"] > 0),
                                  max_abs=C.dist([a["max_abs"] for a in amps])),
        )
    metas = []
    for p in files:
        if p.suffix.lower() == ".csv" and "metadata" in p.name.lower():
            try:
                m = summarize_meta(p)
                m["dataset"] = which(p.name)
                metas.append(m)
            except Exception as e:
                metas.append(dict(file=p.name, error=str(e)[:100]))
    C.write_json(P["out"] / "s6_external.json", dict(found=True, datasets=out, metadata=metas,
                                                     source="all .gz files (structure, length); amplitude on a sample",
                                                     seconds=round(time.time() - t0, 1)))
    log(f"STEP 6 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    ap.add_argument("--amp-sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    main(ap.parse_args())
