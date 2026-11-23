"""Step 2: read the header of EVERY EDF in the manifest (only the first 256 + n*256 bytes).
Channels, sampling rates, durations, reference suffixes, header-vs-file-size check, sex/age in the
EDF patient field, EDF+ annotation channels. Uses ALL files.

Private output: cache/edf_headers.csv (one row per file, resumable).
Aggregate output: outputs/s2_edf_headers.json
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import common as C

FIELDS = ["dataset", "path", "pid", "ses", "montage", "ses_year", "size", "ok", "error", "fmt", "n_signals",
          "n_records", "record_dur", "duration_s", "fs_eeg", "fs_all", "mixed_fs_eeg", "labels", "ref_suffixes",
          "size_diff", "nrec_from_size", "header_year", "sex", "age", "dims_eeg", "has_annot", "n_annot", "annot_texts",
          "prefilter"]


def read_annotations(path, h, max_records=20000):
    """Parse EDF+ TALs from the 'EDF Annotations' signal. Returns list of texts."""
    labels = h["labels"]
    idx = [i for i, l in enumerate(labels) if l.strip().upper() == "EDF ANNOTATIONS"]
    if not idx:
        return []
    bps = 3 if h["fmt"] == "BDF" else 2
    offs, o = [], 0
    for i, n in enumerate(h["nsamp"]):
        offs.append(o)
        o += n * bps
    rec_bytes = o
    start = 256 + 256 * h["n_signals"]
    texts = []
    nrec = h["n_records"] if h["n_records"] > 0 else int(h["n_records_from_size"])
    with open(path, "rb") as f:
        f.seek(start)
        for _ in range(min(nrec, max_records)):
            rec = f.read(rec_bytes)
            if len(rec) < rec_bytes:
                break
            for i in idx:
                chunk = rec[offs[i]: offs[i] + h["nsamp"][i] * bps].decode("latin-1", "replace")
                for tal in chunk.split("\x00"):
                    if "\x14" not in tal:
                        continue
                    parts = tal.split("\x14")
                    for t in parts[1:]:
                        t = t.strip()
                        if t:
                            texts.append(t)
    return texts


def one(row):
    out = {k: row.get(k, "") for k in ("dataset", "path", "pid", "ses", "montage", "ses_year", "size")}
    try:
        h = C.parse_edf_header(row["path"])
    except Exception as e:  # unreadable header
        out.update(ok=0, error=f"{type(e).__name__}: {str(e)[:80]}")
        return out
    names = [C.norm_label(l) for l in h["labels"]]
    fs = [n / h["record_dur"] if h["record_dur"] > 0 else 0 for n in h["nsamp"]]
    eeg_idx = [i for i, n in enumerate(names) if C.channel_category(n) == "standard 10-20"]
    fs_eeg = sorted({round(fs[i], 3) for i in eeg_idx})
    nrec = h["n_records"] if h["n_records"] > 0 else h["n_records_from_size"]
    pf = C.parse_patient_field(h["patient"])
    annot = []
    if "EDF ANNOTATIONS" in names:
        try:
            annot = read_annotations(row["path"], h)
        except Exception:
            annot = ["<unreadable annotation channel>"]
    out.update(
        ok=1, error="", fmt=h["fmt"], n_signals=h["n_signals"], n_records=h["n_records"], record_dur=h["record_dur"],
        duration_s=round(nrec * h["record_dur"], 3),
        fs_eeg=fs_eeg[0] if len(fs_eeg) == 1 else ("|".join(map(str, fs_eeg)) if fs_eeg else ""),
        fs_all="|".join(map(str, sorted({round(x, 3) for x in fs}))), mixed_fs_eeg=int(len(fs_eeg) > 1),
        labels="|".join(names), ref_suffixes="|".join(sorted({C.ref_suffix(h["labels"][i]) for i in eeg_idx})) or "no EEG",
        size_diff=h["size_diff"] if h["size_diff"] is not None else "", nrec_from_size=round(h["n_records_from_size"], 3),
        header_year=C.header_year(h["startdate"]) or "", sex=pf["sex"] or "", age=pf["age"] if pf["age"] is not None else "",
        dims_eeg="|".join(sorted({h["dims"][i] for i in eeg_idx})), has_annot=int("EDF ANNOTATIONS" in names),
        n_annot=len(annot), annot_texts="\x1f".join(annot[:500]),
        prefilter=h["prefilter"][eeg_idx[0]] if eeg_idx else "")
    return out


def num(x, f=float):
    try:
        return f(x)
    except (TypeError, ValueError):
        return None


def aggregate(rows, manifest_by_path):
    ok = [r for r in rows if str(r["ok"]) == "1"]
    res = dict(n_files=len(rows), n_header_ok=len(ok), n_header_failed=len(rows) - len(ok),
               fail_reasons=C.top(Counter(re.sub(r"\d+", "#", r["error"]) for r in rows if str(r["ok"]) != "1"), 10))
    if not ok:
        return res
    n = len(ok)
    res["format"] = dict(Counter(r["fmt"] for r in ok))
    nsig = [int(r["n_signals"]) for r in ok]
    res["n_signals"] = C.dist(nsig)
    res["n_signals_top"] = C.top(Counter(nsig), 12)
    lab = Counter()
    cat = Counter()
    std_all = 0
    per_elec = Counter()
    sigs_std, sigs_full = Counter(), Counter()
    for r in ok:
        names = set(r["labels"].split("|"))
        lab.update(names)
        cat.update({C.channel_category(x) for x in names})
        pres = C.std19_present(names)
        std_all += all(pres)
        for nm, p in zip(C.STD19_NAMES, pres):
            per_elec[nm] += p
        sigs_full[tuple(sorted(names))] += 1
        sigs_std[tuple(sorted(x for x in names if C.channel_category(x) == "standard 10-20"))] += 1
    res["channel_labels_pct_files"] = [[k, c, round(100 * c / n, 1)] for k, c in lab.most_common(80)]
    res["n_distinct_channel_labels"] = len(lab)
    res["channel_category_pct_files"] = {k: round(100 * c / n, 1) for k, c in cat.most_common()}
    res["all_19_standard_pct"] = round(100 * std_all / n, 2)
    res["all_19_standard_n"] = std_all
    res["per_electrode_pct"] = {k: round(100 * per_elec[k] / n, 2) for k in C.STD19_NAMES}
    res["n_distinct_channel_sets"] = len(sigs_full)
    res["n_distinct_standard_sets"] = len(sigs_std)
    res["ref_suffix_files"] = dict(Counter(r["ref_suffixes"] for r in ok))
    res["montage_x_ref"] = {f"{m} | {s}": c for (m, s), c in Counter((r["montage"], r["ref_suffixes"]) for r in ok).most_common(20)}
    res["fs_eeg"] = C.top(Counter(str(r["fs_eeg"]) for r in ok), 15)
    res["files_mixed_fs_eeg"] = sum(int(r["mixed_fs_eeg"]) for r in ok)
    res["files_mixed_fs_any_channel"] = sum(1 for r in ok if "|" in r["fs_all"])
    dur = [num(r["duration_s"]) for r in ok]
    res["duration_s"] = C.dist(dur)
    res["total_hours"] = round(sum(d for d in dur if d) / 3600, 1)
    res["n_under_60s"] = sum(1 for d in dur if d is not None and d < 60)
    res["n_under_10s"] = sum(1 for d in dur if d is not None and d < 10)
    res["n_over_1h"] = sum(1 for d in dur if d is not None and d > 3600)
    res["n_over_6h"] = sum(1 for d in dur if d is not None and d > 6 * 3600)
    res["record_dur_values"] = C.top(Counter(str(r["record_dur"]) for r in ok), 6)
    sd = [num(r["size_diff"], int) for r in ok]
    res["size_mismatch_files"] = sum(1 for x in sd if x not in (None, 0))
    res["size_truncated_files"] = sum(1 for x in sd if x is not None and x < 0)
    res["size_extra_bytes_files"] = sum(1 for x in sd if x is not None and x > 0)
    res["n_records_unknown"] = sum(1 for r in ok if str(r["n_records"]) in ("-1", ""))
    res["partial_record_files"] = sum(1 for r in ok if abs(num(r["nrec_from_size"]) - round(num(r["nrec_from_size"]))) > 1e-6)
    hy = [num(r["header_year"], int) for r in ok]
    res["header_year"] = C.dist(hy)
    res["header_year_top"] = C.top(Counter(x for x in hy if x), 40)
    agree = [(num(r["header_year"], int), num(r["ses_year"], int)) for r in ok if r["header_year"] and r["ses_year"]]
    res["header_vs_session_year"] = dict(n_both=len(agree), n_equal=sum(1 for a, b in agree if a == b))
    res["dims_eeg"] = dict(Counter(r["dims_eeg"] for r in ok))
    res["prefilter_top"] = C.top(Counter(r["prefilter"] for r in ok), 8)
    res["files_with_annotation_channel"] = sum(int(r["has_annot"]) for r in ok)
    res["n_annotations"] = sum(int(r["n_annot"] or 0) for r in ok)
    texts_files = Counter()
    for r in ok:
        if r["annot_texts"]:
            texts_files.update({t.strip().lower()[:40] for t in r["annot_texts"].split("\x1f")})
    # privacy: only annotation texts that occur in >= 10 files (generic event names, not free text)
    res["annotation_texts_in_10plus_files"] = [[k, c] for k, c in texts_files.most_common(40) if c >= 10]
    res["annotation_texts_rare_distinct"] = sum(1 for c in texts_files.values() if c < 10)
    # patient-level sex / age (age at the patient's earliest session)
    byp = defaultdict(list)
    for r in ok:
        byp[r["pid"]].append(r)
    sexes, ages, sex_missing, age_missing, sex_inconsistent = Counter(), [], 0, 0, 0
    for pid, rs in byp.items():
        s = {r["sex"] for r in rs if r["sex"] in ("M", "F")}
        if len(s) == 1:
            sexes[s.pop()] += 1
        elif len(s) > 1:
            sex_inconsistent += 1
        else:
            sex_missing += 1
        rs2 = sorted(rs, key=lambda r: (r["ses"], r["path"]))
        a = next((num(r["age"], int) for r in rs2 if r["age"] != ""), None)
        if a is None:
            age_missing += 1
        else:
            ages.append(a)
    res["patients"] = len(byp)
    res["sex_patients"] = dict(sexes)
    res["sex_missing_patients"] = sex_missing
    res["sex_inconsistent_patients"] = sex_inconsistent
    res["age_first_session"] = C.dist(ages)
    res["age_missing_patients"] = age_missing
    res["age_bands"] = dict(Counter(("<18" if a < 18 else "18-39" if a < 40 else "40-64" if a < 65 else "65+") for a in ages))
    return res


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 2 EDF headers")
    man = C.read_csv(P["cache"] / "manifest_edf.csv")
    cache_p = P["cache"] / "edf_headers.csv"
    done = {}
    if cache_p.exists() and not args.fresh:
        for r in C.read_csv(cache_p):
            done[(r["path"], r["size"])] = r
    todo = [r for r in man if (r["path"], r["size"]) not in done]
    log(f"  {len(man)} EDF in manifest, {len(done)} cached, {len(todo)} to read")
    t0 = time.time()
    new = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            new.append(r)
            C.progress(i, len(todo), t0, 1000, "headers")
    keep = {(r["path"], r["size"]) for r in man}
    rows = [r for k, r in done.items() if k in keep] + new
    C.write_csv(cache_p, rows, FIELDS)
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    mp = {r["path"]: r for r in man}
    res = {k: aggregate(v, mp) for k, v in by_ds.items()}
    res["TUSZ_all"] = aggregate([r for r in rows if r["dataset"] in ("TUSZ_train", "TUSZ_dev", "TUSZ_eval")], mp)
    res["ALL_LABELLED"] = aggregate([r for r in rows if r["dataset"] in C.LABELLED], mp)
    res["ALL_MAIN"] = aggregate([r for r in rows if r["dataset"] in C.MAIN], mp)
    C.write_json(P["out"] / "s2_edf_headers.json", dict(by_dataset=res, seconds_reading=round(time.time() - t0, 1),
                                                        source="all EDF files (headers only)"))
    log(f"STEP 2 done: {len(rows)} headers ({sum(1 for r in rows if str(r['ok']) != '1')} failed) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    main(ap.parse_args())
