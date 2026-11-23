"""Step 3: seizure labels from the TUH .csv (per channel) and .csv_bi (per recording) files.
Reads ALL sidecar files of the labelled datasets (TUSZ train/dev/eval, TUEP both groups).

Aggregate output: outputs/s3_labels.json. Private cache: cache/seizure_events.csv (per event, has paths).
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import common as C

BCKG = {"bckg", "background", "null", "none"}


_MEMO = {}


def read_tuh_csv(p):
    if p in _MEMO:
        return _MEMO[p]
    _MEMO[p] = out = _read_tuh_csv(p)
    return out


def _read_tuh_csv(p):
    meta, rows, header = {}, [], None
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re.match(r"#\s*([\w ]+?)\s*=\s*(.*)", line)
                if m:
                    meta[m.group(1).strip().lower()] = m.group(2).strip()
                continue
            parts = [x.strip() for x in line.split(",")]
            if header is None and parts[0].lower() == "channel":
                header = [x.lower() for x in parts]
                continue
            if header is None:
                header = ["channel", "start_time", "stop_time", "label", "confidence"]
            d = dict(zip(header, parts))
            d["start_time"] = float(d["start_time"])
            d["stop_time"] = float(d["stop_time"])
            d["label"] = d.get("label", "").lower()
            rows.append(d)
    dur = None
    if "duration" in meta:
        m = re.match(r"([\d.]+)", meta["duration"])
        dur = float(m.group(1)) if m else None
    return meta, rows, dur


def merge_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def analyse(ds, rows, hdr, t0):
    ev_cache = []
    lab_rows, lab_files, bi_rows, bi_files = Counter(), Counter(), Counter(), Counter()
    csv_channels = Counter()
    conf = Counter()
    errors = Counter()
    dur_mismatch = 0
    events = []  # (pid, ses, path, start, stop, type, n_types)
    rec_seiz = {}
    rec_dur = {}
    derived_from_csv = 0
    for i, r in enumerate(rows, 1):
        C.progress(i, len(rows), t0, 2000, f"{ds} sidecars")
        edf = Path(r["path"])
        stem = edf.with_suffix("")
        ch_rows, bi = [], None
        if r["has_csv"] == "1":
            try:
                _, ch_rows, _ = read_tuh_csv(str(stem) + ".csv")
                lab_rows.update(x["label"] for x in ch_rows)
                lab_files.update({x["label"] for x in ch_rows})
                csv_channels.update({x["channel"] for x in ch_rows})
            except Exception as e:
                errors[f"csv {type(e).__name__}"] += 1
        if r["has_csv_bi"] == "1":
            try:
                _, bi, bdur = read_tuh_csv(str(stem) + ".csv_bi")
                bi_rows.update(x["label"] for x in bi)
                bi_files.update({x["label"] for x in bi})
                conf.update(x.get("confidence", "") for x in bi if x["label"] not in BCKG)
                hd = hdr.get(r["path"])
                if bdur is not None and hd is not None and abs(bdur - hd) > 1.0:
                    dur_mismatch += 1
            except Exception as e:
                errors[f"csv_bi {type(e).__name__}"] += 1
                bi = None
        rec_dur[r["path"]] = hdr.get(r["path"])
        if bi is not None:
            ev = [(x["start_time"], x["stop_time"], x["label"]) for x in bi if x["label"] not in BCKG]
        elif ch_rows:
            derived_from_csv += 1
            ev = [(s, e, "seiz") for s, e in merge_intervals([(x["start_time"], x["stop_time"]) for x in ch_rows if x["label"] not in BCKG])]
        else:
            ev = []
        ch_seiz = [x for x in ch_rows if x["label"] not in BCKG]
        for s, e, binlab in ev:
            ov = Counter()
            for x in ch_seiz:
                o = min(e, x["stop_time"]) - max(s, x["start_time"])
                if o > 0:
                    ov[x["label"]] += o
            typ = ov.most_common(1)[0][0] if ov else binlab
            events.append((r["pid"], r["ses"], r["path"], s, e, typ, len(ov)))
            ev_cache.append(dict(dataset=ds, path=r["path"], start=s, stop=e, type=typ, n_types_overlapping=len(ov)))
        rec_seiz[r["path"]] = sum(e - s for s, e, _ in ev)
    durs = [e - s for _, _, _, s, e, _, _ in events]
    seiz_recs = {p for _, _, p, *_ in events}
    seiz_ses = {(a, b) for a, b, *_ in events}
    seiz_pat = {a for a, *_ in events}
    ev_per_rec = Counter(p for _, _, p, *_ in events)
    ev_per_pat = Counter(a for a, *_ in events)
    type_ev = Counter(t for *_, t, _ in events)
    type_rec = defaultdict(set)
    type_pat = defaultdict(set)
    for pid, ses, p, s, e, t, _ in events:
        type_rec[t].add(p)
        type_pat[t].add(pid)
    total_rec_h = sum(d for d in rec_dur.values() if d) / 3600
    out = dict(
        labels_read=True, n_edf=len(rows), n_patients=len({r["pid"] for r in rows}),
        n_sessions=len({(r["pid"], r["ses"]) for r in rows}),
        n_with_csv=sum(int(r["has_csv"]) for r in rows), n_with_csv_bi=sum(int(r["has_csv_bi"]) for r in rows),
        parse_errors=dict(errors), csv_bi_vs_header_duration_mismatch_gt1s=dur_mismatch,
        events_derived_from_csv_only=derived_from_csv,
        csv_label_rows=dict(lab_rows.most_common()), csv_label_files=dict(lab_files.most_common()),
        csv_bi_label_rows=dict(bi_rows.most_common()), csv_bi_label_files=dict(bi_files.most_common()),
        csv_channel_names_top=C.top(csv_channels, 30), n_distinct_csv_channels=len(csv_channels),
        seizure_confidence_values=C.top(conf, 8),
        n_seizure_events=len(events), seizure_duration_s=C.dist(durs),
        n_seizures_under_10s=sum(1 for d in durs if d < 10), n_seizures_over_5min=sum(1 for d in durs if d > 300),
        seizure_hours=round(sum(durs) / 3600, 2), recorded_hours=round(total_rec_h, 1),
        seizure_pct_of_recorded_time=round(100 * sum(durs) / 3600 / total_rec_h, 2) if total_rec_h else None,
        recordings_with_seizure=len(seiz_recs), sessions_with_seizure=len(seiz_ses), patients_with_seizure=len(seiz_pat),
        events_per_seizure_recording=C.dist(list(ev_per_rec.values())),
        events_per_seizure_patient=C.dist(list(ev_per_pat.values())),
        events_with_several_channel_types=sum(1 for *_, n in events if n > 1),
        seizure_types=[dict(type=t, events=c, recordings=len(type_rec[t]), patients=len(type_pat[t]),
                            median_duration_s=C.dist([e - s for _, _, _, s, e, tt, _ in events if tt == t]).get("median"))
                       for t, c in type_ev.most_common()],
    )
    return out, ev_cache


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 3 labels")
    man = C.read_csv(P["cache"] / "manifest_edf.csv")
    hdr = {}
    hp = P["cache"] / "edf_headers.csv"
    if hp.exists():
        for r in C.read_csv(hp):
            if r["ok"] == "1":
                hdr[r["path"]] = float(r["duration_s"])
    res = {}
    events_cache = []
    t0 = time.time()
    for ds in list(C.DS) + ["TUSZ_all"]:
        if ds == "TUSZ_all":
            rows = [r for r in man if r["dataset"] in ("TUSZ_train", "TUSZ_dev", "TUSZ_eval")]
        else:
            rows = [r for r in man if r["dataset"] == ds]
        if not rows:
            continue
        if ds not in C.LABELLED and ds != "TUSZ_all":
            res[ds] = dict(n_edf=len(rows), n_with_csv=sum(int(r["has_csv"]) for r in rows),
                           n_with_csv_bi=sum(int(r["has_csv_bi"]) for r in rows), labels_read=False)
            continue
        res[ds], ev = analyse(ds, rows, hdr, time.time())
        if ds != "TUSZ_all":
            events_cache += ev
        log(f"  {ds}: {res[ds]['n_seizure_events']} seizure events in {res[ds]['recordings_with_seizure']} recordings / {res[ds]['patients_with_seizure']} patients")
    # pooled TUSZ
    C.write_csv(P["cache"] / "seizure_events.csv", events_cache, ["dataset", "path", "start", "stop", "type", "n_types_overlapping"])
    C.write_json(P["out"] / "s3_labels.json", dict(by_dataset=res, source="all .csv/.csv_bi files of the labelled datasets",
                                                   seconds=round(time.time() - t0, 1)))
    log(f"STEP 3 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    main(ap.parse_args())
