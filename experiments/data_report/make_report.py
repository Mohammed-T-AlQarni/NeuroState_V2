"""Build EEG_Data_Report.docx and EEG_Data_Report.pdf from the aggregate outputs of steps 1-7.
Only aggregate numbers go into the report (counts, percentages, ranges); no IDs, no free text.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import common as C

NAMES = {
    "TUSZ_train": "TUSZ train", "TUSZ_dev": "TUSZ dev", "TUSZ_eval": "TUSZ eval", "TUSZ_all": "TUSZ (all splits)",
    "TUEP_epilepsy": "TUEP epilepsy", "TUEP_no_epilepsy": "TUEP no epilepsy",
    "TUEG_epilepsy_Y": "TUEG Epilepsy=Y set", "TUEG_partial": "TUEG partial",
    "TUSZ_incomplete_copy": "TUSZ incomplete copy",
}
REC_COLS = ["TUSZ_all", "TUEP_epilepsy", "TUEP_no_epilepsy", "TUEG_epilepsy_Y", "TUEG_partial"]


# ---------------------------------------------------------------- formatting helpers
def n_(x):
    if x is None or x == "":
        return "n/a"
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.1f}"
    return f"{int(x):,}"


def pct(a, b, d=1):
    if not b:
        return "n/a"
    return f"{100 * a / b:.{d}f}%"


def gb(b):
    return "n/a" if b is None else (f"{b / 1e9:,.1f} GB" if b >= 1e8 else f"{b / 1e6:,.1f} MB")


def dur(s):
    if s is None:
        return "n/a"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def rng(d, f=dur):
    if not d or not d.get("n"):
        return "n/a"
    return f"{f(d['min'])} / {f(d['median'])} / {f(d['max'])}"


def num1(x):
    return "n/a" if x is None else (f"{x:.0f}" if abs(x) >= 100 or float(x).is_integer() else f"{x:.1f}")


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def clean_num(a):
    a = str(a)
    return re.sub(r"\.0(?=\D|$)", "", a)


def top_share(pairs, total, k=3, suffix=""):
    if not pairs or not total:
        return "n/a"
    return ", ".join(f"{clean_num(a)}{suffix} ({100 * c / total:.0f}%)" for a, c in pairs[:k])


# ---------------------------------------------------------------- content
class Doc:
    def __init__(self):
        self.blocks = []

    def h(self, t, level=1):
        self.blocks.append(("h", level, t))

    def p(self, t, style="body"):
        self.blocks.append(("p", style, t))

    def note(self, t):
        self.blocks.append(("p", "note", t))

    def bullets(self, items, numbered=False):
        self.blocks.append(("list", numbered, [i for i in items if i]))

    def table(self, header, rows, widths=None, note=None):
        self.blocks.append(("table", header, rows, widths))
        if note:
            self.note(note)


def build(J, P, sample_n):
    s1, s2, s3, s4, s5, s6, s7 = (J.get(k, {}) for k in ("s1", "s2", "s3", "s4", "s5", "s6", "s7"))
    H = g(s2, "by_dataset", default={})
    L = g(s3, "by_dataset", default={})
    I = g(s1, "datasets", default={})
    Q = g(s5, "by_dataset", default={})
    D = Doc()
    today = dt.date.today().isoformat()
    n_tuep = (g(L, "TUEP_epilepsy", "n_patients", default=0) or 0) + (g(L, "TUEP_no_epilepsy", "n_patients", default=0) or 0)

    # ---------- shared numbers
    main = H.get("ALL_MAIN", {})
    tusz = L.get("TUSZ_all", {})
    te, tn = L.get("TUEP_epilepsy", {}), L.get("TUEP_no_epilepsy", {})
    he, hn = H.get("TUEP_epilepsy", {}), H.get("TUEP_no_epilepsy", {})
    ratio = (te.get("n_edf", 0) / tn["n_edf"]) if tn.get("n_edf") else None
    uniq_main = g(s7, "unique_patients", "main")
    q_all = Q.get("ALL", {})

    D.blocks.append(("title", "EEG data on the NeuroState drive", f"Overview for AI model development · prepared {today} · data read on {g(J, 'meta', 'run_date', default=today)}"))
    D.note("How to read the numbers: every figure marked [all] was computed from every file of that dataset. "
           f"Figures marked [sample] come from a stratified random sample of {n_(sample_n)} recordings (seed {g(s5, 'seed', default='n/a')}). "
           "The report holds only totals, percentages and ranges: no patient IDs, no per-patient values, no clinical free text. "
           "Scripts and raw totals: Claude_Workspace\\reports\\scripts.")

    # ---------- 1 summary
    D.h("1. Summary")
    seiz_pat_tuep = te.get("patients_with_seizure")
    D.bullets([
        f"The drive holds {n_(main.get('n_files'))} EEG recordings in EDF format ({n_(main.get('total_hours'))} hours, "
        f"{n_(uniq_main)} distinct patients) from three TUH corpora (TUSZ, TUEP, TUEG), plus two small external datasets (Guinea-Bissau, Nigeria) in a different format.",
        f"Seizure labels: TUSZ has {n_(tusz.get('n_edf'))} recordings from {n_(tusz.get('n_patients'))} patients in official patient-separated train/dev/eval splits, "
        f"with {n_(tusz.get('n_seizure_events'))} seizure events timed to the second in {n_(tusz.get('recordings_with_seizure'))} recordings ({n_(tusz.get('patients_with_seizure'))} patients).",
        f"Diagnosis labels: TUEP gives one label per patient: {n_(te.get('n_patients'))} epilepsy and {n_(tn.get('n_patients'))} no-epilepsy patients, "
        f"but {n_(te.get('n_edf'))} vs {n_(tn.get('n_edf'))} recordings ({ratio:.1f} to 1)" if ratio else "TUEP: see section 4.",
        f"Format is consistent: {pct(g(main, 'format', 'EDF', default=0), main.get('n_header_ok'), 0)} plain EDF; the 19 standard 10-20 electrodes are present in "
        f"{n_(main.get('all_19_standard_pct'))}% of files; sampling rates and reference montages vary and must be harmonised.",
        "Suggested plan: treat seizure detection (TUSZ) and epilepsy diagnosis (TUEP) as two tasks; pre-train on TUSZ, fine-tune on TUEP; split strictly by patient, "
        "remove seizure stretches from TUEP inputs, and check age, medication, montage and era differences between groups.",
    ], numbered=True)

    # ---------- 2 files and format
    D.h("2. Files and format")
    D.p("Each dataset below was scanned file by file [all]. 'Patients' and 'sessions' come from the TUH folder and file names "
        "(patient / session / montage / recording).")
    rows = []
    for k in list(C.DS):
        d = I.get(k)
        if not d or not d.get("exists"):
            rows.append([NAMES[k], C.DS[k]["role"], "folder not found", "", "", "", "", ""])
            continue
        other = ", ".join(f"{e['ext']} {n_(e['n'])}" for e in d["ext"] if e["ext"] not in (".edf", ".csv", ".csv_bi"))[:80]
        rows.append([NAMES[k], C.DS[k]["role"], n_(d["n_edf"]), gb(d["n_bytes"]), n_(d["n_patients"]), n_(d["n_sessions"]),
                     f"{n_(d['edf_with_csv'])} / {n_(d['edf_with_csv_bi'])}", other or "none"])
    D.table(["Dataset", "Role", "EDF files", "Size", "Patients", "Sessions", "EDFs with .csv / .csv_bi", "Other file types"], rows,
            note="Source: [all]. Role: labelled = has seizure or diagnosis labels; candidates = flagged Epilepsy=Y from EEG-report metadata, not confirmed; "
                 "unlabelled = no usable labels; ignore = incomplete duplicate.")
    # file types across drive
    ov = g(s1, "drive_overview", default=[])
    allext = {}
    for f in ov:
        for e, c in f["top_ext"]:
            allext[e] = allext.get(e, 0) + c
    eeg_types = {".edf": "EDF/EDF+", ".bdf": "BDF", ".set": "EEGLAB .set", ".fdt": "EEGLAB .fdt", ".vhdr": "BrainVision .vhdr",
                 ".eeg": "BrainVision .eeg", ".fif": "MNE .fif", ".cnt": "Neuroscan .cnt", ".gdf": "GDF", ".mat": "MATLAB .mat"}
    present = [f"{v} ({n_(allext[e])})" for e, v in eeg_types.items() if allext.get(e)]
    absent = [v for e, v in eeg_types.items() if not allext.get(e)]
    fmt = main.get("format", {})
    D.p(f"EEG file types found on the drive (top file types per folder): {', '.join(present) or 'none'}. Not found: {', '.join(absent)}. "
        f"Inside the EDF headers [all]: " + ", ".join(f"{k} {n_(v)}" for k, v in fmt.items()) +
        ". 'EDF' means plain EDF (no EDF+ annotation channel); 'EDF+C/D' would mean EDF+ with an annotation channel.")
    # organisation patterns
    pats = []
    for k in ("TUSZ_train", "TUEP_epilepsy", "TUEG_partial"):
        d = I.get(k, {})
        dp = d.get("dir_patterns") or {}
        parts = [v[0][0] for _, v in sorted(dp.items(), key=lambda x: int(x[0])) if v]
        fp = g(d, "file_patterns", ".edf", default=[])
        if parts:
            pats.append(f"{NAMES[k]}: {'/'.join(parts)}/{fp[0][0] if fp else '?'}.edf")
    D.p("Folder organisation (identifier-free patterns; <pid> = 8-letter patient code, s### = session number, YYYY = year, t### = recording token): ")
    D.bullets(pats + ["Each recording has a same-name .csv (seizure events per channel) and .csv_bi (seizure/background intervals for the whole recording) in TUSZ and TUEP; TUEG folders have no annotation files."])
    docs = g(s1, "docs", default=[])
    if docs:
        byf = {}
        for x in docs:
            byf.setdefault(x["folder"], []).append(x["file"])
        D.p("Documentation files found: " + "; ".join(f"{f}: {', '.join(sorted(v)[:14])}{' …' if len(v) > 14 else ''}" for f, v in byf.items()) + ".")
    # top-level folders
    big = sorted(ov, key=lambda x: -x["bytes"])[:14]
    if big:
        D.table(["Folder (two levels under G:\\NeuroState)", "Files", "Size", "Main file types"],
                [[f["folder"], n_(f["n"]), gb(f["bytes"]), ", ".join(f"{e} {n_(c)}" for e, c in f["top_ext"][:4])] for f in big],
                note=f"Source: [all], largest 14 folders; software folders (.venv, .git, site-packages …) skipped ({n_(g(s1, 'drive_skipped_software_dirs'))}).")
    dup = g(s7, "tusz_incomplete_copy", default={})
    man_n = g(s1, "tueg_Y_manifest_edf_lines")
    D.bullets([
        f"TUSZ incomplete copy (TUH_Additional\\TUSZ_v2.0.6\\edf): {n_(dup.get('n_files'))} EDFs, of which {n_(dup.get('present_in_complete_copy'))} have the same name in the complete copy. It is ignored." if dup else None,
        f"TUEG Epilepsy=Y manifest lists {n_(man_n)} EDFs; the folder holds {n_(g(I, 'TUEG_epilepsy_Y', 'n_edf'))}." if man_n is not None else None,
    ])

    # ---------- 3 recording details
    D.h("3. Recording details")
    D.p("Read from the header of every EDF [all] (the header is the first few kilobytes of each file and describes channels, sampling rate and length).")

    def col(k):
        return H.get(k, {})

    def fs_str(d):
        return top_share(d.get("fs_eeg"), d.get("n_header_ok"), 3, " Hz")

    def cat(d, c):
        return f"{g(d, 'channel_category_pct_files', c, default=0):.0f}%" if d.get("n_header_ok") else "n/a"

    def mont(k):
        m = g(I, k, "montage_folders", default=[])
        return top_share(m, g(I, k, "n_edf", default=0), 4)

    def refs(d):
        r = d.get("ref_suffix_files") or {}
        tot = sum(r.values())
        return ", ".join(f"{a} {100 * c / tot:.0f}%" for a, c in sorted(r.items(), key=lambda x: -x[1])[:3]) if tot else "n/a"

    spec = [
        ("EDF files [all]", lambda d, k: n_(d.get("n_files"))),
        ("Header unreadable [all]", lambda d, k: n_(d.get("n_header_failed"))),
        ("Channels per file: min / median / max", lambda d, k: rng(d.get("n_signals"), num1)),
        ("All 19 standard 10-20 electrodes", lambda d, k: f"{n_(d.get('all_19_standard_pct'))}%"),
        ("A1/A2 ear electrodes", lambda d, k: cat(d, "ear/mastoid (A1/A2)")),
        ("Other EEG electrodes (T1/T2, SP1/SP2 …)", lambda d, k: cat(d, "other EEG electrode")),
        ("EKG/ECG", lambda d, k: cat(d, "EKG/ECG")),
        ("EMG", lambda d, k: cat(d, "EMG")),
        ("EOG / eye", lambda d, k: cat(d, "EOG/eye")),
        ("Photic stimulation", lambda d, k: cat(d, "photic")),
        ("Derived (IBI/BURSTS/SUPPR)", lambda d, k: cat(d, "derived (IBI/BURSTS/SUPPR)")),
        ("Reference suffix of EEG channels", lambda d, k: refs(d)),
        ("Montage folders", lambda d, k: mont("TUSZ_train") if k == "TUSZ_all" else mont(k)),
        ("Sampling rate of EEG channels", lambda d, k: fs_str(d)),
        ("Files with mixed EEG sampling rates", lambda d, k: n_(d.get("files_mixed_fs_eeg"))),
        ("Duration: min / median / max", lambda d, k: rng(d.get("duration_s"))),
        ("Total hours", lambda d, k: n_(d.get("total_hours"))),
        ("Shorter than 1 min", lambda d, k: n_(d.get("n_under_60s"))),
        ("Longer than 1 h", lambda d, k: n_(d.get("n_over_1h"))),
        ("Recording year (EDF header): min / median / max", lambda d, k: rng(d.get("header_year"), lambda x: f"{x:.0f}")),
        ("Distinct channel sets", lambda d, k: n_(d.get("n_distinct_channel_sets"))),
        ("Files with EDF+ annotation channel", lambda d, k: n_(d.get("files_with_annotation_channel"))),
    ]
    D.table(["Item"] + [NAMES[k] for k in REC_COLS], [[lab] + [f(col(k), k) for k in REC_COLS] for lab, f in spec],
            note="Source: [all] EDF headers. Montage folders for TUSZ are shown for the train split. Percentages are % of files. "
                 "Reference: -REF = common reference (folders 01_tcp_ar, 03_tcp_ar_a), -LE = linked ears (02_tcp_le).")
    pe = g(H, "ALL_MAIN", "per_electrode_pct", default={})
    low = [f"{k} {v:.1f}%" for k, v in pe.items() if v < 99.5]
    D.p("Standard electrodes [all]: " + ("every one of the 19 (Fp1, Fp2, F7, F3, Fz, F4, F8, T3/T7, C3, Cz, C4, T4/T8, T5/P7, P3, Pz, P4, T6/P8, O1, O2) is present in at least 99.5% of files."
                                        if not low else "electrodes missing from more than 0.5% of files: " + ", ".join(low) + "."))
    labs = g(H, "ALL_MAIN", "channel_labels_pct_files", default=[])
    if labs:
        D.p("Most common channel names after removing the 'EEG ' prefix and the -REF/-LE suffix [all]: " +
            ", ".join(f"{a} ({c:.0f}%)" for a, _, c in labs[:40]) + f". In total {n_(g(H, 'ALL_MAIN', 'n_distinct_channel_labels'))} distinct channel names occur.")
    ann = g(H, "ALL_MAIN", "files_with_annotation_channel", default=0)
    D.p(f"Annotations inside the EDF files [all]: {n_(ann)} files carry an EDF+ annotation channel"
        + (f" ({n_(g(H, 'ALL_MAIN', 'n_annotations'))} annotations; event names found in 10 or more files: "
           + ", ".join(f"{a} ({n_(c)})" for a, c in g(H, 'ALL_MAIN', 'annotation_texts_in_10plus_files', default=[])[:20]) + ")" if ann else "")
        + ". All seizure labels therefore come from the .csv / .csv_bi files (section 4).")

    # ---------- 4 labels
    D.h("4. Labels and group counts")
    D.p("Three label sources exist. (a) TUSZ: seizure start and stop times for each recording (.csv_bi, one row per interval, label 'seiz' or 'bckg') "
        "and per channel with the seizure type (.csv). (b) TUEP: the class folder (00_epilepsy / 01_no_epilepsy) is the published label for the patient, "
        "and TUEP also ships .csv/.csv_bi seizure annotations. (c) TUEG Epilepsy=Y: patients whose EEG report metadata mentions epilepsy; candidates only, "
        "with no annotation files and no matched negatives. The rest of TUEG has no usable labels.")
    rows = []
    for k in ("TUSZ_train", "TUSZ_dev", "TUSZ_eval", "TUEP_epilepsy", "TUEP_no_epilepsy", "TUEG_epilepsy_Y", "TUEG_partial"):
        i, l = I.get(k, {}), L.get(k, {})
        level = {"TUSZ": "per recording (seizure intervals)", "TUEP": "per patient (folder) + seizure intervals", "TUEG": "per patient (candidate)" if k == "TUEG_epilepsy_Y" else "none"}[C.DS[k]["family"]]
        rows.append([NAMES[k], level, n_(i.get("n_patients")), n_(i.get("n_sessions")), n_(i.get("n_edf")),
                     rng(i.get("edf_per_patient"), num1), n_(i.get("patients_multi_session")),
                     n_(l.get("n_seizure_events")) if l.get("labels_read") else "-",
                     n_(l.get("recordings_with_seizure")) if l.get("labels_read") else "-",
                     n_(l.get("patients_with_seizure")) if l.get("labels_read") else "-"])
    D.table(["Group", "Label level", "Patients", "Sessions", "Recordings", "Recordings per patient (min/median/max)", "Patients with >1 session",
             "Seizure events", "Recordings with seizure", "Patients with seizure"], rows,
            note="Source: [all] file names and .csv_bi files. A seizure event = one non-background interval in a .csv_bi file.")
    if ratio:
        D.p(f"Class balance in TUEP: patients are balanced ({n_(te.get('n_patients'))} vs {n_(tn.get('n_patients'))}) but recordings are not: "
            f"{n_(te.get('n_edf'))} vs {n_(tn.get('n_edf'))} ({ratio:.1f} to 1), and {n_(he.get('total_hours'))} vs {n_(hn.get('total_hours'))} hours. "
            f"Epilepsy patients with at least one annotated seizure: {n_(seiz_pat_tuep)} of {n_(te.get('n_patients'))} (the TUEP README says 16 of 100). "
            f"No-epilepsy patients with an annotated seizure: {n_(tn.get('patients_with_seizure'))}.")
    # seizure durations
    rows = []
    for k in ("TUSZ_train", "TUSZ_dev", "TUSZ_eval", "TUSZ_all", "TUEP_epilepsy"):
        l = L.get(k, {})
        if not l.get("labels_read") and k != "TUSZ_all":
            continue
        dd = l.get("seizure_duration_s", {})
        rows.append([NAMES[k], n_(l.get("n_seizure_events")), rng(dd), f"{dur(dd.get('p25'))} - {dur(dd.get('p75'))}" if dd.get("n") else "n/a",
                     n_(l.get("n_seizures_under_10s")), n_(l.get("n_seizures_over_5min")), n_(l.get("seizure_hours")),
                     f"{l.get('seizure_pct_of_recorded_time')}%" if l.get("seizure_pct_of_recorded_time") is not None else "n/a",
                     rng(l.get("events_per_seizure_recording"), num1)])
    D.table(["Group", "Seizure events", "Duration min / median / max", "Middle half of durations", "< 10 s", "> 5 min", "Seizure hours",
             "% of recorded time", "Events per seizure recording"], rows, note="Source: [all] .csv_bi files.")
    # seizure types
    cb = g(s4, "seizure_type_codebook") or {}
    cbmap = {}
    for r in cb.get("rows", []):
        if len(r) >= 2 and r[0]:
            cbmap[str(r[0]).strip().lower()] = str(r[1]).strip()
    rows = []
    tep_types = {t["type"]: t for t in L.get("TUEP_epilepsy", {}).get("seizure_types", [])}
    for t in tusz.get("seizure_types", []):
        e = tep_types.get(t["type"], {})
        rows.append([t["type"], cbmap.get(t["type"], ""), n_(t["events"]), n_(t["recordings"]), n_(t["patients"]), dur(t["median_duration_s"]),
                     n_(e.get("events", 0)), n_(e.get("patients", 0))])
    for t, e in tep_types.items():
        if t not in {r[0] for r in rows}:
            rows.append([t, cbmap.get(t, ""), "0", "0", "0", "", n_(e["events"]), n_(e["patients"])])
    D.table(["Type", "Meaning (seizures_types_v02.xlsx)", "TUSZ events", "TUSZ recordings", "TUSZ patients", "TUSZ median duration", "TUEP events", "TUEP patients"], rows,
            note="Source: [all] .csv files. Each .csv_bi event is given the type that covers it longest across channels in the .csv file; "
                 f"{n_(tusz.get('events_with_several_channel_types'))} TUSZ events had more than one type across channels.")
    labrows = []
    for k in ("TUSZ_all", "TUEP_epilepsy", "TUEP_no_epilepsy"):
        l = L.get(k, {})
        labrows.append([NAMES[k], ", ".join(f"{a} {n_(c)}" for a, c in (l.get("csv_label_files") or {}).items()),
                        ", ".join(f"{a} {n_(c)}" for a, c in (l.get("csv_bi_label_files") or {}).items())])
    D.table(["Group", "Label names in .csv (number of files containing each)", "Label names in .csv_bi (files)"], labrows, note="Source: [all].")
    # overlaps
    ch = g(s7, "checks_vs_brief", default=[])
    cf = g(s7, "conflicts", default={})
    if ch:
        D.p("Patients who appear in more than one dataset [all] (same 8-letter TUH patient code). These must sit on the same side of every train/test split:")
        D.table(["Dataset A", "Dataset B", "Shared patients (found)", "Expected from earlier check"],
                [[a["a"].replace("TUEP_epilepsy|TUEP_no_epilepsy", "TUEP (both groups)"), NAMES.get(a["b"], a["b"]), n_(a["found"]), n_(a["expected"])] for a in ch])
        sp = cf.get("tusz_split_overlap", {})
        D.bullets([
            f"TUSZ official splits share {n_(sum(sp.values()))} patients (train-dev {n_(sp.get('train_dev'))}, train-eval {n_(sp.get('train_eval'))}, dev-eval {n_(sp.get('dev_eval'))}).",
            f"TUEP patients in both class folders: {n_(cf.get('tuep_epi_and_no_epi'))}. TUEP no-epilepsy patients also in TUSZ: {n_(cf.get('tuep_no_epi_in_tusz'))}, "
            f"of whom {n_(cf.get('tuep_no_epi_with_tusz_seizure'))} have a seizure annotated in TUSZ (a label conflict if above 0).",
            f"TUEG Epilepsy=Y patients also in TUEP: {n_(cf.get('tueg_Y_in_tuep'))}. TUEG partial patients also in TUSZ or TUEP: {n_(cf.get('tueg_partial_in_labelled'))}; "
            f"TUEG partial and TUEG Epilepsy=Y share {n_(cf.get('tueg_partial_and_tueg_Y'))}.",
        ])
        cl = g(s7, "sessions_common_list")
        if cl:
            D.p(f"TUEP's {cl['file']} lists {n_(cl['n_sessions_parsed'])} sessions; {n_(cl['found_in_tuep'])} are on the drive in TUEP and {n_(cl['found_in_tusz'])} in TUSZ. "
                f"Counting by file names, TUEP and TUSZ share {n_(cl['computed_tuep_tusz_session_overlap'])} sessions.")
    # metadata: what 'no epilepsy' means
    D.h("What the metadata says about the groups", 2)
    D.p("The NEDC annotation guide (v04) defines the metadata Epilepsy column as 'Y if the EEG report indicated the condition, otherwise N'. "
        "N therefore means 'not mentioned in the report', not 'confirmed free of epilepsy'.")
    sheets, _sig = [], set()
    for s_ in g(s4, "sheets", default=[]):
        if s_.get("by_group"):
            sig = (tuple(c["name"] for c in s_["columns"]), s_["n_rows"])
            if sig not in _sig:
                _sig.add(sig)
                sheets.append(s_)
    tuep_sheets = sorted([s for s in sheets if "TUEP_no_epilepsy" in s["by_group"]],
                         key=lambda s: (-s["by_group"]["TUEP_no_epilepsy"]["patients"], 0 if s["file"].lower().endswith("r.xlsx") else 1))
    if tuep_sheets:
        s = tuep_sheets[0]
        ge, gn = s["by_group"].get("TUEP_epilepsy", {}), s["by_group"]["TUEP_no_epilepsy"]
        kw = gn.get("keyword_patients", {})
        rows = [[k, n_(g(ge, "keyword_patients", k, default=0)), n_(v)] for k, v in kw.items() if v or g(ge, "keyword_patients", k, default=0)]
        D.table(["Keyword group (searched in the free-text columns)", f"TUEP epilepsy patients (of {n_(ge.get('patients'))} matched)", f"TUEP no-epilepsy patients (of {n_(gn.get('patients'))} matched)"],
                rows, note=f"Source: [all] rows of {s['file']} (sheet {s['sheet']}); text columns searched: {', '.join(gn.get('keyword_text_columns', [])) or 'none'}. "
                           "Counts are patients whose text mentions the keyword at least once; one patient can count in several rows. Keyword lists are in s4_metadata.py.")
    else:
        D.p("No TUEP metadata sheet could be linked to TUEP patients by ID, so the no-epilepsy group could not be described from metadata.")
    tsz = [s for s in sheets if s["file"].lower().startswith("metadata_v06")]
    if tsz:
        s = tsz[0]
        cols = [c for c in ("Epilepsy", "Electrographic seizure", "Clinical seizure") if any(c in g(v, "categorical_counts", default={}) for v in s["by_group"].values())]
        grs = [k for k in ("TUSZ_train", "TUSZ_dev", "TUSZ_eval", "TUEP_epilepsy", "TUEP_no_epilepsy", "TUEG_epilepsy_Y") if k in s["by_group"]]
        rows = []
        for k in grs:
            v = s["by_group"][k]
            rows.append([NAMES[k], n_(v["rows"])] + [", ".join(f"{a} {n_(b)}" for a, b in g(v, "categorical_counts", c, default={}).items()) or "n/a" for c in cols])
        if cols:
            D.table(["Group", "Rows matched"] + [f"{c} (rows)" for c in cols], rows,
                    note=f"Source: [all] rows of {s['file']}, sheet {s['sheet']} (session level). Values with fewer than 5 rows are grouped as 'other'.")

    # external
    D.h("External datasets (reported separately)", 2)
    ext = g(s6, "datasets", default={})
    if not g(s6, "found"):
        D.p("G:\\NeuroState\\External_Datasets was not found.")
    else:
        rows = []
        for k, d in ext.items():
            if k == "other" and not d.get("n_gz"):
                continue
            metas = [m for m in g(s6, "metadata", default=[]) if m.get("dataset") == k]
            grp = metas[0].get("group_counts") if metas else None
            rows.append([k, n_(d["n_gz"]), gb(d["gz_bytes"]), ", ".join(f"{a} {n_(b)}" for a, b in d["inner_format"].items()),
                         f"{len(d['eeg_channels_found'])}: {' '.join(d['eeg_channels_found'])}",
                         rng(d.get("fs_from_timestamp"), lambda x: f"{x:.0f}") + " Hz" if g(d, "fs_from_timestamp", "n") else "not in file",
                         rng(d.get("duration_s")), ", ".join(f"{a} {n_(b)}" for a, b in (grp or {}).items()) or "n/a", n_(d["unreadable"])])
        D.table(["Dataset", ".gz files", "Size", "Inside", "EEG channels", "Sampling rate (min/median/max)", "Duration min / median / max", "Groups (metadata rows)", "Unreadable"], rows,
                note="Source: [all] .gz files (streamed) and metadata CSVs. Durations use the timestamp column when present.")
        for m in g(s6, "metadata", default=[]):
            if m.get("columns"):
                D.p(f"{m['file']}: {n_(m['n_rows'])} rows; fields: " + ", ".join(f"{c['name']} ({c['pct_missing']:.0f}% missing)" for c in m["columns"]) + ".")
        if ext and all(len(d.get("eeg_channels_found", [])) == 14 for d in ext.values() if d.get("n_gz")):
            D.p("These recordings use a 14-channel consumer headset layout (AF3 … AF4) at a lower sampling rate, with no A1/A2, Fz, Cz, Pz, Fp1/Fp2 or C3/C4. "
                "Only 10 channels can be matched to TUH names (F7, F3, F4, F8, T7=T3, T8=T4, P7=T5, P8=T6, O1, O2). They can serve as an external test of a model built on "
                "those shared channels, not as extra training data for the 19-channel models.")
        else:
            D.p("The channel layout differs from TUH (see the table). Use these data only as an external test on the channels shared with TUH.")

    # ---------- 5 other patient information
    D.h("5. Other patient information available")
    D.p("Age and sex are written in the TUH EDF header ('Age:NN' and M/F in the patient field); recording year is in the session folder name (s###_YYYY) "
        "and the header start date. Values below are per patient (age at the earliest session) [all].")
    rows = []
    for k in ("TUSZ_all", "TUEP_epilepsy", "TUEP_no_epilepsy", "TUEG_epilepsy_Y", "TUEG_partial"):
        d = H.get(k, {})
        npat = d.get("patients") or 0
        a = d.get("age_first_session", {})
        sx = d.get("sex_patients", {})
        yr = d.get("header_year", {})
        i = I.get("TUSZ_train" if k == "TUSZ_all" else k, {})
        rows.append([NAMES[k], n_(npat),
                     f"{num1(a.get('median'))} ({num1(a.get('p25'))}-{num1(a.get('p75'))}); {num1(a.get('min'))}-{num1(a.get('max'))}" if a.get("n") else "n/a",
                     pct(d.get("age_missing_patients", 0), npat, 0), pct(sx.get("F", 0), npat, 0), pct(d.get("sex_missing_patients", 0) + d.get("sex_inconsistent_patients", 0), npat, 0),
                     f"{num1(yr.get('min'))}-{num1(yr.get('max'))} (median {num1(yr.get('median'))})" if yr.get("n") else "n/a",
                     pct(d.get("n_over_1h", 0), d.get("n_header_ok"), 0)])
    D.table(["Group", "Patients", "Age: median (IQR); range", "Age missing", "Female", "Sex missing or inconsistent", "Recording year", "Recordings > 1 h (long-term proxy)"], rows,
            note="Source: [all] EDF headers. 'Sex inconsistent' = different sex in different sessions of the same patient. "
                 "Recordings longer than 1 hour are used as a rough proxy for long-term monitoring; routine EEGs are usually 20-40 minutes.")
    seen = set()
    for s in g(s4, "sheets", default=[]):
        if not s.get("columns"):
            if s.get("error"):
                D.p(f"{s['file']}: could not be read ({s['error']}).")
            continue
        sig = (tuple(c["name"] for c in s["columns"]), s["n_rows"])
        if sig in seen:
            D.p(f"{s['file']} (sheet {s['sheet']}) has the same columns and row count as the sheet above.")
            continue
        seen.add(sig)
        grs = [k for k in ("TUSZ_all", "TUEP_epilepsy", "TUEP_no_epilepsy", "TUEG_epilepsy_Y") if k in (s.get("by_group") or {})]
        rows = []
        for c in s["columns"][:30]:
            extra = ""
            if c.get("dist") and c["dist"].get("n"):
                extra = f"range {num1(c['dist']['min'])}-{num1(c['dist']['max'])}, median {num1(c['dist']['median'])}"
            elif c.get("counts"):
                extra = ", ".join(f"{a} {n_(b)}" for a, b in list(c["counts"].items())[:6])
            elif c.get("n_distinct") is not None:
                extra = f"{n_(c['n_distinct'])} distinct values (not shown)"
            rows.append([c["name"], c["kind"], f"{c['pct_missing']:.0f}%"] +
                        [f"{g(s, 'by_group', k, 'pct_missing', c['name'], default=float('nan')):.0f}%" if g(s, 'by_group', k, 'pct_missing', c['name']) is not None else "-" for k in grs] + [extra])
        D.p(f"{s['file']}, sheet '{s['sheet']}' ({s['folder']}): {n_(s['n_rows'])} rows, {n_(s['n_columns'])} columns, headers on Excel row {s['header_excel_row']}"
            + (f"; linked to {n_(s.get('n_patient_ids'))} patient codes." if s.get("id_column_found") else "; no patient-code column found."))
        D.table(["Field", "Type", "Missing (all rows)"] + [f"Missing: {NAMES[k]}" for k in grs] + ["Summary (aggregate)"], rows)
    meds = []
    for s in sheets:
        for k in ("TUSZ_all", "TUEP_epilepsy", "TUEP_no_epilepsy"):
            v = s["by_group"].get(k, {})
            if "patients_with_any_medication_listed" in v:
                meds.append(f"{s['file']}, {NAMES[k]}: {pct(v['patients_with_any_medication_listed'], v['patients'], 0)} of matched patients have a medication listed; "
                            f"most frequent (rows): {', '.join(f'{a} {n_(b)}' for a, b in v.get('top_medications_rows', [])[:8]) or 'none with 5 or more rows'}.")
    if meds:
        D.p("Medication (group totals only):")
        D.bullets(meds)
    found = {}
    for s in g(s4, "sheets", default=[]):
        for f, cols in (s.get("fields_found") or {}).items():
            if cols:
                found.setdefault(f, set()).update(f"{s['file']}: {c}" for c in cols)
    want = ["age", "sex/gender", "medication", "diagnosis", "comorbidity", "recording type (routine / long-term / ICU / EMU)", "sleep / state", "date / year"]
    D.p("Field check across all metadata sheets: " + "; ".join(f"{f}: {'found (' + ', '.join(sorted(found[f])[:3]) + ')' if f in found else 'not found'}" for f in want) +
        ". Age, sex and year are always available from the EDF headers and folder names, as shown above.")

    # ---------- 6 data quality
    D.h("6. Data quality issues")
    am = H.get("ALL_MAIN", {})
    hdr_fail = {k: H[k].get("n_header_failed", 0) for k in C.MAIN if k in H}
    D.p("Whole-dataset checks [all]:")
    D.bullets([
        f"Unreadable headers: {n_(am.get('n_header_failed'))} of {n_(am.get('n_files'))} files (" + ", ".join(f"{NAMES[k]} {v}" for k, v in hdr_fail.items() if v) + ")."
        + (f" Reasons: {', '.join(f'{a} ({b})' for a, b in am.get('fail_reasons', []))}." if am.get("fail_reasons") else ""),
        f"File size does not match the header: {n_(am.get('size_mismatch_files'))} files ({n_(am.get('size_truncated_files'))} shorter than the header says, i.e. truncated; "
        f"{n_(am.get('size_extra_bytes_files'))} with extra bytes); {n_(am.get('partial_record_files'))} end in a partial data record.",
        f"Very short recordings: {n_(am.get('n_under_60s'))} under 1 minute (of which {n_(am.get('n_under_10s'))} under 10 s).",
        f"Mixed formats: {n_(am.get('n_distinct_channel_sets'))} different channel sets; channel count {rng(am.get('n_signals'), num1)}; sampling rates "
        f"{top_share(am.get('fs_eeg'), am.get('n_header_ok'), 5, ' Hz')}; {n_(am.get('files_mixed_fs_eeg'))} files mix sampling rates across EEG channels.",
        f"Missing standard electrodes: {n_(am.get('n_header_ok', 0) - am.get('all_19_standard_n', 0))} files lack at least one of the 19.",
        "Missing annotation files: " + ", ".join(f"{NAMES[k]} {n_(I[k]['n_edf'] - I[k]['edf_with_both'])}" for k in C.LABELLED if k in I) + " EDFs without both .csv and .csv_bi.",
        "Annotation length differs from the EDF length by more than 1 s: " + ", ".join(f"{NAMES[k]} {n_(L[k].get('csv_bi_vs_header_duration_mismatch_gt1s'))}" for k in C.LABELLED if k in L) + ".",
        "Label file parse errors: " + (", ".join(f"{NAMES[k]} {sum(L[k].get('parse_errors', {}).values())}" for k in C.LABELLED if k in L)) + ".",
        f"Header start year agrees with the session-folder year in {n_(g(am, 'header_vs_session_year', 'n_equal'))} of {n_(g(am, 'header_vs_session_year', 'n_both'))} files.",
        f"Patients with different sex recorded in different sessions: {n_(am.get('sex_inconsistent_patients'))}.",
    ])
    D.p(f"Signal checks on a stratified sample [sample] ({n_(q_all.get('n_sampled'))} files; three 20 s windows per file at 10%, 50% and 90% of its length; 19 standard channels; read with MNE):")
    rows = []
    for k in C.MAIN + ["ALL"]:
        q = Q.get(k)
        if not q:
            continue
        rows.append([NAMES.get(k, "All sampled"), n_(q.get("n_sampled")), n_(q.get("n_failed")), n_(q.get("files_with_flat_channel")),
                     n_(q.get("files_all_channels_flat")), n_(q.get("files_with_high_sd_channel")), n_(q.get("files_with_extreme_amplitude")),
                     n_(q.get("files_duration_mismatch_gt1s")), rng(q.get("median_channel_sd_uv"), num1)])
    th = g(s5, "thresholds", default={})
    D.table(["Dataset", "Sampled", "Could not read", "Files with a flat channel", "All channels flat", "Files with a very noisy channel",
             "Files with extreme amplitude", "Length differs from header > 1 s", "Typical channel SD (µV) min/median/max"], rows,
            note=f"Source: [sample]. Flat = channel SD < {th.get('flat_sd_uv')} µV; very noisy = SD > {th.get('high_sd_uv')} µV; extreme = any sample beyond ±{th.get('extreme_peak_uv')} µV. "
                 "Strata (dataset × montage folder) and their sizes are in outputs/s5_signal_qc.json.")
    if g(s6, "found"):
        D.p("External datasets: " + "; ".join(f"{k}: {n_(d['unreadable'])} unreadable .gz files, {n_(d['n_distinct_column_sets'])} column layout(s), "
                                              f"{n_(g(d, 'amplitude_sample', 'files_with_flat_channel'))} of {n_(g(d, 'amplitude_sample', 'n'))} sampled files with a flat channel [sample]"
                                              for k, d in ext.items() if d.get("n_gz")) + ".")

    # ---------- 7 suggested models and plan
    D.h("7. Suggested models and plan")
    D.p("**Two different questions.** The data supports two tasks that should be kept apart, because they need different labels, inputs and success measures:")
    D.bullets([
        f"**Seizure detection** (is a seizure happening in this 2-10 s stretch of EEG?). TUSZ has the right labels: {n_(tusz.get('n_seizure_events'))} seizure events timed to the second "
        f"in {n_(tusz.get('recordings_with_seizure'))} recordings, with official splits that keep each patient in one split.",
        f"**Epilepsy diagnosis** (does this person have epilepsy, judged from an EEG that usually shows no seizure?). TUEP has the right labels, but for only "
        f"{n_(te.get('n_patients', 0) + tn.get('n_patients', 0))} patients, and only {n_(seiz_pat_tuep)} epilepsy patients have an annotated seizure. "
        f"This is the clinically harder and more interesting question; with {n_(n_tuep)} patients, results will have wide uncertainty.",
    ])
    D.p("Recommendation: do both, report them separately, and use seizure detection as a stepping stone for diagnosis.")
    D.p("**Order of pre-training.** Pre-training works best on the larger and more precisely labelled data. "
        f"TUSZ provides {n_(tusz.get('seizure_hours'))} hours of exactly timed seizures ({n_(tusz.get('n_seizure_events'))} events) plus {n_(tusz.get('recorded_hours'))} hours of labelled background from {n_(tusz.get('n_patients'))} patients; "
        f"TUEP provides one coarse label for each of {n_(n_tuep)} patients. Pre-training on the epilepsy labels and then fine-tuning on seizures would start from the weakest signal. The reverse order fits this data better:")
    D.bullets([
        f"Optional stage 0: self-supervised pre-training (no labels, e.g. masked-signal reconstruction or contrastive learning) on the unlabelled TUEG partial download "
        f"({n_(g(H, 'TUEG_partial', 'n_files'))} recordings, {n_(g(H, 'TUEG_partial', 'total_hours'))} hours). This is the largest pool and teaches the network what normal and abnormal EEG look like.",
        "Stage 1: supervised pre-training on seizure vs background windows from TUSZ train; tune on TUSZ dev; touch TUSZ eval only once, at the end.",
        "Stage 2: fine-tune on TUEP epilepsy vs no epilepsy with patient-level cross-validation (e.g. 5 folds x repeats, nested for tuning). Average the window predictions of each patient into one patient score.",
        "Always compare with the same model trained on TUEP from scratch. Seizure-detector features may not carry over to the subtle interictal signs of epilepsy, and only this comparison shows whether pre-training helps.",
    ], numbered=True)
    D.p("**Fitting the team's existing models.**")
    D.bullets([
        "EEGNet: works for both tasks. Use one fixed channel layout (the 19 standard electrodes, re-referenced to common average, or the TCP bipolar montage that TUH uses, which can be computed from both -REF and -LE recordings), "
        "one sampling rate (resample everything, e.g. to 250 Hz or 128 Hz), and fixed windows (about 2-4 s for seizures, longer windows pooled per patient for diagnosis).",
        "Microstate features + logistic regression / random forest: a good, explainable baseline for TUEP diagnosis because it needs few patients. Compute microstates on clean background EEG only "
        "(seizure and artefact stretches removed), fit the microstate maps inside each training fold only, and use the same channels and reference for every patient.",
        f"Hybrid EEGNet + microstates and the microstate sequence CNN: try them after the two baselines. With {n_(n_tuep)} patients, differences of a few points in accuracy are within noise; "
        "report balanced accuracy and AUROC with patient-level confidence intervals (bootstrap over patients), and repeat with several random seeds.",
    ])
    hr_e, hr_n = he.get("total_hours"), hn.get("total_hours")
    D.p("**Class balance.** " + (f"TUEP has {ratio:.1f} times more epilepsy recordings than no-epilepsy recordings ({n_(hr_e)} vs {n_(hr_n)} hours), although patients are {n_(te.get('n_patients'))} vs {n_(tn.get('n_patients'))}. " if ratio else "") +
        "Count each patient once: cap or weight recordings per patient, average predictions per patient, and report balanced accuracy and AUROC, not plain accuracy. "
        f"For seizure detection the imbalance is much larger (seizures are about {tusz.get('seizure_pct_of_recorded_time', 'n/a')}% of TUSZ recorded time): sample background windows down or weight the loss, "
        "and score with event-based measures (share of seizures detected, false alarms per 24 hours), as in the NEDC scoring software, rather than window accuracy.")
    D.p("**Traps that give results that look too good:**")
    ov_tusz = [c for c in ch if c["a"].startswith("TUEP")]
    D.bullets([
        "**Same patient in training and testing.** Split by patient, never by recording or window: windows from one patient are very similar, so a recording-level split mostly measures recognition of the person. "
        "TUEP patients often have several sessions. Carry the cross-dataset overlaps through every split: " +
        "; ".join(f"TUEP and {NAMES.get(c['b'], c['b'])} share {n_(c['found'])} patients" for c in ch) +
        ". If you pre-train on TUSZ, remove the shared patients from pre-training or keep them out of the TUEP test folds.",
        f"**Seizures make diagnosis too easy.** {n_(seiz_pat_tuep)} TUEP epilepsy patients have annotated seizures and the no-epilepsy group has {n_(tn.get('patients_with_seizure'))}. "
        "A model can learn 'seizure present' instead of 'epilepsy'. Cut annotated seizure stretches (plus a margin, e.g. 1 minute either side) out of the TUEP inputs, and report results separately for epilepsy patients with no seizure on record.",
        "**Groups differ in things other than epilepsy.** Section 5 compares age, sex and recording year; section 3 compares montage folders, sampling rates and recording length. "
        "Antiseizure medication slows the EEG background, so a model may detect the drug rather than the disease. Test this directly: train a simple model on these side variables alone "
        "(age, sex, montage, sampling rate, year, duration, number of recordings). If it separates the groups well, the EEG model may be using the same shortcut. "
        "Harmonise montage, sampling rate and window length, and match or stratify by age.",
        "**Labels are softer than they look.** In the TUH metadata, Epilepsy = N means the EEG report did not mention epilepsy, not that epilepsy was ruled out. The TUEG Epilepsy=Y set holds candidates "
        "with no matched negatives: use it for pre-training or sanity checks, not as confirmed positives.",
        "**Tuning on the test set.** Choose settings on TUSZ dev or inner cross-validation folds only. If anyone has already looked at TUSZ eval results while developing earlier models "
        "(for example in the NeuroState_CNN_Seizure_Detection project), say so in the paper.",
        "**Different hospitals and devices.** The external Guinea-Bissau and Nigeria data use a different headset, channels and sampling rate. Use them only as an external test on shared channels, and expect lower performance.",
    ])
    D.p("**Suggested first steps.** (1) Fix the patient-level splits and write them to files before any modelling. (2) Build one preprocessing pipeline (channel names, reference, resampling, filtering, windowing, "
        "seizure removal for TUEP) and check it on a few recordings by eye. (3) Run the two baselines (microstates + logistic regression; EEGNet from scratch). "
        "(4) Add TUSZ pre-training, then the hybrid and sequence models. (5) Run the side-variable check. (6) Test on the external datasets last.")

    # ---------- 8 questions
    D.h("8. Questions the research team should ask me")
    D.bullets([
        "Data access: does our TUH data-use agreement cover every team member, and may data or derived features leave this PC? Who needs to sign?",
        "Which TUEP metadata file is authoritative (metadata_v00.xlsx or metadata_v00r.xlsx), and are the EEG report texts available to check the labels?",
        "How was the TUEG Epilepsy=Y set selected, and is a comparable set of Epilepsy=N patients available (or planned for download)?",
        "Will the rest of TUEG be downloaded? Is the partial download free to use for unlabelled pre-training?",
        "What is the end goal: a clinical question (diagnosis from a first routine EEG) or a technical benchmark (seizure detection)? This decides the main task and metric.",
        "Have TUSZ eval results already been seen during earlier work (NeuroState_CNN_Seizure_Detection)? If so, which data can still be treated as untouched test data?",
        "What computing power and storage can we use? Can preprocessed windows be stored next to the raw data?",
        "What licence and ethics terms apply to the Guinea-Bissau and Nigeria datasets, and may they be used for publication?",
        "Which input format do the existing EEGNet and microstate pipelines expect (channels, reference, sampling rate, window length)?",
        "Are the recordings with problems listed in section 6 known, and should they be excluded or repaired (on a copy, never on the original files)?",
    ], numbered=True)

    # ---------- appendix
    D.h("Appendix: sources and reproducibility")
    D.bullets([
        f"Step 1 inventory: [all] files ({g(s1, 'seconds', default='n/a')} s). Step 2 EDF headers: [all] EDFs ({g(s2, 'seconds_reading', default='n/a')} s). "
        f"Step 3 labels: [all] .csv/.csv_bi of TUSZ and TUEP. Step 4 metadata: [all] spreadsheet rows. Step 5 signal QC: [sample] of {n_(sample_n)}. "
        "Step 6 external: [all] .gz files, amplitude on a sample. Step 7 overlaps: [all].",
        "Scripts: Claude_Workspace\\reports\\scripts\\*.py. Raw totals: Claude_Workspace\\reports\\scripts\\outputs\\*.json. "
        "Per-file working tables with file paths are kept outside the report folder (Claude_Workspace\\cache\\eeg_data_report) and should not be shared.",
        "All data files were opened read-only; the scripts refuse to write anywhere outside Claude_Workspace.",
    ])
    return D


# ---------------------------------------------------------------- renderers
def md_runs(t):
    parts = re.split(r"(\*\*[^*]+\*\*)", t)
    return [(p[2:-2], True) if p.startswith("**") and p.endswith("**") else (p, False) for p in parts if p]


def render_docx(D, path):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(1.8)
        s.top_margin = s.bottom_margin = Cm(1.6)

    def para(text, style=None, size=None, italic=False):
        p = doc.add_paragraph(style=style)
        for t, b in md_runs(text):
            r = p.add_run(t)
            r.bold = b
            r.italic = italic
            if size:
                r.font.size = Pt(size)
        return p

    for b in D.blocks:
        if b[0] == "title":
            h = doc.add_heading(b[1], 0)
            para(b[2], italic=True)
        elif b[0] == "h":
            doc.add_heading(b[2], b[1])
        elif b[0] == "p":
            if b[1] == "note":
                p = para(b[2], size=8.5, italic=True)
                p.runs and [setattr(r.font.color, "rgb", RGBColor(0x55, 0x55, 0x55)) for r in p.runs]
            else:
                para(b[2])
        elif b[0] == "list":
            for it in b[2]:
                para(it, style="List Number" if b[1] else "List Bullet")
        elif b[0] == "table":
            header, rows = b[1], b[2]
            t = doc.add_table(rows=1, cols=len(header))
            t.style = "Table Grid"
            for i, h in enumerate(header):
                c = t.rows[0].cells[i]
                c.text = ""
                r = c.paragraphs[0].add_run(str(h))
                r.bold = True
                r.font.size = Pt(8)
                sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear")
                sh.set(qn("w:fill"), "DCE6F0")
                c._tc.get_or_add_tcPr().append(sh)
            trPr = t.rows[0]._tr.get_or_add_trPr()
            th = OxmlElement("w:tblHeader")
            th.set(qn("w:val"), "true")
            trPr.append(th)
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""
                    r = cells[i].paragraphs[0].add_run(str(v))
                    r.font.size = Pt(8)
            doc.add_paragraph()
    doc.save(C.out_path(path))


def render_pdf(D, path):
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=12.5, spaceAfter=5)
    note = ParagraphStyle("n", parent=body, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"), fontName="Helvetica-Oblique")
    cell = ParagraphStyle("c", parent=body, fontSize=7, leading=8.5, spaceAfter=0)
    cellh = ParagraphStyle("ch", parent=cell, fontName="Helvetica-Bold")
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1F3864"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F3864"))
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=20)

    def mk(t, style):
        s = "".join(f"<b>{escape(x)}</b>" if bold else escape(x) for x, bold in md_runs(str(t)))
        return Paragraph(s, style)

    W = A4[0] - 3.2 * cm
    story = []
    for b in D.blocks:
        if b[0] == "title":
            story += [mk(b[1], title), mk(b[2], note), Spacer(1, 6)]
        elif b[0] == "h":
            story.append(mk(b[2], h1 if b[1] == 1 else h2))
        elif b[0] == "p":
            story.append(mk(b[2], note if b[1] == "note" else body))
        elif b[0] == "list":
            items = [ListItem(mk(i, body), leftIndent=12) for i in b[2]]
            story.append(ListFlowable(items, bulletType="1" if b[1] else "bullet", start="1" if b[1] else None, leftIndent=14, bulletFontSize=8))
        elif b[0] == "table":
            header, rows = b[1], b[2]
            data = [[mk(h, cellh) for h in header]] + [[mk(v, cell) for v in r] for r in rows]
            from reportlab.pdfbase.pdfmetrics import stringWidth
            nc = len(header)
            def longest_word(txts, font):
                return max([stringWidth(w, font, 7) for t_ in txts for w in str(t_).split()] or [0])
            minw = [max(longest_word([header[i]], "Helvetica-Bold"), min(longest_word([r[i] for r in rows], "Helvetica"), 70)) + 8 for i in range(nc)]
            want = [min(max([len(str(r[i])) for r in rows] + [1]), 60) for i in range(nc)]
            if sum(minw) >= W:
                widths = [W * m / sum(minw) for m in minw]
            else:
                extra = W - sum(minw)
                widths = [m + extra * w_ / sum(want) for m, w_ in zip(minw, want)]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F0")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
            story += [t, Spacer(1, 4)]
    doc = SimpleDocTemplate(str(C.out_path(path)), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="EEG Data Report", author="NeuroState")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"EEG Data Report - page {d.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    J = {}
    for k, f in (("s1", "s1_inventory"), ("s2", "s2_edf_headers"), ("s3", "s3_labels"), ("s4", "s4_metadata"),
                 ("s5", "s5_signal_qc"), ("s6", "s6_external"), ("s7", "s7_overlaps"), ("meta", "run_meta")):
        p = P["out"] / f"{f}.json"
        if p.exists():
            J[k] = C.read_json(p)
        else:
            log(f"  warning: {p.name} missing; its sections will show n/a")
    D = build(J, P, g(J, "s5", "n_sample"))
    docx_p = P["reports"] / "EEG_Data_Report.docx"
    pdf_p = P["reports"] / "EEG_Data_Report.pdf"
    render_docx(D, docx_p)
    render_pdf(D, pdf_p)
    log(f"REPORT written: {docx_p} and {pdf_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    main(ap.parse_args())
