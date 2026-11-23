"""Step 8: TUAB v3.0.1 normal-EEG recordings that already have a copy on the drive.

Reads the TUAB file listing saved in Claude_Workspace/inventory (names and sizes only; no EEG is downloaded),
finds local EDFs with the same file name AND size, and builds a `normal_eeg` cohort of patients who are not
already in an epilepsy group (TUEP epilepsy folder, TUEG Epilepsy=Y, TUSZ Epilepsy=Y) or in the TUEP
no-epilepsy group (which keeps its own clinical label `no_epilepsy`).

Private outputs (patient IDs + paths, keep inside Claude_Workspace):
  inventory/tuab_normal_eeg_manifest.csv     patient_id, local_edf_path, tuab_split, label
  inventory/tuab_normal_eeg_exclusions.csv   patient_id, reason, n_matched_files
Aggregate output: outputs/s8_tuab_normal.json (counts only)
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import common as C

LISTING_REL = "Claude_Workspace/inventory/tuab_v301_ls_lR.txt"
MANIFEST_REL = "Claude_Workspace/inventory/tuab_normal_eeg_manifest.csv"
EXCL_REL = "Claude_Workspace/inventory/tuab_normal_eeg_exclusions.csv"
TUSZ_META = ["TUH/TUH_Additional/TUSZ_v2.0.6/DOCS/metadata_v06r.xlsx",
             "NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/DOCS/metadata_v06r.xlsx"]
# when a TUAB file has several identical local copies, report the first in this order
PREFER = ["TUEG_partial", "TUEG_epilepsy_Y", "TUSZ_train", "TUSZ_dev", "TUSZ_eval", "TUEP_epilepsy", "TUEP_no_epilepsy",
          "TUSZ_incomplete_copy", "TUAB_download", "other"]
# TUAB normal recordings downloaded on 23 Nov 2026 (rsync from the TUH server), kept in their TUAB layout
DOWNLOAD_REL = "New_TUH_Downloads/TUAB_v3.0.1_normal"


def parse_listing(path: Path) -> list[dict]:
    """Parse `ls -lR .../tuh_eeg_abnormal/v3.0.1/edf/` output."""
    rows, cur = [], None
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.endswith(":"):
                cur = line[:-1]
            elif line.lower().endswith(".edf") and cur:
                parts = line.split()
                rel = cur.split("/edf/", 1)[1] if "/edf/" in cur else ""
                bits = rel.split("/")
                m = C.RE_EDF.match(parts[-1][:-4])
                rows.append(dict(name=parts[-1], size=int(parts[4]), split=bits[0] if bits else "",
                                 label=bits[1] if len(bits) > 1 else "", montage=bits[2] if len(bits) > 2 else "",
                                 pid=m.group("pid").lower() if m else ""))
    return rows


def tusz_y_patients(P, tusz_pids: set[str]) -> set[str] | None:
    """Patients in TUSZ whose metadata_v06r.xlsx (sheet TUSZ) has Epilepsy = Y in any session."""
    import pandas as pd
    for rel in TUSZ_META:
        p = C.rel_path(rel)
        if p.is_file():
            raw = pd.read_excel(p, sheet_name="TUSZ", header=None)
            df = raw.iloc[3:].copy()
            df.columns = [str(x).strip() for x in raw.iloc[2]]
            idc = next((c for c in df.columns if "subject" in c.lower()), None)
            epc = next((c for c in df.columns if c.strip().lower() == "epilepsy"), None)
            if not idc or not epc:
                return None
            rx = re.compile(r"(?<![a-z])([a-z]{8})(?![a-z])")
            pid = df[idc].astype(str).str.lower().map(lambda s: rx.search(s).group(1) if rx.search(s) else None)
            return set(pid[df[epc].astype(str).str.strip().str.upper() == "Y"].dropna()) & tusz_pids
    return None


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 8 TUAB normal EEG already on the drive")
    listing = C.rel_path(LISTING_REL)
    if not listing.is_file():
        log(f"  {LISTING_REL} not found: step skipped")
        C.write_json(P["out"] / "s8_tuab_normal.json", dict(found=False))
        return
    tuab = parse_listing(listing)
    normal = [r for r in tuab if r["label"] == "normal"]
    abnormal = [r for r in tuab if r["label"] == "abnormal"]

    # local EDF index: (name, size) -> paths. Walk the whole root except Claude_Workspace (read-only).
    man = C.read_csv(P["cache"] / "manifest_edf.csv")
    ds_of = {os.path.normcase(r["path"]): r["dataset"] for r in man}
    dl_root = os.path.normcase(str(C.rel_path(DOWNLOAD_REL)))
    def dataset_of(path):
        p = os.path.normcase(path)
        return "TUAB_download" if p.startswith(dl_root) else ds_of.get(p, "other")
    local, by_name = defaultdict(list), defaultdict(set)
    n_local = 0
    for dp, dns, fns in C.iter_tree(P["root"]):
        if Path(dp) == P["ws"]:
            dns.clear()
        if "Claude_Workspace" in Path(dp).parts:
            continue
        for fn, sz in fns:
            if fn.lower().endswith(".edf"):
                n_local += 1
                local[(fn.lower(), sz)].append(os.path.join(dp, fn))
                by_name[fn.lower()].add(sz)

    def pick(paths):
        return sorted(paths, key=lambda p: (PREFER.index(dataset_of(p)), p))[0]

    matched, name_only, multi = [], 0, 0
    for r in normal:
        hit = local.get((r["name"].lower(), r["size"]))
        if hit:
            multi += len(hit) > 1
            matched.append(dict(r, path=pick(hit), local_dataset=dataset_of(pick(hit))))
        elif r["name"].lower() in by_name:
            name_only += 1
    abn_matched = sum(1 for r in abnormal if (r["name"].lower(), r["size"]) in local)

    grp = defaultdict(set)
    for r in man:
        if r["pid"]:
            grp[r["dataset"]].add(r["pid"])
    tusz = grp["TUSZ_train"] | grp["TUSZ_dev"] | grp["TUSZ_eval"]
    epi406 = grp["TUEP_epilepsy"] | grp["TUEG_epilepsy_Y"]
    ctrl = grp["TUEP_no_epilepsy"]
    ty = tusz_y_patients(P, tusz)
    tusz_y_new = (ty - epi406 - ctrl) if ty is not None else None

    M = {r["pid"] for r in matched}
    reason = {}
    for p in sorted(M):
        if p in epi406:
            reason[p] = "epilepsy: TUEP epilepsy folder or TUEG Epilepsy=Y"
        elif p in ctrl:
            reason[p] = "TUEP no-epilepsy group (keeps label no_epilepsy)"
        elif tusz_y_new is not None and p in tusz_y_new:
            reason[p] = "epilepsy: TUSZ Epilepsy=Y"
    final = M - set(reason)
    final_files = [r for r in matched if r["pid"] in final]

    # seizure flags for the final cohort (information only, not an exclusion)
    seiz = set()
    ev = P["cache"] / "seizure_events.csv"
    if ev.exists():
        p2pid = {r["path"]: r["pid"] for r in man}
        seiz = {p2pid.get(e["path"]) for e in C.read_csv(ev)}
    other_local = sum(1 for r in man if r["pid"] in final) - sum(1 for r in final_files if r["local_dataset"] != "other")

    C.write_csv(C.rel_path(MANIFEST_REL),
                [dict(patient_id=r["pid"], local_edf_path=r["path"], tuab_split=r["split"], label="normal_eeg")
                 for r in sorted(final_files, key=lambda r: (r["pid"], r["name"]))],
                ["patient_id", "local_edf_path", "tuab_split", "label"])
    n_by = Counter(r["pid"] for r in matched)
    C.write_csv(C.rel_path(EXCL_REL), [dict(patient_id=p, reason=why, n_matched_files=n_by[p]) for p, why in sorted(reason.items())],
                ["patient_id", "reason", "n_matched_files"])

    split_pat = defaultdict(set)
    for r in final_files:
        split_pat[r["split"]].add(r["pid"])
    res = dict(
        found=True, listing=LISTING_REL.replace("/", "\\"), manifest=MANIFEST_REL.replace("/", "\\"), exclusions_file=EXCL_REL.replace("/", "\\"),
        tuab_files=len(tuab), tuab_patients=len({r["pid"] for r in tuab}),
        tuab_normal_files=len(normal), tuab_normal_patients=len({r["pid"] for r in normal}),
        tuab_normal_by_split={s: dict(files=sum(1 for r in normal if r["split"] == s), patients=len({r["pid"] for r in normal if r["split"] == s}))
                              for s in sorted({r["split"] for r in normal})},
        tuab_abnormal_files=len(abnormal), tuab_abnormal_files_on_drive=abn_matched,
        local_edf_files_scanned=n_local,
        matched_files=len(matched), matched_patients=len(M), matched_files_with_several_local_copies=multi,
        name_match_but_size_differs=name_only,
        matched_files_by_local_dataset=dict(Counter(r["local_dataset"] for r in matched)),
        not_on_drive_files=len(normal) - len(matched),
        overlap_raw=dict(epilepsy_tuep_or_tuegY=len(M & epi406), tuep_no_epilepsy=len(M & ctrl),
                         tusz_epilepsy_Y=(len(M & tusz_y_new) if tusz_y_new is not None else None)),
        tusz_y_list_available=tusz_y_new is not None, tusz_y_new_patients=len(tusz_y_new) if tusz_y_new is not None else None,
        excluded=dict(Counter(reason.values())), excluded_patients=len(reason),
        excluded_files=sum(1 for r in matched if r["pid"] in reason),
        final_patients=len(final), final_files=len(final_files),
        final_by_split={s: dict(files=sum(1 for r in final_files if r["split"] == s), patients=len(v)) for s, v in sorted(split_pat.items())},
        final_patients_in_both_splits=len(split_pat.get("train", set()) & split_pat.get("eval", set())),
        final_by_local_dataset=dict(Counter(r["local_dataset"] for r in final_files)),
        final_patients_also_in_tusz=len(final & tusz), final_patients_with_tusz_seizure=len(final & seiz),
        final_patients_other_local_edfs=other_local,
    )
    C.write_json(P["out"] / "s8_tuab_normal.json", res)
    log(f"  TUAB normal: {res['tuab_normal_files']} files / {res['tuab_normal_patients']} patients; on drive: {res['matched_files']} / {res['matched_patients']}")
    log(f"  excluded {res['excluded_patients']} patients ({res['excluded']}); final normal_eeg: {res['final_patients']} patients, {res['final_files']} files")
    log("STEP 8 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    main(ap.parse_args())
