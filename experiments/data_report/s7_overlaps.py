"""Step 7: patient-ID overlaps between datasets (counts only), checked against the numbers in the brief,
plus session-level overlap via TUEP DOCS/sessions_common_with_tusz.list and label conflicts
(e.g. a TUEP no-epilepsy patient with seizures annotated in TUSZ). Uses ALL files.
Aggregate output: outputs/s7_overlaps.json
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import common as C

EXPECTED = {("TUEP_epilepsy|TUEP_no_epilepsy", "TUSZ_train"): 17, ("TUEP_epilepsy|TUEP_no_epilepsy", "TUSZ_dev"): 3,
            ("TUEG_epilepsy_Y", "TUSZ_train"): 4, ("TUEG_epilepsy_Y", "TUSZ_eval"): 2}


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 7 overlaps")
    man = C.read_csv(P["cache"] / "manifest_edf.csv")
    pats, sess = defaultdict(set), defaultdict(set)
    stems = defaultdict(set)
    for r in man:
        if r["pid"]:
            pats[r["dataset"]].add(r["pid"])
            sess[r["dataset"]].add(f"{r['pid']}_{r['ses']}")
            stems[r["dataset"]].add(Path(r["path"]).stem)
    keys = [k for k in C.DS if pats.get(k)]
    matrix = {a: {b: len(pats[a] & pats[b]) for b in keys} for a in keys}
    tuep = pats["TUEP_epilepsy"] | pats["TUEP_no_epilepsy"]
    def get(a):
        return tuep if a == "TUEP_epilepsy|TUEP_no_epilepsy" else pats[a]
    checks = [dict(a=a, b=b, expected=exp, found=len(get(a) & pats[b])) for (a, b), exp in EXPECTED.items()]
    tusz_all = pats["TUSZ_train"] | pats["TUSZ_dev"] | pats["TUSZ_eval"]
    seiz_pats = set()
    ev = P["cache"] / "seizure_events.csv"
    path2pid = {r["path"]: r["pid"] for r in man}
    if ev.exists():
        for r in C.read_csv(ev):
            if r["dataset"].startswith("TUSZ"):
                seiz_pats.add(path2pid.get(r["path"]))
    conflicts = dict(
        tuep_epi_and_no_epi=len(pats["TUEP_epilepsy"] & pats["TUEP_no_epilepsy"]),
        tuep_no_epi_in_tusz=len(pats["TUEP_no_epilepsy"] & tusz_all),
        tuep_no_epi_with_tusz_seizure=len(pats["TUEP_no_epilepsy"] & seiz_pats),
        tusz_split_overlap=dict(train_dev=len(pats["TUSZ_train"] & pats["TUSZ_dev"]), train_eval=len(pats["TUSZ_train"] & pats["TUSZ_eval"]),
                                dev_eval=len(pats["TUSZ_dev"] & pats["TUSZ_eval"])),
        tueg_Y_in_tuep=len(pats["TUEG_epilepsy_Y"] & tuep),
        tueg_partial_in_labelled=len(pats["TUEG_partial"] & (tuep | tusz_all)),
        tueg_partial_and_tueg_Y=len(pats["TUEG_partial"] & pats["TUEG_epilepsy_Y"]),
    )
    inc = stems.get("TUSZ_incomplete_copy", set())
    complete = stems["TUSZ_train"] | stems["TUSZ_dev"] | stems["TUSZ_eval"]
    dup = dict(n_files=len(inc), present_in_complete_copy=len(inc & complete))
    # sessions_common_with_tusz.list (read-only)
    common_list = None
    for p in C.rel_path("TUH/TUEP").rglob("sessions_common_with_tusz*"):
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        keys_l = set()
        for l in lines:
            m = re.search(r"([a-z]{8})[_/\\]+(s\d{3})", l.lower())
            if m:
                keys_l.add(f"{m.group(1)}_{m.group(2)}")
        tusz_s = sess["TUSZ_train"] | sess["TUSZ_dev"] | sess["TUSZ_eval"]
        tuep_s = sess["TUEP_epilepsy"] | sess["TUEP_no_epilepsy"]
        common_list = dict(file=p.name, n_lines=len(lines), n_sessions_parsed=len(keys_l),
                           found_in_tuep=len(keys_l & tuep_s), found_in_tusz=len(keys_l & tusz_s),
                           computed_tuep_tusz_session_overlap=len(tuep_s & tusz_s))
        break
    uniq = dict(main=len(set().union(*[pats[k] for k in C.MAIN if k in pats])),
                labelled=len(set().union(*[pats[k] for k in C.LABELLED if k in pats])))
    C.write_json(P["out"] / "s7_overlaps.json", dict(patients=dict((k, len(v)) for k, v in pats.items()), unique_patients=uniq, matrix=matrix,
                                                     checks_vs_brief=checks, conflicts=conflicts, tusz_incomplete_copy=dup,
                                                     sessions_common_list=common_list, source="all files (patient IDs from file names)"))
    log("STEP 7 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    main(ap.parse_args())
