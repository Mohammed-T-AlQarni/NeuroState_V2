"""Split files for the NeuroState epilepsy study (format: ops repo shared/INTERFACES.md, `splits/`).

Input: the cohort table written by labels.py (pid, group, label, tusz_split, has_tuab_normal, ...).

    T1   TUEP patients, StratifiedGroupKFold(5, shuffle=True, random_state=100*r), repeats r = 0, 1, 2;
         validation = stratified 15% of the training patients, random_state=100*r+f.
    T2   TUAB eval normals.
    T3   held-out: TUEG Y / TUSZ Y patients who also have a TUAB normal recording (kept out of all training);
         tuep: TUEP epilepsy patients with a TUAB normal recording (tested inside their T1 fold).
    TUSZ train / dev / eval: the official splits minus every TUEP patient and every other test patient
         (T2, T3 held-out): test patients are never used for pretraining (RULES 7).
    run_B_extra = TUSZ Y + TUEG Y minus T3 held-out;  run_C_extra = run_B_extra + TUAB train.

check() stops with an error if any test patient (T1 fold, T2, T3 held-out, T4) is in a training, validation or
stage 1 list, or a patient is in two groups. --expect compares the counts with shared/PLAN.md.

Usage:
    python -m experiments.epilepsy.splits --cohort <work>/cohort.csv --out %NS_DATA% --expect
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

N_REPEATS, N_FOLDS, VAL_FRAC = 3, 5, 0.15
TUEP = ("tuep_epi", "tuep_ctrl")
EXTERNAL = ("nigeria", "guinea_bissau")

# counts in shared/PLAN.md (24 / 27 Nov 2026)
EXPECT = {
    "patients tuep_epi": 100, "patients tuep_ctrl": 100, "patients tueg_y": 306, "patients tusz_y": 295,
    "patients tuab_train": 1151, "patients tuab_eval": 148,
    "T1 test patients per fold (epilepsy)": 20, "T1 test patients per fold (no_epilepsy)": 20,
    "t2": 148, "t3_heldout": 23, "t3_heldout tueg_y": 1, "t3_heldout tusz_y": 22, "t3_tuep": 29,
    "run_B_extra": 578, "run_B_extra tusz_y": 273, "run_B_extra tueg_y": 305, "run_C_extra": 1729,
    "run_C_extra tuab_train": 1151, "tusz_train": 562, "tusz_dev": 50, "tusz_eval": 43,
    "patients nigeria epilepsy": 128, "patients nigeria no_epilepsy": 97,
    "patients guinea_bissau epilepsy": 51, "patients guinea_bissau no_epilepsy": 46,
}


class LeakError(AssertionError):
    pass


def _bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def make_splits(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c = cohort.copy()
    c["has_tuab_normal"] = _bool(c["has_tuab_normal"])
    s = {}
    tuep = c[c.group.isin(TUEP)].sort_values("pid").reset_index(drop=True)
    y = (tuep.label == "epilepsy").astype(int).to_numpy()
    for r in range(N_REPEATS):
        fold = [-1] * len(tuep)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=100 * r)
        for f, (_, te) in enumerate(sgkf.split(tuep.pid, y, groups=tuep.pid)):
            for i in te:
                fold[i] = f
        t1 = pd.DataFrame({"pid": tuep.pid, "fold": fold, "label": tuep.label})
        s[f"t1_r{r}"] = t1
        for f in range(N_FOLDS):
            tr = t1[t1.fold != f]
            _, val = train_test_split(tr, test_size=VAL_FRAC, stratify=tr.label, random_state=100 * r + f)
            s[f"t1_r{r}_f{f}_val"] = val[["pid", "label"]].sort_values("pid").reset_index(drop=True)

    def pids(mask):
        return sorted(c.pid[mask])

    s["t2"] = pd.DataFrame({"pid": pids(c.group == "tuab_eval")})
    t3h = pids(c.group.isin(["tueg_y", "tusz_y"]) & c.has_tuab_normal)
    s["t3_heldout"] = pd.DataFrame({"pid": t3h})
    s["t3_tuep"] = pd.DataFrame({"pid": pids((c.group == "tuep_epi") & c.has_tuab_normal)})
    test_other = set(s["t2"].pid) | set(t3h) | set(c.pid[c.group.isin(EXTERNAL)])
    for split in ("train", "dev", "eval"):
        m = (c.tusz_split == split) & ~c.group.isin(TUEP) & ~c.pid.isin(test_other)
        s[f"tusz_{split}"] = pd.DataFrame({"pid": pids(m)})
    b = c[c.group.isin(["tusz_y", "tueg_y"]) & ~c.pid.isin(t3h)]
    s["run_B_extra"] = b[["pid", "label"]].sort_values("pid").reset_index(drop=True)
    cc = pd.concat([b, c[c.group == "tuab_train"]])
    s["run_C_extra"] = cc[["pid", "label"]].sort_values("pid").reset_index(drop=True)
    return s


def check(s: dict[str, pd.DataFrame], cohort: pd.DataFrame):
    """Raise LeakError on any leak. Messages give counts and list names, never patient codes."""
    problems = []
    if not cohort.pid.is_unique:
        problems.append(f"{int(cohort.pid.duplicated().sum())} patients appear in two groups")
    group = dict(zip(cohort.pid, cohort.group))
    tuep = set(cohort.pid[cohort.group.isin(TUEP)])
    train_lists = {k: set(v.pid) for k, v in s.items() if k.startswith("run_") or k in ("tusz_train", "tusz_dev")}
    test_other = {"t2": set(s["t2"].pid), "t3_heldout": set(s["t3_heldout"].pid),
                  "T4": set(cohort.pid[cohort.group.isin(EXTERNAL)])}
    for r in range(N_REPEATS):
        t1 = s[f"t1_r{r}"]
        if set(t1.pid) != tuep or not t1.pid.is_unique:
            problems.append(f"t1_r{r} does not list every TUEP patient exactly once")
        for f in range(N_FOLDS):
            test = set(t1.pid[t1.fold == f])
            train = set(t1.pid[t1.fold != f])
            val = set(s[f"t1_r{r}_f{f}_val"].pid)
            if test & train:
                problems.append(f"t1_r{r} fold {f}: {len(test & train)} test patients also in training folds")
            if test & val:
                problems.append(f"t1_r{r}_f{f}_val: {len(test & val)} test patients in validation")
            if not val <= train:
                problems.append(f"t1_r{r}_f{f}_val: {len(val - train)} patients not from the training folds")
            for k, v in train_lists.items():
                if test & v:
                    problems.append(f"{k}: {len(test & v)} T1 test patients of r{r} fold {f}")
    for name, test in test_other.items():
        for k, v in train_lists.items():
            if test & v:
                problems.append(f"{k}: {len(test & v)} {name} test patients")
        for r in range(N_REPEATS):
            if test & set(s[f"t1_r{r}"].pid):
                problems.append(f"t1_r{r}: {len(test & set(s[f't1_r{r}'].pid))} {name} test patients")
    for k in ("tusz_train", "tusz_dev", "tusz_eval"):
        if set(s[k].pid) & tuep:
            problems.append(f"{k}: {len(set(s[k].pid) & tuep)} TUEP patients")
    tusz_sets = [set(s[k].pid) for k in ("tusz_train", "tusz_dev", "tusz_eval")]
    if tusz_sets[0] & tusz_sets[1] or tusz_sets[0] & tusz_sets[2] or tusz_sets[1] & tusz_sets[2]:
        problems.append("a patient is in two TUSZ splits")
    if any(group.get(p) not in ("tuep_epi",) for p in s["t3_tuep"].pid):
        problems.append("t3_tuep holds patients outside tuep_epi")
    if problems:
        raise LeakError("split check failed:\n  " + "\n  ".join(problems))


def counts(s, cohort) -> dict:
    group = dict(zip(cohort.pid, cohort.group))
    out = {}
    for (g, l), n in cohort.groupby(["group", "label"], dropna=False).size().items():
        if g in EXTERNAL:
            out[f"patients {g} {l}"] = int(n)
    for g, n in cohort.group.value_counts().items():
        out[f"patients {g}"] = int(n)
    for r in range(N_REPEATS):
        t1 = s[f"t1_r{r}"]
        for lab in ("epilepsy", "no_epilepsy"):
            per = t1[t1.label == lab].groupby("fold").size()
            key = f"T1 test patients per fold ({lab})"
            out.setdefault(key, sorted(set(per.tolist())))
            out[key] = sorted(set(out[key]) | set(per.tolist()))
    for k in ("t2", "t3_heldout", "t3_tuep", "run_B_extra", "run_C_extra", "tusz_train", "tusz_dev", "tusz_eval"):
        out[k] = len(s[k])
    for k, gs in (("t3_heldout", ["tueg_y", "tusz_y"]), ("run_B_extra", ["tusz_y", "tueg_y"]),
                  ("run_C_extra", ["tuab_train"])):
        for g in gs:
            out[f"{k} {g}"] = sum(group.get(p) == g for p in s[k].pid)
    out["validation patients per fold"] = sorted({len(v) for k, v in s.items() if k.endswith("_val")})
    return out


def compare(found: dict, expect=EXPECT) -> list[str]:
    bad = []
    for k, v in expect.items():
        f = found.get(k)
        if isinstance(f, list):
            if f != [v]:
                bad.append(f"{k}: expected {v}, found {f}")
        elif f != v:
            bad.append(f"{k}: expected {v}, found {f}")
    return bad


def write(s, out):
    d = Path(out) / "splits"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in s.items():
        v.to_csv(d / f"{k}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True, help="package folder; files go to <out>/splits/")
    ap.add_argument("--expect", action="store_true", help="compare counts with shared/PLAN.md")
    a = ap.parse_args()
    cohort = pd.read_csv(a.cohort, dtype={"tusz_split": str, "tuab_split": str}, keep_default_na=False)
    cohort["label"] = cohort["label"].replace("", None)
    s = make_splits(cohort)
    check(s, cohort)
    write(s, a.out)
    found = counts(s, cohort)
    for k, v in found.items():
        print(f"{k}: {v}")
    if a.expect:
        bad = compare(found)
        print(f"\n--expect: {len(bad)} mismatch(es) with shared/PLAN.md")
        for b in bad:
            print("  " + b)


if __name__ == "__main__":
    main()
