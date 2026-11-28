"""splits.py on a cohort made from the fake package (fake_data.py)."""
import pandas as pd
import pytest

from experiments.epilepsy import fake_data as fd
from experiments.epilepsy import splits as S


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    pkg = fd.write_package(tmp_path_factory.mktemp("fk") / "data", seed=0)
    meta = pd.read_csv(pkg / "meta.csv")
    lab = dict(pd.read_csv(pkg / "labels.csv")[["pid", "label"]].values)
    c = meta[["pid", "group"]].copy()
    c["label"] = c.pid.map(lab)
    c["tusz_split"] = ""
    seiz = sorted(c.pid[c.group == "tusz_seiz"])
    for p, s in zip(seiz, ["train", "train", "train", "dev", "eval", "eval"]):
        c.loc[c.pid == p, "tusz_split"] = s
    tusz_y = sorted(c.pid[c.group == "tusz_y"])
    c.loc[c.pid == tusz_y[0], "tusz_split"] = "train"     # a TUSZ Y patient in TUSZ train ...
    c.loc[c.pid == tusz_y[1], "tusz_split"] = "dev"
    c["has_tuab_normal"] = False
    c.loc[c.pid == tusz_y[0], "has_tuab_normal"] = True   # ... who is also a T3 held-out patient
    c.loc[c.pid == sorted(c.pid[c.group == "tueg_y"])[0], "has_tuab_normal"] = True
    for p in sorted(c.pid[c.group == "tuep_epi"])[:2]:
        c.loc[c.pid == p, "has_tuab_normal"] = True
    tuab_eval = sorted(c.pid[c.group == "tuab_eval"])[0]
    c.loc[c.pid == tuab_eval, "tusz_split"] = "train"     # a T2 patient in TUSZ train
    return c


def test_splits_pass_checks(cohort):
    s = S.make_splits(cohort)
    S.check(s, cohort)
    tuep = set(cohort.pid[cohort.group.isin(S.TUEP)])
    for r in range(3):
        assert set(s[f"t1_r{r}"].pid) == tuep
        assert list(s[f"t1_r{r}"].columns) == ["pid", "fold", "label"]
    assert list(s["t1_r0_f0_val"].columns) == ["pid", "label"]
    assert list(s["t2"].columns) == ["pid"] and list(s["run_B_extra"].columns) == ["pid", "label"]
    assert len(s["t3_heldout"]) == 2 and len(s["t3_tuep"]) == 2
    held = set(s["t3_heldout"].pid) | set(s["t2"].pid)
    for k in ("tusz_train", "tusz_dev", "tusz_eval", "run_B_extra", "run_C_extra"):
        assert not held & set(s[k].pid), k                 # test patients never in training / stage 1 lists
    assert len(s["tusz_train"]) == 3 and len(s["tusz_dev"]) == 2 and len(s["tusz_eval"]) == 2


def test_folds_are_balanced(cohort):
    s = S.make_splits(cohort)
    for r in range(3):
        t1 = s[f"t1_r{r}"]
        for lab in ("epilepsy", "no_epilepsy"):
            per = t1[t1.label == lab].groupby("fold").size()
            assert per.max() - per.min() <= 1


def test_same_cohort_same_splits(cohort):
    a, b = S.make_splits(cohort), S.make_splits(cohort.sample(frac=1, random_state=3))
    for k in a:
        assert a[k].sort_values(list(a[k].columns)).reset_index(drop=True).equals(
            b[k].sort_values(list(b[k].columns)).reset_index(drop=True)), k


def _leaky(cohort, edit):
    s = S.make_splits(cohort)
    edit(s)
    with pytest.raises(S.LeakError):
        S.check(s, cohort)


def test_leak_test_patient_in_validation(cohort):
    def edit(s):
        t1 = s["t1_r1"]
        p = t1.pid[t1.fold == 2].iloc[0]
        s["t1_r1_f2_val"] = pd.concat([s["t1_r1_f2_val"], pd.DataFrame({"pid": [p], "label": ["epilepsy"]})])
    _leaky(cohort, edit)


def test_leak_t1_patient_in_run_extra(cohort):
    def edit(s):
        p = s["t1_r0"].pid.iloc[0]
        s["run_B_extra"] = pd.concat([s["run_B_extra"], pd.DataFrame({"pid": [p], "label": ["epilepsy"]})])
    _leaky(cohort, edit)


def test_leak_t2_patient_in_run_c(cohort):
    def edit(s):
        s["run_C_extra"] = pd.concat([s["run_C_extra"], pd.DataFrame({"pid": [s["t2"].pid.iloc[0]], "label": ["normal_eeg"]})])
    _leaky(cohort, edit)


def test_leak_t3_patient_in_stage1(cohort):
    def edit(s):
        s["tusz_train"] = pd.concat([s["tusz_train"], pd.DataFrame({"pid": [s["t3_heldout"].pid.iloc[0]]})])
    _leaky(cohort, edit)


def test_leak_external_patient_in_training(cohort):
    def edit(s):
        p = cohort.pid[cohort.group == "nigeria"].iloc[0]
        s["run_C_extra"] = pd.concat([s["run_C_extra"], pd.DataFrame({"pid": [p], "label": ["epilepsy"]})])
    _leaky(cohort, edit)


def test_patient_in_two_groups(cohort):
    dup = pd.concat([cohort, cohort[cohort.group == "tuab_train"].head(1).assign(group="tueg_y")])
    s = S.make_splits(cohort)
    with pytest.raises(S.LeakError):
        S.check(s, dup)


def test_expect_reports_mismatches(cohort):
    s = S.make_splits(cohort)
    found = S.counts(s, cohort)
    bad = S.compare(found)
    assert any(b.startswith("patients tuep_epi: expected 100") for b in bad)
    exact = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in found.items()}
    assert S.compare(found, {k: exact[k] for k in ("t2", "run_B_extra", "patients tuep_epi")}) == []
