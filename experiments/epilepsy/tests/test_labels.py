"""labels.py on a fake drive (TUH-like folders, metadata workbooks, TUAB manifest and listing, external CSVs)."""
import numpy as np
import pandas as pd
import pytest

from _fakes import edf_name, fake_pid, write_edf, write_emotiv_csv
from experiments.epilepsy import labels as L

TINY = np.zeros((2, 512))


def _edf(path, age=40, sex="F"):
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_edf(path, TINY, 256, ["EEG FP1-REF", "EEG FP2-REF"], age=age, sex=sex)


def _sheet(path, sheet, rows, header_row):
    """Workbook with a header at Excel row `header_row` (1-based) and columns Subject ID / Epilepsy / Other."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Subject ID", "Electrographic Seizure", "Epilepsy", "Other Diagnosis"]
    data = [["note"] + [""] * 3] * (header_row - 1) + [cols] + [[f"{p}_s001", "N", e, o] for p, e, o in rows]
    pd.DataFrame(data).to_excel(path, sheet_name=sheet, header=False, index=False)


@pytest.fixture(scope="module")
def drive(tmp_path_factory):
    root = tmp_path_factory.mktemp("drive") / "NeuroState"
    pid = {k: fake_pid(i) for i, k in enumerate(
        ["epi1", "epi2", "epi3", "ctl1", "ctl2", "ctl3", "tueg1", "tueg2", "szY1", "szY2", "szN1", "szN2",
         "tab1", "tab2", "tab3"])}
    # TUEP
    for k in ("epi1", "epi2", "epi3"):
        _edf(root / "TUH/TUEP/00_epilepsy" / pid[k] / "s001_2010/02_tcp_le" / edf_name(pid[k]))
    for k in ("ctl1", "ctl2", "ctl3"):
        _edf(root / "TUH/TUEP/01_no_epilepsy" / pid[k] / "s001_2011/01_tcp_ar" / edf_name(pid[k]))
    _edf(root / "TUH/TUEP/00_epilepsy" / pid["epi1"] / "s002_2012/02_tcp_le" / edf_name(pid["epi1"], 2), age=41)
    # TUEG Epilepsy = Y folder (tueg2 is also in TUSZ train)
    for k in ("tueg1", "tueg2"):
        _edf(root / "TUH/TUH_Additional/TUEG_v2.0.2/epilepsy_Y/edf/000" / pid[k] / "s001_2016/01_tcp_ar" / edf_name(pid[k]))
    # TUSZ: epi1 (a TUEP patient) and szY1, szN1, tueg2 in train; szY2 in dev; szN2 in eval
    tusz = root / "NeuroState_CNN_Seizure_Detection/data/TUSZ_v2.0.6/edf"
    for k, s in (("epi1", "train"), ("szY1", "train"), ("szN1", "train"), ("tueg2", "train"), ("szY2", "dev"),
                 ("szN2", "eval")):
        _edf(tusz / s / pid[k] / "s001_2003/02_tcp_le" / edf_name(pid[k]))
    _sheet(root / "TUH/TUH_Additional/TUSZ_v2.0.6/DOCS/metadata_v06r.xlsx", "TUSZ",
           [(pid["szY1"], "Y", ""), (pid["szY2"], "N", ""), (pid["szY2"], "Y", ""), (pid["szN1"], "N", ""),
            (pid["epi1"], "Y", ""), (pid["tueg2"], "Y", ""), (pid["szN2"], "N", "history of epilepsy"),
            (pid["ctl3"], "Y", "")], header_row=3)
    _sheet(root / "TUH/TUEP/DOCS/metadata_v00r.xlsx", "TUEP",
           [(pid["epi1"], "Y", ""), (pid["ctl1"], "N", ""), (pid["ctl2"], "Y", "")], header_row=1)
    # TUAB normal manifest (tab1, tab2 train; tab3 eval; epi2 must be dropped: already in an epilepsy group)
    inv = root / "Claude_Workspace/inventory"
    inv.mkdir(parents=True)
    man = []
    for k, s in (("tab1", "train"), ("tab2", "train"), ("tab3", "eval"), ("epi2", "train"), ("szN2", "eval")):
        p = root / f"New_TUH_Downloads/TUAB_v3.0.1_normal/edf/{s}/normal/01_tcp_ar" / edf_name(pid[k])
        _edf(p)
        man.append(dict(patient_id=pid[k], local_edf_path=str(p), tuab_split=s, label="normal_eeg"))
    pd.DataFrame(man).to_csv(inv / "tuab_normal_eeg_manifest.csv", index=False)
    # TUAB listing: every normal recording, incl. patients in epilepsy groups (for T3)
    lines = []
    for k, s in (("tab1", "train"), ("tab2", "train"), ("tab3", "eval"), ("epi2", "train"), ("szY1", "train"),
                 ("szN2", "eval")):
        lines += [f"data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf/{s}/normal/01_tcp_ar:",
                  f"-rw-r--r-- 1 u g 1000 Jan 1 2020 {edf_name(pid[k])}", ""]
    lines += ["data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf/train/abnormal/01_tcp_ar:",
              f"-rw-r--r-- 1 u g 1000 Jan 1 2020 {edf_name(pid['tueg1'])}", ""]
    (inv / "tuab_v301_ls_lR.txt").write_text("\n".join(lines))
    # external
    ext = root / "External_Datasets"
    (ext / "EEGs_Nigeria").mkdir(parents=True)
    (ext / "EEGs_Guinea-Bissau").mkdir(parents=True)
    for f in ("signal-1-1.csv.gz", "signal-2-1.csv.gz", "signal-2-2.csv.gz"):
        write_emotiv_csv(ext / "EEGs_Nigeria" / f, seconds=2)
    write_emotiv_csv(ext / "EEGs_Guinea-Bissau" / "signal-1.csv.gz", seconds=2)
    pd.DataFrame({"subject.id": [1, 2, 2, 3], "session.id": [1, 1, 2, 1], "Group": ["epilepsy", "control", "control", "control"],
                  "csv.file": ["signal-1-1.csv.gz", "signal-2-1.csv.gz", "signal-2-2.csv.gz", "missing.csv.gz"]}
                 ).to_csv(ext / "metadata_nigeria.csv", index=False)
    pd.DataFrame({"subject.id": [1], "Group": ["Epilepsy"], "Eyes.condition": ["x"]}).to_csv(
        ext / "metadata_guineabissau.csv", index=False)
    return root, pid


@pytest.fixture(scope="module")
def built(drive):
    root, pid = drive
    cohort, recs, conflicts, info = L.build(root)
    return root, pid, cohort.set_index("pid"), recs, conflicts, info


def test_groups(built):
    _, pid, c, _, _, _ = built
    g = c.group.to_dict()
    assert g[pid["epi1"]] == "tuep_epi" and g[pid["ctl3"]] == "tuep_ctrl"
    assert g[pid["tueg2"]] == "tueg_y"                       # TUEG Y wins over TUSZ Y
    assert g[pid["szY1"]] == "tusz_y" and g[pid["szY2"]] == "tusz_y"   # Y in any session
    assert g[pid["szN1"]] == "tusz_seiz"                     # TUSZ, no diagnosis group: stage 1 only
    assert g[pid["tab1"]] == "tuab_train" and g[pid["tab3"]] == "tuab_eval"
    assert g[pid["epi2"]] == "tuep_epi"                      # never normal_eeg
    assert g[pid["szN2"]] == "tuab_eval"                     # TUAB normal beats stage-1-only
    assert c.label[pid["szN1"]] is None or pd.isna(c.label[pid["szN1"]])
    assert g["ng0001"] == "nigeria" and c.label["ng0001"] == "epilepsy"
    assert c.label["ng0002"] == "no_epilepsy" and "ng0003" not in g   # no signal file -> not a patient
    assert c.label["gb0001"] == "epilepsy"
    assert c.index.is_unique


def test_tuab_and_tusz_flags(built):
    _, pid, c, _, _, info = built
    assert bool(c.has_tuab_normal[pid["szY1"]]) and bool(c.has_tuab_normal[pid["epi2"]])
    assert not bool(c.has_tuab_normal[pid["tueg1"]])          # abnormal TUAB recording only
    assert c.tusz_split[pid["szY2"]] == "dev" and c.tusz_split[pid["epi1"]] == "train"
    assert info["tuab_manifest_patients_already_in_other_group"] == 1


def test_recordings(built):
    _, pid, _, recs, _, _ = built
    d = recs[recs.role == "diag"]
    assert (d.pid == pid["epi1"]).sum() == 2                  # only its TUEP files, not the TUSZ copy
    assert set(d.group[d.pid == pid["szY1"]]) == {"tusz_y"}
    s = recs[recs.role == "seizure"]
    assert pid["epi1"] not in set(s.pid)                      # TUEP patients never in stage 1
    assert {pid["szY1"], pid["szN1"], pid["tueg2"], pid["szY2"], pid["szN2"]} <= set(s.pid)
    assert set(d.path[d.pid == "ng0002"].map(lambda p: p.split("signal-")[1])) == {"2-1.csv.gz"}   # first session
    first = d[d.pid == pid["epi1"]].sort_values(["ses", "tok"]).iloc[0]
    assert first.ses == "s001" and first.year == 2010 and first.montage == "02_tcp_le"


def test_conflicts_flagged_not_resolved(built):
    _, pid, c, _, conflicts, _ = built
    issues = {(r.pid, r.issue) for r in conflicts.itertuples()}
    assert (pid["ctl2"], "Epilepsy = Y in TUEP metadata_v00r") in issues
    assert (pid["ctl3"], "Epilepsy = Y in TUSZ metadata_v06r") in issues
    assert any(p == pid["szN2"] and "free text" in i for p, i in issues)
    assert c.group[pid["ctl2"]] == "tuep_ctrl"                # flagged, label unchanged


def test_meta_and_write(built, tmp_path):
    _, pid, c, recs, conflicts, _ = built
    cohort = c.reset_index()
    meta = L.make_meta(cohort, recs)
    m = meta.set_index("pid")
    assert list(meta.columns) == L.META_COLS
    assert m.age[pid["epi1"]] == 40 and m.sex[pid["epi1"]] == "F" and m.sfreq_orig[pid["epi1"]] == 256
    assert m.year[pid["epi1"]] == 2010 and m.montage[pid["ctl1"]] == "01_tcp_ar"
    L.write(cohort, recs, conflicts, meta, tmp_path / "out", tmp_path / "work")
    lab = pd.read_csv(tmp_path / "out" / "labels.csv")
    assert list(lab.columns) == L.LABELS_COLS
    assert "tusz_seiz" not in set(lab.group) and lab.pid.is_unique
    assert set(lab.label) == {"epilepsy", "no_epilepsy", "normal_eeg"}
    for f in ("cohort.csv", "recordings.csv", "label_conflicts.csv"):
        assert (tmp_path / "work" / f).exists() and not (tmp_path / "out" / f).exists()
