"""The fake package must match NeuroState_Epilepsy_Ops shared/INTERFACES.md exactly.

Run from the repo root:  python -m pytest experiments/epilepsy/tests -q
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, so plain `pytest` also works
from experiments.epilepsy import fake_data as fd  # noqa: E402

ALLOWED_LABELS = {"epilepsy", "no_epilepsy", "normal_eeg"}
ALLOWED_GROUPS = {"tuep_epi", "tuep_ctrl", "tueg_y", "tusz_y", "tuab_train", "tuab_eval", "tusz_seiz",
                  "nigeria", "guinea_bissau"}
SPLIT_FILES = (["t1_r0", "t1_r1", "t1_r2"] + [f"t1_r{r}_f{f}_val" for r in range(3) for f in range(5)]
               + ["t2", "t3_heldout", "t3_tuep", "tusz_train", "tusz_dev", "tusz_eval", "run_B_extra", "run_C_extra"])


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    return fd.write_package(tmp_path_factory.mktemp("fake_pkg") / "data", seed=0)


def _split(pkg, name):
    return pd.read_csv(pkg / "splits" / f"{name}.csv")


def test_top_level_files(pkg):
    for name in ["MANIFEST.sha256", "PACKAGE_INFO.json", "labels.csv", "meta.csv"]:
        assert (pkg / name).is_file(), name
    for name in SPLIT_FILES:
        assert (pkg / "splits" / f"{name}.csv").is_file(), name
    for d in ["windows19", "windows10", "seizure"]:
        assert (pkg / d).is_dir(), d


def test_channel_order():
    assert fd.CH19 == "FP1 FP2 F7 F3 FZ F4 F8 T3 C3 CZ C4 T4 T5 P3 PZ P4 T6 O1 O2".split()
    assert fd.CH10 == "F7 F3 F4 F8 T3 T4 T5 T6 O1 O2".split()
    assert [fd.CH19[i] for i in fd.IDX10] == fd.CH10


def test_labels_and_meta(pkg):
    lab = pd.read_csv(pkg / "labels.csv")
    meta = pd.read_csv(pkg / "meta.csv")
    assert list(lab.columns) == fd.LABELS_COLS
    assert list(meta.columns) == fd.META_COLS
    assert lab.pid.is_unique and meta.pid.is_unique
    assert set(lab.label) <= ALLOWED_LABELS
    assert set(meta.group) == ALLOWED_GROUPS
    assert set(lab.pid) <= set(meta.pid)
    # only fake codes, never anything shaped like a TUH code
    for p in meta.pid:
        assert re.fullmatch(r"fake\d{4}", p), p
    expected = {"tuep_epi": "epilepsy", "tuep_ctrl": "no_epilepsy", "tueg_y": "epilepsy", "tusz_y": "epilepsy",
                "tuab_train": "normal_eeg", "tuab_eval": "normal_eeg"}
    for g, l in expected.items():
        assert set(lab.label[lab.group == g]) == {l}, g


def test_t1_folds(pkg):
    lab = pd.read_csv(pkg / "labels.csv")
    tuep = set(lab.pid[lab.group.isin(["tuep_epi", "tuep_ctrl"])])
    for r in range(3):
        t1 = _split(pkg, f"t1_r{r}")
        assert list(t1.columns) == ["pid", "fold", "label"]
        assert set(t1.pid) == tuep and t1.pid.is_unique
        assert set(t1.fold) == set(range(5))
        for f in range(5):
            test = t1[t1.fold == f]
            assert set(test.label) == {"epilepsy", "no_epilepsy"}, (r, f)
            val = _split(pkg, f"t1_r{r}_f{f}_val")
            assert list(val.columns) == ["pid", "label"]
            train_pids = set(t1.pid[t1.fold != f])
            # no T1 test patient in its fold's training or validation lists
            assert not set(val.pid) & set(test.pid), (r, f)
            assert set(val.pid) <= train_pids, (r, f)
            assert set(val.label) == {"epilepsy", "no_epilepsy"}, (r, f)
            # run B / C extras never contain TUEP patients
            for run in ["run_B_extra", "run_C_extra"]:
                assert not set(_split(pkg, run).pid) & set(test.pid)


def test_other_splits(pkg):
    meta = pd.read_csv(pkg / "meta.csv")
    group = dict(zip(meta.pid, meta.group))
    t1_pids = set(_split(pkg, "t1_r0").pid)
    t2, t3h, t3t = _split(pkg, "t2"), _split(pkg, "t3_heldout"), _split(pkg, "t3_tuep")
    b, c = _split(pkg, "run_B_extra"), _split(pkg, "run_C_extra")
    assert {group[p] for p in t2.pid} == {"tuab_eval"}
    assert {group[p] for p in t3h.pid} <= {"tueg_y", "tusz_y"}
    assert {group[p] for p in t3t.pid} == {"tuep_epi"}
    # held-out T3 and T2 patients are never trained on
    for held in (t2, t3h):
        assert not set(held.pid) & (set(b.pid) | set(c.pid) | t1_pids)
    assert {group[p] for p in b.pid} == {"tueg_y", "tusz_y"}
    assert set(b.pid) < set(c.pid)
    assert {group[p] for p in set(c.pid) - set(b.pid)} == {"tuab_train"}
    seiz = [set(_split(pkg, n).pid) for n in ["tusz_train", "tusz_dev", "tusz_eval"]]
    assert all(len(s) > 0 for s in seiz)
    assert not (seiz[0] & seiz[1] or seiz[0] & seiz[2] or seiz[1] & seiz[2])
    assert {group[p] for s in seiz for p in s} == {"tusz_seiz"}


def test_window_files(pkg):
    meta = pd.read_csv(pkg / "meta.csv")
    for row in meta.itertuples():
        if row.group == "tusz_seiz":
            z = np.load(pkg / "seizure" / f"{row.pid}.npz")
            assert set(z.files) == {"x", "y", "start_s"}
            assert z["x"].dtype == np.float16 and z["x"].shape == (row.n_windows, 19, 512)
            assert set(np.unique(z["y"])) <= {0, 1} and z["y"].sum() > 0
            assert z["start_s"].dtype == np.float32 and z["start_s"].shape == (row.n_windows,)
            continue
        z10 = np.load(pkg / "windows10" / row.group / f"{row.pid}.npz")
        assert set(z10.files) == {"x", "x_ms", "start_s"}
        for k in ("x", "x_ms"):
            assert z10[k].dtype == np.float16 and z10[k].shape == (row.n_windows, 10, 512), (row.pid, k)
            assert np.isfinite(z10[k].astype(np.float32)).all()
        assert z10["start_s"].dtype == np.float32 and z10["start_s"].shape == (row.n_windows,)
        p19 = pkg / "windows19" / row.group / f"{row.pid}.npz"
        if row.group in fd.EXTERNAL:
            assert not p19.exists()
            continue
        z19 = np.load(p19)
        assert set(z19.files) == {"x", "x_ms", "start_s"}
        for k in ("x", "x_ms"):
            assert z19[k].dtype == np.float16 and z19[k].shape == (row.n_windows, 19, 512), (row.pid, k)
            assert np.array_equal(fd.ref10(z19[k]), z10[k])  # 10-channel copy: same windows, own average ref
        assert np.array_equal(z19["start_s"], z10["start_s"])


def test_manifest(pkg):
    lines = (pkg / "MANIFEST.sha256").read_text().splitlines()
    listed = {}
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        listed[rel] = digest
    on_disk = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"}
    assert set(listed) == on_disk
    for rel, digest in listed.items():
        assert hashlib.sha256((pkg / rel).read_bytes()).hexdigest() == digest, rel


def test_package_info(pkg):
    info = json.loads((pkg / "PACKAGE_INFO.json").read_text())
    meta = pd.read_csv(pkg / "meta.csv")
    assert info["fake"] is True
    assert info["counts_per_group"] == meta.group.value_counts().to_dict()
    pp = info["preprocessing"]
    assert pp["sfreq"] == 128 and pp["window_samples"] == 512 and pp["dtype"] == "float16"
    assert pp["channels19"] == fd.CH19 and pp["channels10"] == fd.CH10


def test_same_seed_same_package(pkg, tmp_path):
    other = fd.write_package(tmp_path / "again", seed=0)
    def body(p):  # PACKAGE_INFO carries the date, so compare everything else
        return [l for l in (p / "MANIFEST.sha256").read_text().splitlines() if not l.endswith("PACKAGE_INFO.json")]
    assert body(pkg) == body(other)


def test_refuses_non_empty_folder(tmp_path):
    (tmp_path / "x.txt").write_text("keep me")
    with pytest.raises(SystemExit):
        fd.write_package(tmp_path, seed=0)
    assert (tmp_path / "x.txt").read_text() == "keep me"
