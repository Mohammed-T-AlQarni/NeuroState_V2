"""package.py: manifest verify on the fake package; a small end-to-end build on synthetic EDF files."""
import json

import numpy as np
import pandas as pd
import pytest

from _fakes import edf_name, fake_pid, write_csv_bi, write_emotiv_csv, write_tuh_edf
from experiments.epilepsy import fake_data as fd
from experiments.epilepsy import labels as L
from experiments.epilepsy import package as PK
from experiments.epilepsy import preprocess as P


@pytest.fixture
def fake_pkg(tmp_path):
    return fd.write_package(tmp_path / "data", seed=0)


def test_verify_ok(fake_pkg):
    assert PK.verify(fake_pkg) == []


def test_verify_one_changed_byte(fake_pkg):
    p = next((fake_pkg / "windows19" / "tuep_epi").glob("*.npz"))
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    probs = PK.verify(fake_pkg)
    assert len(probs) == 1 and probs[0].startswith("checksum differs")


def test_verify_missing_and_extra(fake_pkg):
    (fake_pkg / "splits" / "t2.csv").unlink()
    (fake_pkg / "extra.txt").write_text("x")
    probs = PK.verify(fake_pkg)
    assert any(p.startswith("missing: splits/t2.csv") for p in probs)
    assert any(p.startswith("not in manifest: extra.txt") for p in probs)


def test_verify_no_manifest(tmp_path):
    assert PK.verify(tmp_path) == ["MANIFEST.sha256 missing"]


def _small_cohort(root):
    """10 TUEP patients (5 + 5), 1 TUSZ Y with a seizure, 1 TUAB normal train, 1 eval, 1 stage 1 only, 1 external."""
    rows, recs = [], []

    def add(i, group, label, tusz_split="", seconds=150, seizure=None, has_tuab=False):
        pid = fake_pid(i)
        d = root / group / pid
        d.mkdir(parents=True)
        edf = write_tuh_edf(d / edf_name(pid), seconds=seconds, order_seed=i)
        if seizure:
            write_csv_bi(edf, [seizure])
        rows.append(dict(pid=pid, group=group, label=label, source=L.GROUP_SOURCE[group], tusz_split=tusz_split,
                         tuab_split="", has_tuab_normal=has_tuab, n_recordings=1))
        if group != "tusz_seiz":
            recs.append(dict(pid=pid, group=group, role="diag", split="", ses="s001", tok="t000", year=2010,
                             montage="02_tcp_le", path=str(edf)))
        if tusz_split:
            recs.append(dict(pid=pid, group=group, role="seizure", split=tusz_split, ses="s001", tok="t000",
                             year=2010, montage="02_tcp_le", path=str(edf)))
        return pid

    for i in range(5):
        add(i, "tuep_epi", "epilepsy")
        add(10 + i, "tuep_ctrl", "no_epilepsy")
    add(20, "tusz_y", "epilepsy", tusz_split="train", seconds=300, seizure=(100, 110, "seiz"))
    add(21, "tuab_train", "normal_eeg")
    add(22, "tuab_eval", "normal_eeg")
    add(23, "tusz_seiz", None, tusz_split="dev", seconds=120, seizure=(20, 30, "seiz"))
    ext = write_emotiv_csv(root / "signal-1.csv.gz", seconds=90)
    rows.append(dict(pid="ng0001", group="nigeria", label="epilepsy", source="Nigeria", tusz_split="", tuab_split="",
                     has_tuab_normal=False, n_recordings=1))
    recs.append(dict(pid="ng0001", group="nigeria", role="diag", split="", ses="s001", tok="t000", year=None,
                     montage="emotiv_14", path=str(ext)))
    return pd.DataFrame(rows, columns=L.COHORT_COLS), pd.DataFrame(recs, columns=L.REC_COLS)


def test_end_to_end_build(tmp_path):
    cohort, recs = _small_cohort(tmp_path / "raw")
    out = tmp_path / "pkg"
    st = P.Settings(minutes=1, missing="drop")
    info = PK.run(tmp_path, out, tmp_path / "work", st, workers=1, cohort=cohort, recs=recs, read_headers=True)
    assert PK.verify(out) == []
    lab = pd.read_csv(out / "labels.csv")
    assert list(lab.columns) == L.LABELS_COLS and "tusz_seiz" not in set(lab.group)
    meta = pd.read_csv(out / "meta.csv")
    assert list(meta.columns) == L.META_COLS
    diag = meta[meta.group != "tusz_seiz"]
    assert (diag.n_windows == 15).all()                         # 1 minute = 15 windows for every patient
    for row in diag.itertuples():
        z10 = np.load(out / "windows10" / row.group / f"{row.pid}.npz")
        assert z10["x"].shape == (15, 10, 512) and z10["x"].dtype == np.float16
        if row.group == "nigeria":
            assert not (out / "windows19" / row.group / f"{row.pid}.npz").exists()
            continue
        z = np.load(out / "windows19" / row.group / f"{row.pid}.npz")
        assert set(z.files) == {"x", "x_ms", "start_s"} and z["x"].shape == (15, 19, 512)
    # TUSZ Y patient: windows around the seizure (40-170 s) removed before the cap
    p = cohort.pid[cohort.group == "tusz_y"].iloc[0]
    s = np.load(out / "windows19" / "tusz_y" / f"{p}.npz")["start_s"]
    assert not ((s > 36) & (s < 170)).any()
    # stage 1 windows for both TUSZ patients, seizure labels present
    for p in cohort.pid[cohort.tusz_split != ""]:
        z = np.load(out / "seizure" / f"{p}.npz")
        assert set(z.files) == {"x", "y", "start_s"} and z["y"].dtype == np.int8 and z["y"].sum() > 0
    pinfo = json.loads((out / "PACKAGE_INFO.json").read_text())
    assert pinfo["preprocessing"]["minutes"] == 1 and pinfo["preprocessing"]["missing"] == "drop"
    assert pinfo["stats"]["windows"]["tusz_y"]["windows_seizure"] > 0
    assert (out / "splits" / "t1_r2_f4_val.csv").exists()
    assert (tmp_path / "work" / "recordings.csv").exists()     # local working file, not in the package
    assert not (out / "recordings.csv").exists()


def test_build_refuses_non_empty(tmp_path):
    cohort, recs = _small_cohort(tmp_path / "raw")
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        PK.run(tmp_path, out, tmp_path / "work", P.Settings(minutes=1, missing="drop"), workers=1,
               cohort=cohort, recs=recs)


def test_guard_refuses_raw_data(tmp_path):
    root = tmp_path / "NeuroState"
    (root / "Claude_Workspace").mkdir(parents=True)
    with pytest.raises(SystemExit):
        L._guard(root / "TUH" / "out", root)
    L._guard(root / "Claude_Workspace" / "epilepsy" / "data", root)   # allowed
