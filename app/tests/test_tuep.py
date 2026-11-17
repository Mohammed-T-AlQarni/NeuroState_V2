"""TUEP inventory, metadata linkage, overlap checks and per-patient preparation, on a SYNTHETIC TUEP-shaped tree."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import tuep_fixture as fx
from conftest import REPO, needs_inputs

pytestmark = needs_inputs  # the adapter imports validate_microstates, which needs the NeuroState-Microstates clone


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("tuep") / "v3.1.0"
    info = fx.build(root)
    return root, info


@pytest.fixture(scope="module")
def inventory(tree, tmp_path_factory):
    from app.tools import tuep_inventory as ti
    root, info = tree
    out = tmp_path_factory.mktemp("inv")
    checks = ti.build(root, out)
    return out, checks, info


def test_paths_and_labels(inventory):
    out, checks, info = inventory
    rec = pd.read_csv(out / "inventory_recordings.csv")
    assert len(rec) == info["files_e"] + info["files_n"]
    assert set(rec.label) == {"epilepsy", "no_epilepsy"} and set(rec.group) == {"E", "N"}
    r = rec.iloc[0]
    assert r.patient_id == fx.pid(0) and r.session_id == "s001" and r.session_year == 2010 and r.token == "t000"
    assert checks["observed"]["epilepsy"] == {"patients": 10, "sessions": info["sessions_e"], "files": info["files_e"]}
    assert checks["counts_match_readme"] is False  # the synthetic tree is smaller than the real release


def test_tuh_channel_names_and_le_reference_are_compatible(inventory):
    out, _, _ = inventory
    rec = pd.read_csv(out / "inventory_recordings.csv")
    assert rec.compatible.all() and (rec.standard_found == 19).all()
    assert set(rec.loc[rec.patient_id == fx.pid(1), "montage_config"]) == {"02_tcp_le"}


def test_seizures_tusz_flags_and_hashes(inventory):
    out, checks, _ = inventory
    rec = pd.read_csv(out / "inventory_recordings.csv")
    sz = rec[rec.n_seizures > 0]
    assert len(sz) == 1 and sz.iloc[0].seizure_intervals == "10.000-18.000" and sz.iloc[0].seizure_s == 8.0
    assert set(rec.in_tusz) == {"yes", "no"}
    assert set(rec.loc[rec.patient_id == fx.pid(0), "in_tusz"]) == {"yes"}
    assert rec.sha256.str.len().eq(64).all() and rec.data_sha256.str.len().eq(64).all()
    assert checks["duplicate_files"] == [] and checks["label_conflicts"] == []


def test_metadata_linkage(inventory):
    out, checks, info = inventory
    m = checks["metadata"]
    assert m["found"] and m["sheet"] == "sessions" and m["key_mode"] == "separate"
    assert m["key_columns"] == ["Subject ID", "Session"]
    assert m["sessions_total"] == info["sessions_e"] + info["sessions_n"]
    assert m["sessions_linked"] == m["sessions_total"] - 1 and m["sessions_without_metadata"] == 1
    assert m["metadata_rows_without_eeg"] == 1
    ses = pd.read_csv(out / "inventory_sessions.csv")
    row = ses[(ses.patient_id == fx.pid(0)) & (ses.session_id == "s001")].iloc[0]
    assert row.meta_Diagnosis == "epilepsy" and row.meta_Medications == "levetiracetam"
    assert set(m["review_by_class"]["meta_Diagnosis"]) == {"epilepsy", "no_epilepsy"}


def test_overlap_checks_find_label_conflicts_and_duplicates(tree, tmp_path):
    from app.tools import tuep_inventory as ti
    root, _ = tree
    bad = tmp_path / "bad"
    shutil.copytree(root, bad)
    src = next((bad / "00_epilepsy" / fx.pid(2)).rglob("*.edf"))
    # the same patient also under the other class folder, and a copy of one file inside another patient
    dst = bad / "01_no_epilepsy" / fx.pid(2) / "s009_2020" / "01_tcp_ar" / f"{fx.pid(2)}_s009_t000.edf"
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    dst2 = bad / "01_no_epilepsy" / fx.pid(15) / "s001_2010" / "01_tcp_ar" / f"{fx.pid(15)}_s001_t005.edf"
    b = bytearray(src.read_bytes())
    b[8:16] = b"EDITED  "  # same signal, different header
    dst2.write_bytes(bytes(b))
    ids = tmp_path / "tusz_ids.txt"
    ids.write_text(f"{fx.pid(3)}\n{fx.pid(12)}\n")
    checks = ti.build(bad, tmp_path / "inv", overlaps={"tusz": ids})
    assert checks["label_conflicts"] == [fx.pid(2)]
    assert any(d["cross_patient"] is False and len(d["files"]) == 2 for d in checks["duplicate_files"])
    assert any(d["cross_patient"] and fx.pid(15) in d["patients"] for d in checks["duplicate_signals_different_header"])
    pat = pd.read_csv(tmp_path / "inv" / "inventory_patients.csv")
    assert set(pat.loc[pat.overlap_tusz, "patient_id"]) == {fx.pid(3), fx.pid(12)}


def test_prepare_one_file_per_patient_windows_and_seizure_cut(tree, inventory, tmp_path):
    from app.tools import tuep_prepare as tp
    root, info = tree
    inv, _, _ = inventory
    prep_dir = tmp_path / "prepared"
    tp.main(["--root", str(root), "--inventory", str(inv), "--out", str(prep_dir), "--max-minutes-per-patient", "1",
             "--seizure-margin-s", "5"])
    subs = pd.read_csv(prep_dir / "subjects.csv")
    assert len(subs) == info["patients"] and set(subs.Group) == {"E", "N"}
    for s in subs.participant_id:
        z = np.load(prep_dir / f"{s}.npz")
        assert z["dl"].shape == z["ms"].shape and z["ms"].shape[0] == 19 and z["ms"].shape[1] % 512 == 0
        assert z["ms"].shape[1] <= 60 * 128  # the 1-minute cap
    st = pd.read_csv(prep_dir / "status.csv")
    first = st[st.recording_path.str.endswith(f"{fx.pid(0)}_s001_t000.edf")].iloc[0]
    # 40 s file, seizure 10-18 s, margin 5 s: pieces 0-5 s (640 -> 512 samples) and 23-40 s (2176 -> 2048 samples)
    assert first.status == "prepared" and first.prepared_s == (512 + 2048) / 128
    assert (st.status == "not_needed").sum() > 0  # epilepsy patients have more than the cap
    import run_experiment as rx
    ids, y, data = rx.load(str(prep_dir), ["E", "N"])
    assert len(ids) == info["patients"] and sorted(set(y.tolist())) == [0, 1]


@pytest.mark.slow
def test_run_experiment_unchanged_splits_by_patient_and_converter_reads_inventory(tree, inventory, tmp_path):
    from app.tools import tuep_prepare as tp
    root, info = tree
    inv, _, _ = inventory
    prep_dir, res = tmp_path / "prepared", tmp_path / "results"
    tp.main(["--root", str(root), "--inventory", str(inv), "--out", str(prep_dir), "--max-minutes-per-patient", "1"])
    r = subprocess.run([sys.executable, "run_experiment.py", "--prep", str(prep_dir), "--out", str(res), "--groups", "E", "N",
                        "--models", "baseline", "seqcnn", "--ms-input", "onehot", "--skip-leak", "--folds", "5", "--epochs", "1",
                        "--crops", "4", "--batch", "8"], cwd=REPO / "experiments", capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    oof = pd.read_csv(res / "oof_predictions.csv")
    for model, g in oof.groupby("model"):
        assert sorted(g.subject) == sorted(f"sub-{fx.pid(i)}" for i in range(info["patients"])), model  # each patient once
    from app.tools import build_study_results as bsr
    tp_inv = tmp_path / "inv2"
    shutil.copytree(inv, tp_inv)
    from app.tools import tuep_inventory as ti
    ti.build(root, tp_inv, do_hash=False, status_csv=prep_dir / "status.csv")
    out = tmp_path / "study" / "epilepsy" / "results.json"
    bsr.main(["--run", str(res), "--study", "epilepsy", "--out", str(out), "--labels", "epilepsy", "no epilepsy",
              "--inventory", str(tp_inv)])
    d = json.loads(out.read_text())
    assert d["cohort"]["recordings_ingested"] == info["files_e"] + info["files_n"]
    rec = pd.read_csv(tp_inv / "inventory_recordings.csv")
    assert d["cohort"]["recordings_processed"] == int((rec.status == "prepared").sum()) > 0
    assert d["cohort"]["unique_patients_training"] == info["patients"]
