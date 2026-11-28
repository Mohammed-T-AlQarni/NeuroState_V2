"""preprocess.py and seizure_cut.py on synthetic EDF / Emotiv files (no real data)."""
import numpy as np
import pytest

from _fakes import (dominant_freq, edf_name, fake_pid, write_csv_bi, write_emotiv_csv, write_tuh_edf)
from experiments.epilepsy import preprocess as P
from experiments.epilepsy import seizure_cut as SC


def test_norm_channel():
    assert P.norm_channel("EEG FP1-REF") == "FP1"
    assert P.norm_channel("EEG T3-LE") == "T3"
    assert P.norm_channel("EEG T7-REF") == "T3"
    assert P.norm_channel("EEG P8-LE") == "T6"
    assert P.norm_channel("EEG EKG1-REF") == "EKG1"
    assert P.norm_channel(" eeg fz-avg ") == "FZ"


def test_map_channels_le_and_extra():
    names = [f"EEG {c}-LE" for c in reversed(P.CH19)] + ["EEG EKG1-LE", "EEG T1-LE", "PHOTIC-REF"]
    mapping, missing = P.map_channels(names)
    assert missing == [] and set(mapping) == set(P.CH19)
    assert mapping["FP1"] == "EEG FP1-LE"
    mapping, missing = P.map_channels([n for n in names if "CZ" not in n])
    assert missing == ["CZ"]


@pytest.fixture
def settings():
    return P.Settings(minutes=2)          # team defaults otherwise: interpolate 1-2, 300 uV, photic removal


@pytest.mark.parametrize("suffix", ["-LE", "-REF"])
def test_windows_shapes_order_and_cap(tmp_path, settings, suffix):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(1)), seconds=600, suffix=suffix)
    res = P.process_patient([{"path": str(edf)}], settings)
    assert res.n_windows == 30                                   # 2 minutes = 30 windows of 4 s
    assert res.x.shape == res.x_ms.shape == (30, 19, 512) and res.x.dtype == np.float16
    assert res.x10.shape == res.x10_ms.shape == (30, 10, 512) and res.x10.dtype == np.float16
    assert res.start_s.dtype == np.float32 and np.allclose(res.start_s, np.arange(30) * 4)
    # channel k carries 2 + k Hz in CH19 order: the output must be in the fixed order
    assert np.allclose(dominant_freq(res.x), 2 + np.arange(19), atol=0.3)
    assert np.allclose(dominant_freq(res.x10), [2 + P.CH19.index(c) for c in P.CH10], atol=0.3)
    # average reference: 19 channels and the 10-channel copy each sum to ~0
    assert np.abs(res.x.astype(np.float32).mean(axis=1)).max() < 0.05
    assert np.abs(res.x10.astype(np.float32).mean(axis=1)).max() < 0.05
    assert res.sfreq_orig == 256
    assert res.stats["recordings_used"] == 1


def test_bands_differ_but_aligned(tmp_path, settings):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(2)), seconds=300)
    res = P.process_patient([{"path": str(edf)}], settings)
    # electrode O2 (k = 18) carries 20 Hz: inside 1-40, at the edge of 2-20 (attenuated)
    p_x = np.abs(res.x[:, 18].astype(np.float32)).mean()
    p_ms = np.abs(res.x_ms[:, 18].astype(np.float32)).mean()
    assert p_ms < p_x
    assert res.x.shape == res.x_ms.shape


def test_amplitude_rejection(tmp_path, settings):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(3)), seconds=600, spike_at=(30.0, 0))
    res = P.process_patient([{"path": str(edf)}], settings)
    assert res.n_windows == 30
    assert 28.0 not in set(res.start_s.tolist())                 # the window with the 500 uV artefact
    assert res.stats["windows_bad_amplitude"] >= 1


def test_amplitude_limit_300(tmp_path, settings):
    # diagnosis windows: a 200 uV burst is kept, a 400 uV burst is dropped (limit 300 uV peak-to-peak)
    e1 = write_tuh_edf(tmp_path / edf_name(fake_pid(14)), seconds=300, spike_at=(30.0, 0), spike_uv=200)
    e2 = write_tuh_edf(tmp_path / edf_name(fake_pid(15)), seconds=300, spike_at=(30.0, 0), spike_uv=400)
    r1 = P.process_patient([{"path": str(e1)}], settings)
    r2 = P.process_patient([{"path": str(e2)}], settings)
    assert 28.0 in set(r1.start_s.tolist()) and r1.stats["windows_bad_amplitude"] == 0
    assert 28.0 not in set(r2.start_s.tolist()) and r2.stats["windows_bad_amplitude"] >= 1


def test_whole_recording_flat_electrode_is_interpolated(tmp_path, settings):
    # a flat F4 is not flat after the average reference; it is found before it and treated as missing
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(11)), seconds=300, flat=("F4",))
    res = P.process_patient([{"path": str(edf)}], settings)
    assert res.n_windows == 30 and res.stats["electrodes_interpolated"] == 1
    assert res.stats["recordings_interpolated"] == 1 and res.stats["windows_flat"] == 0
    # a flat electrode with missing="drop" drops the recording
    r2 = P.process_patient([{"path": str(edf)}], P.Settings(minutes=2, missing="drop"))
    assert r2.n_windows == 0 and r2.stats["skip_missing_electrode"] == 1


def test_three_missing_or_flat_electrodes_drop_the_recording(tmp_path, settings):
    e1 = write_tuh_edf(tmp_path / edf_name(fake_pid(16)), seconds=300, flat=("F4", "C3", "O1"))
    e2 = write_tuh_edf(tmp_path / edf_name(fake_pid(17)), seconds=300, drop=("CZ",), flat=("F4", "O1"))
    for e in (e1, e2):
        res = P.process_patient([{"path": str(e)}], settings)
        assert res.n_windows == 0 and res.stats["skip_too_many_missing"] == 1
    e3 = write_tuh_edf(tmp_path / edf_name(fake_pid(18)), seconds=300, drop=("CZ",), flat=("F4",))
    res = P.process_patient([{"path": str(e3)}], P.Settings(minutes=2, ptp_uv=1000))
    assert res.n_windows == 30 and res.stats["electrodes_interpolated"] == 2


def test_photic_windows_dropped_only_when_active(tmp_path):
    st = P.Settings(minutes=10)                    # look at every window of the 5-minute file
    stim = write_tuh_edf(tmp_path / edf_name(fake_pid(19)), seconds=300, photic=(100, 130))
    idle = write_tuh_edf(tmp_path / edf_name(fake_pid(20)), seconds=300, photic="idle")
    none = write_tuh_edf(tmp_path / edf_name(fake_pid(21)), seconds=300)
    r = P.process_patient([{"path": str(stim)}], st)
    s = r.start_s
    assert not ((s > 86) & (s < 140)).any()       # stimulation 100-130 s, +/- 10 s
    assert r.stats["windows_photic"] == 13 and r.stats["recordings_with_stimulation"] == 1
    r = P.process_patient([{"path": str(idle)}], st)
    assert r.stats["windows_photic"] == 0 and r.stats["recordings_with_photic_channel"] == 1
    assert r.stats["recordings_with_stimulation"] == 0 and r.n_windows == 75
    r = P.process_patient([{"path": str(none)}], st)
    assert r.stats["windows_photic"] == 0 and r.stats["recordings_with_photic_channel"] == 0
    r = P.process_patient([{"path": str(stim)}], P.Settings(minutes=10, photic=False))
    assert r.stats["windows_photic"] == 0 and r.n_windows == 75


def test_zero_padded_start_dropped(tmp_path, settings):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(12)), seconds=300, zero_start_s=6)
    res = P.process_patient([{"path": str(edf)}], settings)
    assert res.n_windows == 30
    assert res.start_s[0] >= 8.0                                 # windows 0-4 s and 4-8 s hold flat pieces
    assert res.stats["windows_flat"] == 2


def test_common_signal_stretch_dropped(tmp_path, settings):
    # not flat before the reference, but (almost) nothing left after it
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(13)), seconds=300, common_start_s=10)
    res = P.process_patient([{"path": str(edf)}], settings)
    assert res.n_windows == 30 and res.start_s[0] >= 8.0
    assert res.stats["windows_flat"] >= 2


def test_short_recording_skipped(tmp_path, settings):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(4)), seconds=50)
    res = P.process_patient([{"path": str(edf)}], settings)
    assert res.n_windows == 0 and res.stats["skip_under_1_min"] == 1


def test_missing_electrode_drop_or_interpolate(tmp_path):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(5)), seconds=300, drop=("CZ",))
    r1 = P.process_patient([{"path": str(edf)}], P.Settings(minutes=2, missing="drop"))
    assert r1.n_windows == 0 and r1.stats["skip_missing_electrode"] == 1
    # the synthetic electrodes are unrelated sines, so the interpolated one sums to ~165 uV: loosen the limit here
    r2 = P.process_patient([{"path": str(edf)}], P.Settings(minutes=2, missing="interpolate", ptp_uv=1000))
    assert r2.n_windows == 30 and r2.x.shape[1] == 19 and r2.stats["recordings_interpolated"] == 1
    assert np.isfinite(r2.x.astype(np.float32)).all()


def test_minutes_across_recordings(tmp_path):
    a = write_tuh_edf(tmp_path / edf_name(fake_pid(6), 1), seconds=80)     # 20 windows
    b = write_tuh_edf(tmp_path / edf_name(fake_pid(6), 2), seconds=300)
    c = write_tuh_edf(tmp_path / edf_name(fake_pid(6), 3), seconds=300)
    res = P.process_patient([{"path": str(p)} for p in (a, b, c)], P.Settings(minutes=2, missing="drop"))
    assert res.n_windows == 30
    assert res.stats["recordings_used"] == 2 and res.stats["recordings_read"] == 2  # third not opened


def test_settings_defaults_are_the_team_decisions():
    s = P.Settings()
    assert (s.minutes, s.missing, s.max_interp, s.ptp_uv, s.flat_uv, s.photic, s.photic_margin_s) == \
        (5.0, "interpolate", 2, 300.0, 0.5, True, 10.0)
    d = s.as_dict()
    assert d["ptp_uv"] == 300.0 and d["photic_removal"] is True and d["max_interp"] == 2
    with pytest.raises(ValueError):
        P.Settings(minutes=5, missing="maybe")
    with pytest.raises(ValueError):
        P.Settings(minutes=0)


def test_emotiv_csv(tmp_path):
    p = write_emotiv_csv(tmp_path / "signal-1.csv.gz", seconds=120)
    res = P.process_patient([{"path": str(p)}], P.Settings(minutes=1, missing="drop"), kind="emotiv")
    assert res.x is None
    assert res.x10.shape == (15, 10, 512) and res.x10.dtype == np.float16
    assert np.allclose(dominant_freq(res.x10), 2 + np.arange(10), atol=0.3)
    assert np.abs(res.x10.astype(np.float32).mean(axis=1)).max() < 0.05


# ------------------------------------------------------------------------------------------------ seizure cut
def test_read_csv_bi(tmp_path):
    edf = tmp_path / edf_name(fake_pid(7))
    write_csv_bi(edf, [(0, 100, "bckg"), (100, 110, "seiz"), (108, 120, "fnsz"), (200, 300, "bckg")])
    assert SC.intervals_for_edf(edf) == [(100.0, 120.0)]
    assert SC.intervals_for_edf(tmp_path / "none.edf") == []


def test_window_mask_margin():
    start = np.arange(0, 400, 4.0)
    m = SC.window_mask(start, [(100.0, 110.0)], margin_s=60)
    removed = start[m]
    assert removed.min() == 40.0 and removed.max() == 168.0
    assert 168.0 in removed and 36.0 not in removed and 172.0 not in removed


def test_seizure_cut_before_minute_cap(tmp_path):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(8)), seconds=600)
    write_csv_bi(edf, [(100, 110, "seiz")])
    st = P.Settings(minutes=5, missing="drop")
    res = P.process_patient([{"path": str(edf), "seizures": SC.intervals_for_edf(edf)}], st)
    assert res.n_windows == 75                                   # still 5 minutes after the cut
    s = res.start_s
    assert not ((s > 36) & (s < 170)).any()
    assert res.stats["windows_seizure"] == 33                    # windows starting 40 ... 168


def test_seizure_windows_labels(tmp_path):
    edf = write_tuh_edf(tmp_path / edf_name(fake_pid(9)), seconds=300)
    write_csv_bi(edf, [(100, 110, "seiz")])
    x, y, start, st = P.seizure_windows([{"path": str(edf), "seizures": SC.intervals_for_edf(edf)}], "drop", 1000.0)
    assert x.shape[1:] == (19, 512) and x.dtype == np.float16 and y.dtype == np.int8
    assert sorted(start[y == 1].tolist()) == [100.0, 104.0, 108.0]   # >= half of the window inside
    assert len(y) == 75 and st["windows_ambiguous"] == 0
