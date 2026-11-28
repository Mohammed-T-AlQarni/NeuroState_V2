"""Preprocessing for every recording of the NeuroState epilepsy study (settings: ops repo shared/PLAN.md).

One code path for all TUH recordings:
  1. map channel names (strip "EEG ", "-REF", "-LE", ...; T7/T8/P7/P8 -> T3/T4/T5/T6), keep the 19 standard
     electrodes in the fixed order, drop every other channel;
  2. an electrode flat for (almost) the whole recording counts as missing; 1-2 missing electrodes are interpolated
     (spherical spline, MNE interpolate_bads, before windowing), 3 or more drop the recording (team decision
     28 Nov; `missing="drop"` drops any recording with a missing electrode);
  3. average reference, resample to 128 Hz;
  4. two bands from the same cleaned signal (as experiments/prep.py): x = 1-40 Hz (deep models), x_ms = 2-20 Hz
     (microstates);
  5. 4-s windows (19 x 512), no overlap; drop windows with a channel peak-to-peak above `ptp_uv` (300 uV, after
     filtering) or a flat channel (SD below `flat_uv` = 0.5 uV in any 1-s piece), checked both BEFORE the average
     reference (which would hide a flat electrode) and after it on the filtered signal (which catches stretches
     where every electrode carries the same signal);
  6. drop windows overlapping a seizure +/- 60 s (seizure_cut.py) and, where the file has a photic channel, every
     window during photic stimulation +/- 10 s;
  7. keep the first N clean minutes per patient, going through the recordings in session order from the start of
     each recording; recordings under 1 minute are skipped;
  8. a 10-channel copy (F7 F3 F4 F8 T3 T4 T5 T6 O1 O2) of the same windows, re-referenced to the average of those 10
     channels so that it matches the Nigeria / Guinea-Bissau data, which only have these 10 of the 19;
  9. stored as float16 microvolts, with start_s (seconds from the start of the recording) per window.

Nigeria / Guinea-Bissau: `clean_emotiv_csv()` reads the 14-channel Emotiv CSV, keeps the 10 shared channels,
average reference over those 10, same bands, windows and amplitude rule.

Nothing here writes to the raw data; the caller (package.py) writes the output files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from experiments.epilepsy import seizure_cut

SFREQ = 128
WIN = 512                      # 4 s at 128 Hz
WIN_S = WIN / SFREQ
CH19 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
CH10 = ["F7", "F3", "F4", "F8", "T3", "T4", "T5", "T6", "O1", "O2"]
IDX10 = [CH19.index(c) for c in CH10]
BAND_X, BAND_MS = (1.0, 40.0), (2.0, 20.0)
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
DEFAULT_PTP_UV = 300.0         # team decision 28 Nov (diagnosis windows); stage 1 uses 1,000 uV (package.py)
DEFAULT_FLAT_UV = 0.5          # experiments/tuep_curve/prep_clean.py (24 Nov): channel SD < 0.5 uV = flat
DEFAULT_MAX_INTERP = 2         # team decision 28 Nov: 1-2 missing / flat electrodes interpolated, 3+ dropped
WHOLE_FLAT_SHARE = 0.95        # an electrode flat in >= 95% of its 1-s pieces counts as flat for the whole recording
DEFAULT_PHOTIC_MARGIN_S = 10.0 # team decision 28 Nov: photic stimulation +/- 10 s removed
MIN_REC_S = 60.0               # recordings under 1 minute are dropped
EMOTIV_SFREQ = 128.0

_PREFIX = re.compile(r"^(EEG|POL)\s+")
_SUFFIX = re.compile(r"-(REF|LE|AVG|AR)$")


def norm_channel(name: str) -> str:
    """'EEG FP1-REF' -> 'FP1', 'EEG T7-LE' -> 'T3', 'EEG EKG1-REF' -> 'EKG1'."""
    s = _SUFFIX.sub("", _PREFIX.sub("", name.strip().upper())).strip()
    return ALIAS.get(s, s)


def map_channels(ch_names) -> tuple[dict, list]:
    """-> ({standard name: original name}, [missing standard names]). First match wins."""
    mapping = {}
    for ch in ch_names:
        n = norm_channel(ch)
        if n in CH19 and n not in mapping:
            mapping[n] = ch
    return mapping, [c for c in CH19 if c not in mapping]


@dataclass
class Settings:
    minutes: float = 5.0             # clean minutes per patient (team decision 28 Nov)
    missing: str = "interpolate"     # "interpolate" (up to max_interp, else drop) | "drop" (team decision: interpolate)
    ptp_uv: float = DEFAULT_PTP_UV
    flat_uv: float = DEFAULT_FLAT_UV
    seizure_margin_s: float = seizure_cut.MARGIN_S
    max_load_min: float | None = None   # read at most this much of each recording; default 3 x minutes + 2
    max_interp: int = DEFAULT_MAX_INTERP
    photic: bool = True              # remove photic-stimulation windows where a photic channel exists
    photic_margin_s: float = DEFAULT_PHOTIC_MARGIN_S

    def __post_init__(self):
        if self.missing not in ("drop", "interpolate"):
            raise ValueError("missing must be 'drop' or 'interpolate'")
        if not self.minutes or self.minutes <= 0:
            raise ValueError("minutes must be > 0")
        if self.max_load_min is None:
            self.max_load_min = 3 * self.minutes + 2

    def init_args(self) -> dict:
        """Constructor arguments (a picklable dict for worker processes)."""
        return dict(minutes=self.minutes, missing=self.missing, ptp_uv=self.ptp_uv, flat_uv=self.flat_uv,
                    seizure_margin_s=self.seizure_margin_s, max_load_min=self.max_load_min,
                    max_interp=self.max_interp, photic=self.photic, photic_margin_s=self.photic_margin_s)

    @property
    def n_windows(self) -> int:
        return int(round(self.minutes * 60 / WIN_S))

    def as_dict(self):
        return dict(minutes=self.minutes, missing=self.missing, max_interp=self.max_interp,
                    whole_recording_flat_share=WHOLE_FLAT_SHARE, ptp_uv=self.ptp_uv, flat_uv=self.flat_uv,
                    flat_piece_s=1.0, photic_removal=self.photic, photic_margin_s=self.photic_margin_s,
                    seizure_margin_s=self.seizure_margin_s, max_load_min=self.max_load_min, sfreq=SFREQ,
                    window_s=WIN_S, window_samples=WIN, overlap=0, reference="average",
                    band_x_hz=list(BAND_X), band_x_ms_hz=list(BAND_MS), channels19=CH19, channels10=CH10,
                    channels10_reference="average of the 10 channels", dtype="float16", units="uV",
                    min_recording_s=MIN_REC_S)


class Skip(Exception):
    """A recording that is not used; the message is the reason (aggregated in the stats)."""


def load_edf(path, max_seconds=None):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    dur = raw.n_times / raw.info["sfreq"]
    if dur < MIN_REC_S:
        raise Skip("under_1_min")
    if max_seconds and dur > max_seconds:
        raw.crop(tmax=max_seconds, include_tmax=False)
    return raw


def whole_flat_channels(data, sfreq, flat_uv=DEFAULT_FLAT_UV, share=WHOLE_FLAT_SHARE):
    """[C, n] -> indices of channels whose SD is below flat_uv in at least `share` of their 1-s pieces."""
    seg = int(round(sfreq))
    k = data.shape[1] // seg
    if k == 0:
        return []
    sd = data[:, :k * seg].reshape(data.shape[0], k, seg).std(axis=2)
    return [i for i in range(data.shape[0]) if (sd[i] < flat_uv).mean() >= share]


def clean_raw(raw, missing: str = "interpolate", flat_uv: float = DEFAULT_FLAT_UV, max_interp: int = DEFAULT_MAX_INTERP):
    """mne Raw -> (x, x_ms, flat, n_interp): x / x_ms float64 [19, n] microvolts at 128 Hz, average reference,
    1-40 / 2-20 Hz; flat: bool per 4-s window, True when a channel was flat before re-referencing (flat_windows);
    n_interp: electrodes interpolated (missing from the file + flat for the whole loaded recording)."""
    import mne
    mapping, miss = map_channels(raw.ch_names)
    if len(mapping) < 3:
        raise Skip("too_few_electrodes")
    raw = raw.copy().pick([mapping[c] for c in CH19 if c in mapping])
    raw.load_data(verbose="ERROR")
    raw.rename_channels({mapping[c]: c for c in CH19 if c in mapping})
    raw.set_channel_types({c: "eeg" for c in raw.ch_names}, verbose="ERROR")
    dead = [raw.ch_names[i] for i in whole_flat_channels(raw.get_data() * 1e6, raw.info["sfreq"], flat_uv)]
    bad = list(miss) + dead
    if bad and missing == "drop":
        raise Skip("missing_electrode")
    if len(bad) > max_interp:
        raise Skip("too_many_missing")
    if bad:  # interpolate missing and whole-recording-flat electrodes
        if miss:
            add = mne.io.RawArray(np.zeros((len(miss), raw.n_times)),
                                  mne.create_info(miss, raw.info["sfreq"], "eeg"), verbose="ERROR")
            raw.add_channels([add], force_update_info=True)
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose="ERROR")
        raw.info["bads"] = bad
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.reorder_channels(CH19)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose="ERROR")
    data = raw.get_data() * 1e6
    flat = flat_windows(data, flat_uv)                 # before the average reference, which hides flat channels
    data = data - data.mean(axis=0, keepdims=True)     # average reference
    x, x_ms = _bands(data, SFREQ)
    return x, x_ms, flat, len(bad)


def photic_intervals(raw, bin_s=1.0) -> tuple[bool, list[tuple[float, float]]]:
    """(has a photic channel, [(start, stop) seconds of stimulation]) from the file's PHOTIC channel.
    Rule (as the 24 Nov cleaning code): 1-s bins of the photic channel's peak-to-peak; stimulation where a bin
    exceeds 30% of the 99.5th percentile, provided that percentile is more than 5x the median bin (a channel that is
    only noise or always flat shows no stimulation)."""
    ch = next((c for c in raw.ch_names if "PHOTIC" in norm_channel(c)), None)
    if ch is None:
        return False, []
    sf = raw.info["sfreq"]
    ph = raw.copy().pick([ch]).load_data(verbose="ERROR").get_data()[0]
    seg = int(round(bin_s * sf))
    k = len(ph) // seg
    if k == 0:
        return True, []
    r = np.ptp(ph[:k * seg].reshape(k, seg), axis=1)
    hi = np.percentile(r, 99.5)
    if not hi > 5 * (np.median(r) + 1e-12):
        return True, []
    on = np.flatnonzero(r > 0.3 * hi)
    return True, seizure_cut.merge([(i * bin_s, (i + 1) * bin_s) for i in on])


def flat_windows(data, flat_uv=DEFAULT_FLAT_UV):
    """[C, n] signal before re-referencing -> bool [n_windows]: True when any channel has a 1-s piece with
    SD below flat_uv (flat or disconnected electrode, zero-padded start or end of a file)."""
    return flat_pieces(to_windows(data), flat_uv)


def flat_pieces(w, flat_uv=DEFAULT_FLAT_UV, seg_s=1.0):
    """[n, C, 512] -> bool [n]: any channel with a 1-s piece whose SD is below flat_uv. Used twice: before the
    average reference (a flat electrode) and after it on the filtered signal (all electrodes carrying the same
    signal, which the average reference cancels to almost nothing)."""
    seg = int(seg_s * SFREQ)
    if len(w) == 0:
        return np.zeros(0, bool)
    d = w.reshape(w.shape[0], w.shape[1], WIN // seg, seg)
    return (d.std(axis=3) < flat_uv).any(axis=(1, 2))


def _bands(data, sfreq):
    import mne
    x = mne.filter.filter_data(data, sfreq, *BAND_X, method="fir", phase="zero", verbose="ERROR")
    x_ms = mne.filter.filter_data(data, sfreq, *BAND_MS, method="fir", phase="zero", verbose="ERROR")
    return x, x_ms


def clean_emotiv_csv(path, sfreq=EMOTIV_SFREQ, max_seconds=None, flat_uv=DEFAULT_FLAT_UV):
    """Nigeria / Guinea-Bissau Emotiv CSV (.csv.gz) -> (x, x_ms, flat): float64 [10, n] microvolts at 128 Hz."""
    import pandas as pd
    df = pd.read_csv(path, compression="infer", nrows=int(max_seconds * sfreq) if max_seconds else None)
    cols = {}
    for c in df.columns:
        n = norm_channel(str(c).strip().strip('"'))
        if n in CH10 and n not in cols:
            cols[n] = c
    miss = [c for c in CH10 if c not in cols]
    if miss:
        raise Skip("missing_electrode")
    data = df[[cols[c] for c in CH10]].apply(pd.to_numeric, errors="coerce").to_numpy(float).T
    if np.isnan(data).any():
        data = pd.DataFrame(data.T).interpolate(limit_direction="both").to_numpy().T
    if data.shape[1] / sfreq < MIN_REC_S:
        raise Skip("under_1_min")
    data = data - data.mean(axis=1, keepdims=True)
    if sfreq != SFREQ:
        import mne
        data = mne.filter.resample(data, up=SFREQ, down=sfreq, verbose="ERROR")
    flat = flat_windows(data, flat_uv)
    data = data - data.mean(axis=0, keepdims=True)          # average reference over the 10 channels
    x, x_ms = _bands(data, SFREQ)
    return x, x_ms, flat


def to_windows(x):
    """[C, n] -> [n_win, C, 512] (the tail shorter than 4 s is dropped)."""
    c, n = x.shape
    k = n // WIN
    return x[:, :k * WIN].reshape(c, k, WIN).transpose(1, 0, 2)


def amplitude_ok(w, ptp_uv):
    """[n, C, T] -> bool [n]: no channel above ptp_uv peak-to-peak (flat channels: flat_windows)."""
    ptp = w.max(axis=2) - w.min(axis=2)
    return ptp.max(axis=1) <= ptp_uv


def rereference10(w19):
    """[n, 19, T] -> [n, 10, T], re-referenced to the average of the 10 shared channels."""
    w = w19[:, IDX10]
    return w - w.mean(axis=1, keepdims=True)


@dataclass
class PatientResult:
    x: np.ndarray | None = None          # [n, 19, 512] float16 (None for the 10-channel-only external sets)
    x_ms: np.ndarray | None = None
    x10: np.ndarray | None = None        # [n, 10, 512] float16
    x10_ms: np.ndarray | None = None
    start_s: np.ndarray | None = None    # [n] float32
    sfreq_orig: float | None = None
    stats: dict = field(default_factory=dict)

    @property
    def n_windows(self):
        return 0 if self.start_s is None else len(self.start_s)


def _new_stats():
    return dict(recordings_listed=0, recordings_read=0, recordings_used=0, skip_under_1_min=0,
                skip_missing_electrode=0, skip_too_many_missing=0, skip_too_few_electrodes=0, skip_error=0,
                recordings_interpolated=0, electrodes_interpolated=0, recordings_with_photic_channel=0,
                recordings_with_stimulation=0, windows_seen=0, windows_flat=0, windows_bad_amplitude=0,
                windows_photic=0, windows_seizure=0, windows_clean=0, windows_kept=0)


def process_patient(recordings, settings: Settings, kind="edf", loader=None) -> PatientResult:
    """recordings: list of dicts {path, seizures: [(start_s, stop_s), ...]} in session order.
    kind: "edf" (TUH, 19 channels + 10-channel copy) or "emotiv" (external, 10 channels only)."""
    st = _new_stats()
    need = settings.n_windows
    max_s = settings.max_load_min * 60 if settings.max_load_min else None
    parts, sf0 = [], None
    for rec in recordings:
        st["recordings_listed"] += 1
        if sum(len(p[2]) for p in parts) >= need:
            continue
        try:
            if kind == "edf":
                raw = (loader or load_edf)(rec["path"], max_s)
                fs = raw.info["sfreq"]
                has_ph, photic = photic_intervals(raw) if settings.photic else (False, [])
                x, x_ms, flat, n_interp = clean_raw(raw, settings.missing, settings.flat_uv, settings.max_interp)
                st["recordings_interpolated"] += int(n_interp > 0)
                st["electrodes_interpolated"] += n_interp
                st["recordings_with_photic_channel"] += int(has_ph)
                st["recordings_with_stimulation"] += int(bool(photic))
            else:
                x, x_ms, flat = clean_emotiv_csv(rec["path"], max_seconds=max_s, flat_uv=settings.flat_uv)
                fs, photic = EMOTIV_SFREQ, []
        except Skip as e:
            st[f"skip_{e}"] += 1
            continue
        except Exception:  # unreadable file: counted, never fatal for the package
            st["skip_error"] += 1
            continue
        st["recordings_read"] += 1
        w, w_ms = to_windows(x), to_windows(x_ms)
        start = np.arange(len(w), dtype=np.float32) * WIN_S
        amp = amplitude_ok(w, settings.ptp_uv)
        flat = flat[: len(w)] | flat_pieces(w, settings.flat_uv)
        ok = amp & ~flat
        pho = seizure_cut.window_mask(start, photic, settings.photic_margin_s, WIN_S)
        seiz = seizure_cut.window_mask(start, rec.get("seizures") or [], settings.seizure_margin_s, WIN_S)
        st["windows_seen"] += len(w)
        st["windows_flat"] += int(flat.sum())
        st["windows_bad_amplitude"] += int((~amp & ~flat).sum())
        st["windows_photic"] += int((ok & pho).sum())
        st["windows_seizure"] += int((ok & ~pho & seiz).sum())
        keep = np.flatnonzero(ok & ~pho & ~seiz)
        st["windows_clean"] += len(keep)
        have = sum(len(p[2]) for p in parts)
        keep = keep[: need - have]
        if len(keep):
            parts.append((w[keep], w_ms[keep], start[keep]))
            st["recordings_used"] += 1
            sf0 = sf0 or fs
    res = PatientResult(sfreq_orig=sf0, stats=st)
    if not parts:
        return res
    w = np.concatenate([p[0] for p in parts])
    w_ms = np.concatenate([p[1] for p in parts])
    res.start_s = np.concatenate([p[2] for p in parts]).astype(np.float32)
    st["windows_kept"] = len(res.start_s)
    if kind == "edf":
        res.x, res.x_ms = w.astype(np.float16), w_ms.astype(np.float16)
        res.x10, res.x10_ms = rereference10(w).astype(np.float16), rereference10(w_ms).astype(np.float16)
    else:
        res.x10, res.x10_ms = w.astype(np.float16), w_ms.astype(np.float16)
    return res


def seizure_windows(recordings, missing: str, ptp_uv: float, loader=None, max_seconds=None, min_overlap=0.5):
    """Stage 1 windows of TUSZ recordings: every window of the whole recording (no minute cap).
    y = 1 when at least `min_overlap` of the window lies inside a seizure; windows with a smaller, non-zero overlap
    are dropped (neither clearly seizure nor background). Windows above `ptp_uv` peak-to-peak or with a flat
    channel are dropped. -> (x [n, 19, 512] float16, y int8, start_s float32, stats)."""
    xs, ys, ss = [], [], []
    st = dict(recordings_read=0, skipped=0, windows=0, windows_seizure=0, windows_ambiguous=0, windows_bad=0)
    for rec in recordings:
        try:
            raw = (loader or load_edf)(rec["path"], max_seconds)
            x, _, flat, _ = clean_raw(raw, missing)
        except Exception:
            st["skipped"] += 1
            continue
        st["recordings_read"] += 1
        w = to_windows(x)
        start = np.arange(len(w), dtype=np.float32) * WIN_S
        frac = seizure_cut.window_overlap(start, rec.get("seizures") or [], WIN_S)
        ok = amplitude_ok(w, ptp_uv) & ~(flat[: len(w)] | flat_pieces(w))
        amb = (frac > 0) & (frac < min_overlap)
        keep = ok & ~amb
        st["windows_bad"] += int((~ok).sum())
        st["windows_ambiguous"] += int((ok & amb).sum())
        y = (frac >= min_overlap)[keep].astype(np.int8)
        st["windows"] += int(keep.sum())
        st["windows_seizure"] += int(y.sum())
        xs.append(w[keep].astype(np.float16)); ys.append(y); ss.append(start[keep])
    if not xs:
        return None, None, None, st
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss).astype(np.float32), st
