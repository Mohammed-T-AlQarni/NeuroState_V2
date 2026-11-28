"""Test helpers: tiny EDF writer, fake patient codes, synthetic EEG. No real data, no real codes."""
import gzip
import string

import numpy as np

from experiments.epilepsy.preprocess import CH10, CH19


def fake_pid(i: int, prefix: str = "zq") -> str:
    """Eight lowercase letters, like a TUH code, but made up: prefix + 6 letters from i."""
    letters = []
    for _ in range(8 - len(prefix)):
        letters.append(string.ascii_lowercase[i % 26])
        i //= 26
    return prefix + "".join(reversed(letters))


def edf_name(pid: str, ses: int = 1, tok: int = 0) -> str:
    return f"{pid}_s{ses:03d}_t{tok:03d}.edf"


def sines(n_ch, seconds, sfreq, amp=20.0, f0=2.0, seed=0, noise=2.0):
    """Channel k: sine at f0 + k Hz (so channel order can be checked) plus a little noise; microvolts."""
    t = np.arange(int(seconds * sfreq)) / sfreq
    rng = np.random.default_rng(seed)
    x = np.stack([amp * np.sin(2 * np.pi * (f0 + k) * t) for k in range(n_ch)])
    return x + noise * rng.standard_normal(x.shape)


def write_edf(path, data_uv, sfreq, labels, age=40, sex="M", startdate="01.01.10"):
    """Minimal EDF writer (1-s records, int16, +/-3276.7 uV). data_uv: [n_ch, n_samples]."""
    n_ch, n = data_uv.shape
    spr = int(sfreq)
    n_rec = n // spr
    dig = np.clip(np.round(np.asarray(data_uv)[:, : n_rec * spr] * 10), -32767, 32767).astype("<i2")

    def f(s, w):
        return str(s)[:w].ljust(w).encode("latin-1")

    hdr = (f("0", 8) + f(f"X {sex} 01-JAN-1970 X Age:{age}", 80) + f("Startdate X", 80) + f(startdate, 8)
           + f("00.00.00", 8) + f(256 + 256 * n_ch, 8) + f("", 44) + f(n_rec, 8) + f(1, 8) + f(n_ch, 4))
    sig = b"".join(f(l, 16) for l in labels) + f("", 80) * n_ch + f("uV", 8) * n_ch
    sig += f("-3276.7", 8) * n_ch + f("3276.7", 8) * n_ch + f("-32767", 8) * n_ch + f("32767", 8) * n_ch
    sig += f("", 80) * n_ch + f(spr, 8) * n_ch + f("", 32) * n_ch
    body = dig.reshape(n_ch, n_rec, spr).transpose(1, 0, 2).tobytes()
    with open(path, "wb") as fh:
        fh.write(hdr + sig + body)
    return path


def tuh_labels(chans=CH19, suffix="-LE", extra=("EKG1", "T1"), order_seed=None):
    names = [f"EEG {c}{suffix}" for c in chans]
    if order_seed is not None:
        names = list(np.random.default_rng(order_seed).permutation(names))
    return names + [f"EEG {e}{suffix}" for e in extra]


def write_tuh_edf(path, seconds=600, sfreq=256, suffix="-LE", drop=(), extra=("EKG1", "T1"), spike_at=None,
                  order_seed=1, flat=(), zero_start_s=0, common_start_s=0, **kw):
    """A TUH-like EDF: the 19 electrodes (shuffled order, given suffix) + extra channels. Electrode k (in CH19
    order) carries a sine at 2 + k Hz. spike_at: (second, channel) of a 500 uV artefact; flat: electrodes set to 0;
    zero_start_s: every channel 0 for the first seconds (as some TUH files start)."""
    chans = [c for c in CH19 if c not in drop]
    data = sines(len(CH19), seconds, sfreq)
    for c in flat:
        data[CH19.index(c)] = 0.0
    if zero_start_s:
        data[:, : int(zero_start_s * sfreq)] = 0.0
    if common_start_s:  # every electrode carries the same signal (cancels under the average reference)
        n = int(common_start_s * sfreq)
        data[:, :n] = data[0, :n] * 5
    if spike_at is not None:
        s, ch = spike_at
        data[ch, int(s * sfreq): int((s + 0.5) * sfreq)] += 500.0
    rows = {c: data[k] for k, c in enumerate(CH19)}
    names = [f"EEG {c}{suffix}" for c in chans]
    perm = np.random.default_rng(order_seed).permutation(len(chans))
    names = [names[i] for i in perm]
    sig = [rows[chans[i]] for i in perm]
    for e in extra:
        names.append(f"EEG {e}{suffix}")
        sig.append(np.random.default_rng(5).standard_normal(len(sig[0])) * 50)
    return write_edf(path, np.stack(sig), sfreq, names, **kw)


def write_csv_bi(edf_path, events):
    """events: [(start, stop, label)]. Written next to the EDF like TUH does."""
    p = str(edf_path)[:-4] + ".csv_bi"
    lines = ["# version = csv_v1.0.0", "#", "channel,start_time,stop_time,label,confidence"]
    lines += [f"TERM,{s:.4f},{e:.4f},{l},1.0000" for s, e, l in events]
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return p


EMOTIV14 = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"]


def write_emotiv_csv(path, seconds=120, sfreq=128):
    """Emotiv-like .csv.gz: index column, 14 channels with a ~4000 uV offset, extra columns. Shared channel k
    (in CH10 order, T7 = T3 etc.) carries a sine at 2 + k Hz."""
    alias = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    data = sines(len(CH10), seconds, sfreq)
    cols = {}
    for name in EMOTIV14:
        std = next((c for c in CH10 if alias.get(c, c) == name), None)
        cols[name] = (data[CH10.index(std)] if std else np.zeros(data.shape[1])) + 4000.0
    n = data.shape[1]
    with gzip.open(path, "wt") as fh:
        fh.write('"",' + ",".join(f'"{c}"' for c in EMOTIV14 + ["COUNTER", "GYROX"]) + "\n")
        for i in range(n):
            fh.write(f'"{i + 1}",' + ",".join(f"{cols[c][i]:.3f}" for c in EMOTIV14) + f",{i % 128},1500\n")
    return path


def dominant_freq(w):
    """[n, C, T] at 128 Hz -> dominant frequency per channel (Hz), averaged over windows."""
    spec = np.abs(np.fft.rfft(w.astype(np.float64), axis=2)).mean(axis=0)
    f = np.fft.rfftfreq(w.shape[2], 1 / 128)
    return f[spec.argmax(axis=1)]
