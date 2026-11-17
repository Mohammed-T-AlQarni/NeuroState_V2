"""Prepare TUEP for run_experiment.py: ONE prepared file per PATIENT, in exactly prep.py's format.

  experiments/.venv/bin/python app/tools/tuep_prepare.py --root <tuh_eeg_epilepsy/v3.1.0> \
      [--inventory data/tuep/inventory] [--out data/tuep/prepared] [--max-minutes-per-patient 30] \
      [--seizure-margin-s 60] [--include-seizures] [--order chronological|spread]

Why one file per patient: run_experiment.py splits the rows of subjects.csv with StratifiedKFold. With one row per
patient, every recording of a patient is in training or in testing together, with no change to the training code.
One row per recording would put a patient's recordings on both sides of a split.

For each patient, recordings are taken in a fixed order until the cap is reached. Each recording:
  * the 19 standard channels (names normalised as the app does: 'EEG FP1-REF' -> Fp1, T7 -> T3 ...);
  * annotated seizures +/- the margin are cut out (interictal EEG; --include-seizures keeps them for a sensitivity run);
  * each remaining continuous piece is prepared exactly as prep.py does (average reference; 1-40 Hz 'dl' and 2-20 Hz
    'ms' copies at 128 Hz via prep.band_copy), so filtering never crosses a cut;
  * each piece is trimmed to whole 4-s windows (multiples of 512 samples), so run_experiment.windows() never makes a
    window that spans two pieces. Pieces shorter than one window are dropped.
The pieces are concatenated into data/tuep/prepared/sub-<patient>.npz (dl, ms, sfreq, ch_names) and subjects.csv gets
participant_id, Group (E = epilepsy, N = no epilepsy), duration_s. Then, from experiments/:

  .venv/bin/python run_experiment.py --prep ../data/tuep/prepared --out ../results/epilepsy_seqcnn \
      --groups E N --models baseline seqcnn --ms-input onehot --skip-leak

Known approximation: microstate features (transitions) are computed over the concatenated labels, so each junction
between pieces adds at most one switch; training crops (random 4-s starts) can rarely span a junction.
status.csv records what happened to every recording (prepared / not needed (cap) / incompatible / failed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))


def keep_intervals(duration: float, seizures: list[tuple[float, float]], margin: float) -> list[tuple[float, float]]:
    cuts = sorted((max(0.0, a - margin), min(duration, b + margin)) for a, b in seizures)
    out, t = [], 0.0
    for a, b in cuts:
        if a > t:
            out.append((t, a))
        t = max(t, b)
    if t < duration:
        out.append((t, duration))
    return out


def parse_intervals(s) -> list[tuple[float, float]]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [tuple(map(float, x.split("-"))) for x in s.split(";")]


def prepare_recording(edf: Path, seizures, margin: float, include_seizures: bool, min_samples: int, budget: int):
    """Returns (dl pieces, ms pieces, seconds used, n pieces). Exactly prep.py's steps on each piece."""
    import mne

    import prep
    import run_experiment as rx
    from app import pipeline_adapter as pa
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
    plan = pa.channel_plan(raw.ch_names, raw.get_channel_types())
    if plan["missing"]:
        raise ValueError(f"missing standard channels {plan['missing']}")
    pa.apply_channel_plan(raw, plan)
    raw.pick(prep.CHANNELS)
    dur = raw.n_times / raw.info["sfreq"]
    spans = [(0.0, dur)] if include_seizures else keep_intervals(dur, seizures, margin)
    dls, mss, used = [], [], 0
    for a, b in spans:
        if used >= budget:
            break
        if (b - a) * prep.SFREQ < min_samples:
            continue
        seg = raw.copy().crop(a, b, include_tmax=False)
        seg.set_eeg_reference("average", projection=False, verbose="ERROR")
        dl, ms = prep.band_copy(seg, 1.0, 40.0), prep.band_copy(seg, 2.0, 20.0)
        n = min(dl.shape[1], ms.shape[1])
        n -= n % rx.WIN  # whole 4-s windows only
        n = min(n, ((budget - used) // rx.WIN) * rx.WIN)
        if n < min_samples:
            continue
        dls.append(dl[:, :n]), mss.append(ms[:, :n])
        used += n
    return dls, mss, used / prep.SFREQ, len(dls)


def order_recordings(r: pd.DataFrame, how: str) -> pd.DataFrame:
    r = r.sort_values(["session_id", "token", "path"])
    if how == "spread":  # one recording from each session in turn, so the cap samples across sessions
        r = r.assign(_k=r.groupby("session_id").cumcount()).sort_values(["_k", "session_id", "token"]).drop(columns="_k")
    return r


def main(argv=None) -> int:
    import prep
    import run_experiment as rx
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--inventory", type=Path, default=REPO / "data" / "tuep" / "inventory")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "tuep" / "prepared")
    ap.add_argument("--max-minutes-per-patient", type=float, default=30.0,
                    help="cap per patient (also keeps run_experiment's in-memory load ~7 GB for 200 patients)")
    ap.add_argument("--seizure-margin-s", type=float, default=60.0)
    ap.add_argument("--include-seizures", action="store_true", help="keep annotated seizures (sensitivity analysis only)")
    ap.add_argument("--order", choices=["chronological", "spread"], default="chronological")
    ap.add_argument("--patients", nargs="*", help="only these patient ids (testing)")
    a = ap.parse_args(argv)
    inv = pd.read_csv(a.inventory / "inventory_recordings.csv")
    pats = pd.read_csv(a.inventory / "inventory_patients.csv")
    conflicts = set(pats.loc[pats.label_conflict, "patient_id"])
    a.out.mkdir(parents=True, exist_ok=True)
    rx.WIN = int(round(4.0 * prep.SFREQ))
    budget_all = int(a.max_minutes_per_patient * 60 * prep.SFREQ)
    budget_all -= budget_all % rx.WIN
    status, subjects, t0 = [], [], time.time()
    for pid, r in inv.groupby("patient_id"):
        if a.patients and pid not in a.patients:
            continue
        if pid in conflicts:  # a patient under both class folders has no single label: left out, and reported
            status += [{"recording_path": p, "status": "excluded", "status_reason": "patient under both class folders",
                        "prepared_s": 0.0} for p in r.path]
            continue
        dls, mss, used, n_rec = [], [], 0, 0
        for row in order_recordings(r, a.order).itertuples():
            st = {"recording_path": row.path, "status": "", "status_reason": "", "prepared_s": 0.0}
            if not row.compatible:
                st.update(status="incompatible", status_reason=str(row.problems))
            elif used >= budget_all:
                st.update(status="not_needed", status_reason=f"patient cap of {a.max_minutes_per_patient:g} min reached")
            else:
                try:
                    d, m, sec, npc = prepare_recording(a.root / row.path, parse_intervals(row.seizure_intervals),
                                                       a.seizure_margin_s, a.include_seizures, rx.WIN, budget_all - used)
                    if not d:
                        st.update(status="excluded", status_reason="no seizure-free piece of at least 4 s")
                    else:
                        dls += d; mss += m; used += sum(x.shape[1] for x in d); n_rec += 1
                        st.update(status="prepared", prepared_s=round(sec, 3), status_reason=f"{npc} piece(s)")
                except Exception as e:  # noqa: BLE001
                    st.update(status="failed", status_reason=f"{type(e).__name__}: {e}"[:300])
            status.append(st)
        if not dls:
            continue
        sid = f"sub-{pid}"
        np.savez_compressed(a.out / f"{sid}.npz", dl=np.concatenate(dls, 1), ms=np.concatenate(mss, 1), sfreq=prep.SFREQ,
                            ch_names=np.array(prep.CHANNELS))
        subjects.append({"participant_id": sid, "Group": r.group.iloc[0], "duration_s": round(used / prep.SFREQ, 1),
                         "n_recordings": n_rec, "n_pieces": len(dls)})
        print(f"{sid} ({r.group.iloc[0]}): {used / prep.SFREQ / 60:.1f} min from {n_rec} recording(s)", flush=True)
    pd.DataFrame(subjects).to_csv(a.out / "subjects.csv", index=False)
    pd.DataFrame(status).to_csv(a.out / "status.csv", index=False)
    (a.out / "prepare_settings.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "window_samples": rx.WIN,
        "sfreq": prep.SFREQ, "patients_prepared": len(subjects), "minutes": round((time.time() - t0) / 60, 2),
        "environment": rx.env_info()}, indent=2))
    print(f"done: {len(subjects)} patients -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
