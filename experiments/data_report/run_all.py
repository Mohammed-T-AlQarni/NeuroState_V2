"""Run the whole EEG data report: steps 1-7, then the Word + PDF report.

    python run_all.py                    # default root G:\\NeuroState
    python run_all.py --root D:\\other\\NeuroState --qc-n 400

Read-only on the data. Writes only inside <root>\\Claude_Workspace:
  reports\\EEG_Data_Report.docx / .pdf      the report
  reports\\scripts\\*.py                    copy of these scripts (for reproducibility)
  reports\\scripts\\outputs\\*.json          aggregate totals behind every number in the report
  cache\\eeg_data_report\\*.csv              per-file working tables (contain file paths; keep private)
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import common as C

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    ap.add_argument("--qc-n", type=int, default=400, help="signal-QC sample size (>= 200)")
    ap.add_argument("--workers", type=int, default=8, help="threads for header reading")
    ap.add_argument("--skip-drive", action="store_true", help="skip the whole-drive overview")
    ap.add_argument("--from-step", type=int, default=1, help="resume from this step (1-9)")
    a = ap.parse_args()
    P = C.init(a.root)
    if not Path(a.root).is_dir():
        sys.exit(f"Root folder not found: {a.root}")
    log = C.Log(P["out"] / "run_log.txt")
    t0 = time.time()
    # keep a copy of the exact scripts next to the outputs
    for f in list(HERE.glob("*.py")) + list(HERE.glob("*.bat")) + list(HERE.glob("README*.md")) + list(HERE.glob("requirements_report.txt")):
        dst = C.out_path(P["scripts"] / f.name)
        if f.resolve() != dst.resolve():
            shutil.copy2(f, dst)
    import mne, numpy, openpyxl, pandas  # noqa: E401
    C.write_json(P["out"] / "run_meta.json", dict(run_date=dt.date.today().isoformat(), started=dt.datetime.now().isoformat(timespec="seconds"),
                                                   python=sys.version.split()[0], platform=platform.platform(), mne=mne.__version__,
                                                   pandas=pandas.__version__, numpy=numpy.__version__, openpyxl=openpyxl.__version__,
                                                   args=vars(a)))
    import make_report, privacy_check, s1_inventory, s2_edf_headers, s3_labels, s4_metadata, s5_signal_qc, s6_external, s7_overlaps
    steps = [
        (1, s1_inventory.main, dict(skip_drive=a.skip_drive)),
        (2, s2_edf_headers.main, dict(workers=a.workers, fresh=False)),
        (3, s3_labels.main, {}),
        (4, s4_metadata.main, {}),
        (5, s5_signal_qc.main, dict(n=max(a.qc_n, 200), min_per_stratum=8, seed=2026)),
        (6, s6_external.main, dict(amp_sample=20, seed=2026)),
        (7, s7_overlaps.main, {}),
        (8, make_report.main, {}),
        (9, privacy_check.main, {}),
    ]
    for n, fn, kw in steps:
        if n < a.from_step:
            continue
        try:
            rc = fn(SimpleNamespace(root=a.root, **kw))
            if n == 9 and rc:
                sys.exit(1)
        except Exception as e:
            log(f"STEP {n} FAILED: {type(e).__name__}: {e}")
            log(f"Fix the problem, then resume with:  python run_all.py --root \"{a.root}\" --from-step {n}")
            raise
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min. Report: {P['reports'] / 'EEG_Data_Report.docx'}")


if __name__ == "__main__":
    main()
