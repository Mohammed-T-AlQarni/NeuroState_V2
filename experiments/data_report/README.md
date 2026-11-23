# EEG data report (G:\NeuroState)

Builds `G:\NeuroState\Claude_Workspace\reports\EEG_Data_Report.docx` and `.pdf`: an overview of the TUH EEG data
(TUSZ, TUEP, TUEG) and the external Guinea-Bissau / Nigeria data, for a research team planning AI models.

## Run (Windows)

Double-click `run_report.bat`, or in a terminal:

```bat
run_report.bat
```

The first run creates a Python environment in `G:\NeuroState\Claude_Workspace\venvs\eeg_report` and installs the packages
in `requirements_report.txt` (needs internet once). If a step fails, fix the cause and resume, e.g. `run_report.bat --from-step 5`.
Other options: `--qc-n 400` (signal-check sample size, at least 200), `--workers 8` (threads for reading headers),
`--skip-drive` (skip the whole-drive overview).

## Privacy and safety

- Read-only on the data: files are only opened for reading; `common.out_path()` refuses to write outside `Claude_Workspace`.
- The report and `reports\scripts\outputs\*.json` hold only counts, percentages and ranges. Values of clinical / free-text
  columns are never written; categories with fewer than 5 rows are grouped; free text is reduced to keyword counts.
- `privacy_check.py` (last step) scans the report and outputs for any patient code found on the drive and fails if one appears.
- Per-file working tables (with file paths) go to `Claude_Workspace\cache\eeg_data_report` and must not be shared.
- Step 8 writes `Claude_Workspace\inventory\tuab_normal_eeg_manifest.csv` and `tuab_normal_eeg_exclusions.csv`, which hold patient
  codes and paths: keep them private. The TUAB listing (`inventory\tuab_v301_ls_lR.txt`) is made once with
  `ssh nedc-tuh-eeg@www.isip.piconepress.com "ls -lR data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf/"` (names and sizes only, no EEG download).

## Steps

| Step | Script | Reads | Source |
|---|---|---|---|
| 1 | `s1_inventory.py` | folder listing: file types, sizes, name patterns, patients, sessions, sidecars | all files |
| 2 | `s2_edf_headers.py` | every EDF header: channels, sampling rates, durations, reference, size check, sex/age | all EDFs |
| 3 | `s3_labels.py` | every `.csv` / `.csv_bi`: seizure events, durations, types | all label files |
| 4 | `s4_metadata.py` | DOCS spreadsheets: fields, % missing, Y/N counts, keyword counts per group | all rows |
| 5 | `s5_signal_qc.py` | MNE signal check: flat / noisy / extreme channels, length vs header | stratified sample (default 400) |
| 6 | `s6_external.py` | External_Datasets `.gz` and metadata CSVs | all files; amplitude on a sample |
| 7 | `s7_overlaps.py` | patient overlaps between datasets, label conflicts | all files |
| 8 | `s8_tuab_normal.py` | TUAB v3.0.1 normal-EEG files already on the drive (same name and size), minus patients in an epilepsy group or the TUEP controls; writes the private `normal_eeg` manifest | TUAB listing in `inventory\tuab_v301_ls_lR.txt` |
| 9 | `make_report.py` | builds the .docx and .pdf from the outputs | |
| 10 | `privacy_check.py` | scans report and outputs for patient codes | |

## Test without the real data

```bash
python tests/make_fake_neurostate.py /tmp/fakeG/NeuroState
python run_all.py --root /tmp/fakeG/NeuroState --qc-n 200
```
