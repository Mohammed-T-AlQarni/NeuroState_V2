@echo off
setlocal
rem  EEG data report for G:\NeuroState. Read-only on the data: it only writes inside Claude_Workspace.
rem  Double-click this file, or run it in a terminal. Extra options are passed on, e.g.  run_report.bat --qc-n 300
set "ROOT=G:\NeuroState"
set "HERE=%~dp0"
set "VENV=%ROOT%\Claude_Workspace\venvs\eeg_report"
if not exist "%ROOT%\" (echo Cannot find %ROOT%. Is the G: drive connected? & pause & exit /b 1)
if exist "%VENV%\Scripts\python.exe" goto install
echo Creating a private Python environment in %VENV% ...
where py >nul 2>nul && (py -3 -m venv "%VENV%") || (python -m venv "%VENV%")
if not exist "%VENV%\Scripts\python.exe" (echo Python 3 was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run this file again. & pause & exit /b 1)
:install
echo Checking packages: mne, pandas, openpyxl, numpy, python-docx, reportlab ...
"%VENV%\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r "%HERE%requirements_report.txt"
if errorlevel 1 (echo Package installation failed. Check the internet connection and try again. & pause & exit /b 1)
"%VENV%\Scripts\python.exe" "%HERE%run_all.py" --root "%ROOT%" %*
if errorlevel 1 (echo. & echo Something went wrong. See the messages above and %ROOT%\Claude_Workspace\reports\scripts\outputs\run_log.txt & pause & exit /b 1)
echo.
echo Done. Report: %ROOT%\Claude_Workspace\reports\EEG_Data_Report.docx and EEG_Data_Report.pdf
pause
