"""Final safety check: scan the report (.docx text) and every aggregate output for real patient codes
(the 8-letter TUH IDs found on the drive). Prints only how many were found, never the codes.
Exit code 1 if any is found: then do NOT send the report.
"""
from __future__ import annotations

import argparse
import re
import sys

import common as C


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    pids = {r["pid"] for r in C.read_csv(P["cache"] / "manifest_edf.csv") if r["pid"]}
    texts = {}
    for f in list(P["out"].glob("*.json")) + list(P["out"].glob("*.txt")):
        texts[f.name] = f.read_text(encoding="utf-8", errors="replace")
    docx = P["reports"] / "EEG_Data_Report.docx"
    if docx.exists():
        from docx import Document
        d = Document(str(docx))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts += [c.text for c in row.cells]
        texts[docx.name] = "\n".join(parts)
    hits = {}
    for name, t in texts.items():
        found = set(re.findall(r"(?<![a-z])[a-z]{8}(?![a-z])", t.lower())) & pids
        if found:
            hits[name] = len(found)
    if hits:
        log("PRIVACY CHECK FAILED: patient codes found in " + ", ".join(f"{k} ({v})" for k, v in hits.items()) + ". Do not send the report.")
        return 1
    log(f"PRIVACY CHECK PASSED: none of the {len(pids)} patient codes appears in the report or the {len(texts) - (1 if docx.exists() else 0)} output files.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    sys.exit(main(ap.parse_args()))
