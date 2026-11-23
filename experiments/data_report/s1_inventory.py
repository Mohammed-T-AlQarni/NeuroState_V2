"""Step 1: file inventory. Walks folders (read-only), counts file types, sizes, name patterns,
patients, sessions, montage folders and annotation sidecars. Uses ALL files.

Private output (has paths): cache/manifest_edf.csv. Aggregate output: outputs/s1_inventory.json
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import common as C



def walk_dataset(key, log):
    root = C.ds_path(key)
    res = dict(key=key, path_rel=C.DS[key]["rel"], desc=C.DS[key]["desc"], role=C.DS[key]["role"], exists=root.is_dir())
    if not root.is_dir():
        log(f"  {key}: folder not found")
        return res, []
    ext_n, ext_b = Counter(), Counter()
    dir_pat = defaultdict(Counter)   # depth -> pattern counts
    file_pat = defaultdict(Counter)  # ext -> stem pattern counts
    edf_depth = Counter()
    stems_by_dir = defaultdict(set)
    edfs = []
    n_files = 0
    for dp, dns, fns in C.iter_tree(root):
        relp = Path(dp).relative_to(root)
        depth = len(relp.parts)
        for d in dns:
            dir_pat[depth + 1][C.name_pattern(d)] += 1
        for fn, sz in fns:
            n_files += 1
            e = C.ext_of(fn)
            ext_n[e] += 1
            ext_b[e] += sz
            stem = fn[: -len(e)] if e != "(none)" else fn
            file_pat[e][C.name_pattern(stem)] += 1
            stems_by_dir[dp].add((stem, e))
            if e in (".edf", ".bdf"):
                edf_depth[depth] += 1
                edfs.append((dp, fn, stem, sz, relp.parts))
    rows = []
    for dp, fn, stem, sz, parts in edfs:
        m = C.RE_EDF.match(stem)
        pid = m.group("pid").lower() if m else next((p for p in parts if C.RE_PID.match(p)), "")
        ses_dir = next((p for p in parts if C.RE_SES_DIR.match(p)), "")
        sm = C.RE_SES_DIR.match(ses_dir) if ses_dir else None
        ses = (m.group("ses").lower() if m else (sm.group("ses").lower() if sm else ""))
        year = sm.group("year") if sm and sm.group("year") else ""
        montage = parts[-1] if parts and C.RE_MONTAGE_DIR.match(parts[-1]) else (parts[-1] if parts else "")
        st = stems_by_dir[dp]
        rows.append(dict(dataset=key, path=os.path.join(dp, fn), pid=pid, ses=ses, tok=m.group("tok").lower() if m else "",
                         ses_year=year, montage=montage, size=sz, name_ok=int(bool(m)),
                         has_csv=int((stem, ".csv") in st), has_csv_bi=int((stem, ".csv_bi") in st)))
    # sidecars without an EDF
    orphan = Counter()
    for dp, st in stems_by_dir.items():
        edf_stems = {s for s, e in st if e in (".edf", ".bdf")}
        for s, e in st:
            if e in (".csv", ".csv_bi") and s not in edf_stems:
                orphan[e] += 1
    pats = defaultdict(set)
    pses = defaultdict(set)
    for r in rows:
        pats[r["pid"]].add(r["path"])
        pses[r["pid"]].add(r["ses"])
    res.update(
        n_files=n_files, n_bytes=sum(ext_b.values()),
        ext=[dict(ext=e, n=ext_n[e], bytes=ext_b[e]) for e, _ in ext_n.most_common()],
        n_edf=len(rows), edf_bytes=sum(r["size"] for r in rows),
        n_patients=len(pats), n_sessions=len({(r["pid"], r["ses"]) for r in rows}),
        edf_per_patient=C.dist([len(v) for v in pats.values()]),
        sessions_per_patient=C.dist([len(v) for v in pses.values()]),
        patients_multi_session=sum(1 for v in pses.values() if len(v) > 1),
        montage_folders=C.top(Counter(r["montage"] if C.RE_MONTAGE_DIR.match(r["montage"]) else "(other)" for r in rows), 20),
        session_year=C.dist([int(r["ses_year"]) for r in rows if r["ses_year"]]),
        session_year_missing=sum(1 for r in rows if not r["ses_year"]),
        name_not_standard=sum(1 for r in rows if not r["name_ok"]),
        edf_with_csv=sum(r["has_csv"] for r in rows), edf_with_csv_bi=sum(r["has_csv_bi"] for r in rows),
        edf_with_both=sum(1 for r in rows if r["has_csv"] and r["has_csv_bi"]),
        orphan_sidecars=dict(orphan),
        edf_depth=dict(sorted(edf_depth.items())),
        dir_patterns={str(d): C.top(c, 8) for d, c in sorted(dir_pat.items())},
        file_patterns={e: C.top(c, 5) for e, c in file_pat.items()},
    )
    log(f"  {key}: {n_files} files, {len(rows)} EDF, {len(pats)} patients, {res['n_sessions']} sessions")
    return res, rows


def drive_overview(root, log):
    """Counts per top-level and second-level folder under root (skips software folders)."""
    agg = defaultdict(lambda: dict(n=0, bytes=0, ext=Counter()))
    sk = []
    for dp, dns, fns in C.iter_tree(root, sk):
        parts = Path(dp).relative_to(root).parts
        k = "/".join(parts[:2]) if parts else "(root files)"
        for fn, sz in fns:
            a = agg[k]
            a["n"] += 1
            a["bytes"] += sz
            a["ext"][C.ext_of(fn)] += 1
    out = [dict(folder=k, n=v["n"], bytes=v["bytes"], top_ext=C.top(v["ext"], 6)) for k, v in sorted(agg.items())]
    log(f"  drive overview: {len(out)} folders, skipped {len(sk)} software folders")
    return out, len(sk)


def docs_listing():
    found = []
    for rel in C.DOC_DIRS:
        base = C.rel_path(rel)
        if not base.is_dir():
            continue
        cands = [base] if base.name.upper() == "DOCS" else [p for p in base.rglob("DOCS") if p.is_dir() and len(p.relative_to(base).parts) <= 2]
        for d in cands:
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    found.append(dict(folder=str(d.relative_to(C.paths()["root"])), file=str(p.relative_to(d)), bytes=p.stat().st_size))
    return found


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 1 inventory, root =", P["root"])
    t0 = time.time()
    res, all_rows = {}, []
    for key in C.DS:
        r, rows = walk_dataset(key, log)
        res[key] = r
        all_rows += rows
    C.write_csv(P["cache"] / "manifest_edf.csv", all_rows,
                ["dataset", "path", "pid", "ses", "tok", "ses_year", "montage", "size", "name_ok", "has_csv", "has_csv_bi"])
    ext_root = C.rel_path(C.EXTERNAL_REL)
    ext_inv = None
    if ext_root.is_dir():
        n, b = Counter(), Counter()
        for p in ext_root.rglob("*"):
            if p.is_file():
                e = C.ext_of(p.name)
                n[e] += 1
                b[e] += p.stat().st_size
        ext_inv = [dict(ext=e, n=n[e], bytes=b[e]) for e, _ in n.most_common()]
    overview, skipped = drive_overview(P["root"], log) if not args.skip_drive else ([], 0)
    # check the user's TUEG epilepsy_Y manifest (read-only)
    man = C.rel_path("Claude_Workspace/inventory/tueg_epilepsy_Y_edfs.txt")
    man_n = None
    if man.is_file():
        with open(man, encoding="utf-8", errors="replace") as f:
            man_n = sum(1 for line in f if line.strip().lower().endswith(".edf"))
    C.write_json(P["out"] / "s1_inventory.json", dict(
        datasets=res, external_ext=ext_inv, drive_overview=overview, drive_skipped_software_dirs=skipped,
        docs=docs_listing(), tueg_Y_manifest_edf_lines=man_n, seconds=round(time.time() - t0, 1),
        source="all files"))
    log(f"STEP 1 done in {time.time() - t0:.0f}s, {len(all_rows)} EDF in manifest")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    ap.add_argument("--skip-drive", action="store_true", help="skip the whole-drive overview walk")
    main(ap.parse_args())
