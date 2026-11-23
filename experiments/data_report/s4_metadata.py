"""Step 4: metadata spreadsheets in the DOCS folders (TUEP metadata_v00*.xlsx, TUSZ metadata_v06r.xlsx,
seizures_types_v02.xlsx, ...). Reports which fields exist, % missing, numeric ranges, category counts
(only categories with >= 5 rows), and keyword counts per group (patients whose text mentions a keyword).
Never writes individual values, IDs or free text. Aggregate output: outputs/s4_metadata.json
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import common as C

MIN_CELL = 5
KEYWORDS = {
    "normal": r"\bnormal\b",
    "abnormal": r"\babnormal",
    "epilepsy/epileptic": r"epilep",
    "seizure": r"seiz|convuls",
    "syncope/fainting": r"syncop|faint|passed out|loss of consciousness|\bloc\b",
    "non-epileptic / psychogenic (PNES)": r"non[- ]?epileptic|\bpnes\b|psychogenic|pseudo[- ]?seiz|functional seiz|\bnes\b",
    "spells/episodes/events": r"spell|episode|\bevents?\b",
    "migraine/headache": r"migrain|headache",
    "stroke/CVA": r"stroke|\bcva\b|infarct|\btia\b",
    "head injury/TBI": r"\btbi\b|head (injury|trauma)|concussion|brain injury",
    "dementia/memory": r"dement|alzheim|memory",
    "encephalopathy/altered mental status": r"encephalopath|altered mental|\bams\b|confus",
    "tumour/mass": r"tumou?r|\bmass\b|neoplas|glioma|meningioma|metasta",
    "psychiatric (anxiety/depression/etc.)": r"anxiet|depress|psychiatr|bipolar|schizo|panic",
    "alcohol/drugs/withdrawal": r"alcohol|etoh|withdraw|overdose|substance",
    "sleep/drowsy": r"sleep|drows",
    "dizziness/vertigo": r"dizz|vertigo",
    "infection (meningitis/encephalitis/sepsis)": r"mening|encephalitis|sepsis|infect",
    "hypoxia/cardiac arrest": r"hypox|anox|cardiac arrest",
    "developmental (autism/CP/delay)": r"autis|cerebral palsy|developmental|delay",
}
FIELD_HINTS = {
    "age": r"\bage\b|age_|^age",
    "sex/gender": r"\bsex\b|gender",
    "medication": r"medic|drug|\baed\b",
    "diagnosis": r"diagn|epilep|seiz|condition",
    "comorbidity": r"comorb|history|other",
    "recording type (routine / long-term / ICU / EMU)": r"routine|\bltm\b|long|type|emu|icu|setting|location",
    "sleep / state": r"sleep|state|awake|drows",
    "date / year": r"date|year|time",
}


SENSITIVE = r"medic|drug|\baed|diagn|histor|note|remark|comment|comorb|impression|descr|clinical|reason|indicat|summary|condition|other|finding|interpret"


CODES = {"y", "n", "yes", "no", "u", "unk", "unknown", "t", "f", "true", "false", "0", "1", "m", "x", "male", "female"}


def short_codes_only(s: pd.Series) -> bool:
    v = s[~missing_mask(s)].astype(str).str.strip()
    return len(v) > 0 and v.str.lower().isin(CODES).all()


def may_list_values(name: str, s: pd.Series) -> bool:
    """Values of clinical / free-text columns are never listed (only yes/no/unknown-style codes)."""
    return (not re.search(SENSITIVE, name, re.I)) or short_codes_only(s)


def keyword_columns(df, kinds, id_col):
    medrx = FIELD_HINTS["medication"]
    return [c for c in df.columns if c != id_col and kinds[c] in ("text", "categorical")
            and not pd.api.types.is_numeric_dtype(df[c]) and not short_codes_only(df[c])
            and not re.search(medrx, c, re.I) and (kinds[c] == "text" or re.search(SENSITIVE, c, re.I))]


def find_docs():
    files = []
    for rel in C.DOC_DIRS:
        base = C.rel_path(rel)
        if not base.is_dir():
            continue
        if base.name.upper() == "DOCS":
            cands = [base]
        else:
            cands = [p for p in base.rglob("DOCS") if p.is_dir() and len(p.relative_to(base).parts) <= 2]
        for d in cands:
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~$"):
                    files.append(p)
    seen, out = set(), []
    for p in files:   # same file name in two copies of DOCS: keep both but mark
        out.append(p)
    return out


def detect_header(raw: pd.DataFrame) -> int:
    best, best_i = -1, 0
    for i in range(min(15, len(raw))):
        vals = raw.iloc[i].tolist()
        score = sum(1 for v in vals if isinstance(v, str) and v.strip() and not re.fullmatch(r"[\d.\-/: ]+", v.strip()))
        if score > best:
            best, best_i = score, i
    return best_i


def pid_of(v):
    if not isinstance(v, str):
        return None
    m = C.RE_ANY_PID.search(v.strip().lower())
    return m.group(1) if m else None


ID_NAME = r"(^|[^a-z])id([^a-z]|$)|subject|patient|file|name|code|bname|session|token"


def col_kind(s: pd.Series, name: str = ""):
    if name and re.search(ID_NAME, name, re.I):
        return "identifier"
    nn = s.dropna()
    if nn.empty:
        return "empty"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    num = pd.to_numeric(nn, errors="coerce")
    if num.notna().mean() > 0.95:
        return "numeric" if nn.nunique() > 10 else "categorical"
    nun = nn.astype(str).str.strip().str.lower().nunique()
    if nun <= 25 and nun / max(len(nn), 1) < 0.5:
        return "categorical"
    if nn.astype(str).map(lambda x: bool(pid_of(x))).mean() > 0.8:
        return "id"
    return "text"


def missing_mask(s: pd.Series):
    return s.isna() | s.astype(str).str.strip().str.lower().isin(["", "nan", "none", "n/a", "na", "unknown", "-", "?"])


def cat_counts(s: pd.Series):
    v = s[~missing_mask(s)].astype(str).str.strip()
    c = Counter(v)
    shown = {k: n for k, n in c.most_common() if n >= MIN_CELL and len(k) <= 40}
    other = sum(n for k, n in c.items() if k not in shown)
    if other:
        shown["(other values, each < 5 rows or long text)"] = other
    return shown


def summarize_sheet(df: pd.DataFrame, groups: dict, path: Path, sheet: str, header_row: int, extra_pids=()):
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed") or df[c].notna().any()]]
    out = dict(file=path.name, folder=str(path.parent.relative_to(C.paths()["root"])), sheet=sheet,
               header_excel_row=header_row + 1, n_rows=len(df), n_columns=df.shape[1], columns=[])
    kinds = {c: col_kind(df[c], c) for c in df.columns}
    for c in df.columns:
        s = df[c]
        miss = float(missing_mask(s).mean() * 100) if len(s) else 0
        e = dict(name=c, kind=kinds[c], pct_missing=round(miss, 1))
        if kinds[c] == "numeric":
            e["dist"] = C.dist(pd.to_numeric(s, errors="coerce").dropna().tolist())
            for k in ("sum", "mean"):
                e["dist"].pop(k, None)
        elif kinds[c] == "categorical" and may_list_values(c, s):
            e["counts"] = cat_counts(s)
        elif kinds[c] in ("text", "id", "identifier", "categorical"):
            e["n_distinct"] = int(s.dropna().astype(str).nunique())
        out["columns"].append(e)
    # field presence
    out["fields_found"] = {f: [c for c in df.columns if re.search(rx, c, re.I)] for f, rx in FIELD_HINTS.items()}
    # id column
    known = set().union(*groups.values()) | set(extra_pids)
    id_col, best = None, 0.0
    for c in df.columns:
        frac = df[c].dropna().astype(str).map(lambda x: pid_of(x) in known).mean() if df[c].notna().any() else 0
        if frac > best:
            id_col, best = c, frac
    if best < 0.5:
        out["id_column_found"] = False
        return out
    out["id_column_found"] = True
    out["id_column_match_pct"] = round(100 * best, 1)
    pids = df[id_col].map(pid_of)
    out["n_patient_ids"] = int(pids.nunique())
    textcols = keyword_columns(df, kinds, id_col)
    medcols = [c for c in df.columns if re.search(FIELD_HINTS["medication"], c, re.I)]
    by_group = {}
    for g, pset in groups.items():
        mask = pids.isin(pset)
        if not mask.any():
            continue
        sub = df[mask]
        sp = pids[mask]
        gres = dict(rows=int(mask.sum()), patients=int(sp.nunique()), patients_in_group=len(pset))
        gres["pct_missing"] = {c: round(float(missing_mask(sub[c]).mean() * 100), 1) for c in df.columns if c != id_col}
        gres["categorical_counts"] = {c: cat_counts(sub[c]) for c in df.columns
                                      if kinds[c] == "categorical" and may_list_values(c, df[c])
                                      and sub[c].dropna().astype(str).str.strip().str.lower().nunique() <= 12}
        nums = {}
        for c in df.columns:
            if kinds[c] == "numeric":
                d = C.dist(pd.to_numeric(sub[c], errors="coerce").dropna().tolist())
                d.pop("sum", None)
                nums[c] = d
        gres["numeric"] = nums
        # keyword counts at patient level over all text columns
        txt = sub[textcols].fillna("").astype(str).agg(" ".join, axis=1).str.lower() if textcols else pd.Series("", index=sub.index)
        per_pat = defaultdict(str)
        for p, t in zip(sp, txt):
            per_pat[p] += " " + t
        gres["keyword_patients"] = {k: sum(1 for t in per_pat.values() if re.search(rx, t)) for k, rx in KEYWORDS.items()}
        gres["keyword_text_columns"] = textcols
        if medcols:
            any_med = defaultdict(bool)
            medc = Counter()
            for p, row in zip(sp, sub[medcols].itertuples(index=False)):
                toks = set()
                for v in row:
                    if isinstance(v, str) and v.strip() and v.strip().lower() not in ("nan", "none", "n/a", "na", "unknown", "-"):
                        for t in re.split(r"[,;/\n]+", v.lower()):
                            t = re.sub(r"[^a-z\- ]", "", t).strip()
                            if 2 < len(t) <= 30:
                                toks.add(t)
                if toks:
                    any_med[p] = True
                medc.update(toks)
            gres["patients_with_any_medication_listed"] = sum(any_med.values())
            # counts are rows (sessions) mentioning the drug; only drugs in >= 5 rows
            gres["top_medications_rows"] = [[k, c] for k, c in medc.most_common(15) if c >= MIN_CELL]
        by_group[g] = gres
    out["by_group"] = by_group
    return out


def main(args):
    P = C.init(args.root)
    log = C.Log(P["out"] / "run_log.txt")
    log("STEP 4 metadata")
    man = C.read_csv(P["cache"] / "manifest_edf.csv")
    groups = defaultdict(set)
    for r in man:
        if r["pid"]:
            groups[r["dataset"]].add(r["pid"])
    extra = groups.get("TUEG_partial", set()) | groups.get("TUSZ_incomplete_copy", set())
    groups["TUSZ_all"] = groups.get("TUSZ_train", set()) | groups.get("TUSZ_dev", set()) | groups.get("TUSZ_eval", set())
    all_pids = set().union(*groups.values()) if groups else set()   # every patient code on the drive
    groups = {k: v for k, v in groups.items() if k not in ("TUEG_partial", "TUSZ_incomplete_copy")}
    sheets, codebook = [], None
    for p in find_docs():
        log("  reading", p.relative_to(P["root"]))
        try:
            if p.suffix.lower() == ".csv":
                book = {"(csv)": pd.read_csv(p, header=None, dtype=str, keep_default_na=False, na_values=[""])}
            else:
                book = pd.read_excel(p, sheet_name=None, header=None)
        except Exception as e:
            sheets.append(dict(file=p.name, error=f"{type(e).__name__}: {str(e)[:100]}"))
            continue
        for sh, raw in book.items():
            if raw.dropna(how="all").empty:
                continue
            hr = 2 if (p.name.lower().startswith("metadata_v06") and str(sh).upper() == "TUSZ") else detect_header(raw)
            if any(pid_of(v) in all_pids for v in raw.iloc[hr].tolist()):
                # the chosen "header" row holds a patient code, so the sheet has no header row and that row is
                # patient data: never use its values as column names (they would end up in the outputs)
                sheets.append(dict(file=p.name, folder=str(p.parent.relative_to(P["root"])), sheet=str(sh),
                                   n_rows=int(len(raw.dropna(how="all"))), n_columns=int(raw.shape[1]),
                                   note="no header row (the first rows are patient data), so it was not summarised"))
                continue
            df = raw.iloc[hr + 1:].copy()
            df.columns = [str(x) if not (isinstance(x, float) and pd.isna(x)) else f"unnamed_{i}" for i, x in enumerate(raw.iloc[hr])]
            if p.name.lower().startswith("seizures_types"):
                # codebook (label definitions, no patient data): keep it whole
                codebook = dict(file=p.name, sheet=str(sh), columns=list(df.columns),
                                rows=df.dropna(how="all").astype(str).replace("nan", "").values.tolist()[:80])
                continue
            try:
                sheets.append(summarize_sheet(df, groups, p, str(sh), hr, extra))
            except Exception as e:
                sheets.append(dict(file=p.name, sheet=str(sh), error=f"{type(e).__name__}: {str(e)[:100]}"))
    C.write_json(P["out"] / "s4_metadata.json", dict(sheets=sheets, seizure_type_codebook=codebook,
                                                     keywords=KEYWORDS, min_cell=MIN_CELL,
                                                     source="all rows of the metadata spreadsheets"))
    log(f"STEP 4 done: {len(sheets)} sheets summarised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.DEFAULT_ROOT)
    main(ap.parse_args())
