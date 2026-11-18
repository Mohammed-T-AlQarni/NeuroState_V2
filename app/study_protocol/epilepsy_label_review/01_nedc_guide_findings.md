# Step 1: how the TUH metadata spreadsheets assign Y and N

Source: Melles, Oymann, Obeid & Picone, *Annotation Guidelines for NEDC's EEG Corpora*, 1 May 2026
(`guidelines_v04.docx`, SHA-256 `99ef21d5…660853`, downloaded 18 Nov 2026). The README's "(p. 27)" is the report's
page count; the metadata rules are in section "The EEG Metadata Format" (about pages 4–5 of the document).

## What the guide says

| Column (v00 numbering) | Rule, as written in the guide |
|---|---|
| A Subject ID | One entry per **session** (`aaaaaaac_s001`); every EDF in that session shares it. |
| B Electrographic Seizure | "Y" when the annotators saw a seizure **in the EEG signal**; a matching event annotation exists. |
| C Clinical Seizure | "Indication of a **history of seizures in the EEG report**. This does not necessarily mean a seizure was observed in the actual signal." |
| D–M (Stroke, **E Epilepsy**, Parkinson's, …, Depression) | "Contain a value of 'Y' if the EEG report indicated the presence of this condition. **Otherwise, it is labeled 'N'.**" |
| N Other Diagnosis | Free text copied from the report ("Reason for visit: …, Diagnosis: …"); not normalised. |
| O Major Medications | Comma-separated list from the report; spelling corrected, not normalised. |

Other statements that bear on the label:
- The values were **extracted from the report of that session** and "clustered into the most commonly occurring
  conditions"; the annotation team reviewed them after annotating TUSZ and TUEP, met weekly to agree criteria, and
  the project lead reviewed "some amount" of the work. Annotators are a student team; no clinician review of these
  columns is described.
- "Sometimes metadata is not consistent across sessions" (conflicting histories, shared record numbers, deception).
- Reports are "unstructured text… often ambiguous and defensive. The presence of a term does not necessarily
  indicate a positive or negative diagnosis." NEDC moved from keyword searches to manual clustering for this reason.
- "All our patients were referred to a hospital for evaluation of a serious medical condition, so they cannot be
  considered 'normal' subjects."

## What this means for our labels

1. **N is not a negative diagnosis.** N covers "epilepsy not mentioned in this session's report", "uncertain"
   and "explicitly ruled out" alike. The guide gives no separate code for any of them.
2. **The Epilepsy flag is specific but insensitive, measured against TUEP.** Among TUEP's 100 neurologist-reviewed
   epilepsy patients, 43 have Y in at least one session (17 always, 26 mixed) and 57 never; among the 100
   comparison patients, 1 has Y. TUEP's groups were assigned with more evidence (history, medications, EEG
   features, neurologist review), which explains the gap.
3. Therefore: **consistent Y** is a reasonable *screening* signal for epilepsy candidates (still to be verified);
   **consistent N cannot select comparators**; a comparator needs explicit evidence of an assessment.
4. **Clinical Seizure = Y** means a documented seizure history, not epilepsy (it includes provoked seizures) and not
   a seizure on the EEG.
5. The guide does not say how "possible", "rule out", "seizure disorder", provoked seizures or status epilepticus
   were coded. That is the question for NEDC (`02_question_to_nedc.md`).

## The report-search service (checked on the official catalogue page, 18 Nov 2026)

Still offered: email a Python script using only the standard library to help@nedcdata.org; NEDC runs it on the
anonymized reports and returns the names of matching files. The example (`query.tar.gz`) reads a file listing report
paths (argument 1) and prints each matching path. **Its loop has a misplaced `break`, so it only ever tests the first
line of each report** — do not copy that structure. Reports themselves are not distributed.
