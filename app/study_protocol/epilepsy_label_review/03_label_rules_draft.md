# Step 2: label rules for added patients (draft for Dr. Mohammed Shuraim)

TUEP's 100 + 100 neurologist-reviewed patients stay the core cohort. These rules apply only to patients added from
TUSZ or TUEG. They follow the ILAE practical definition of epilepsy (Fisher et al., 2014) and the ILAE definition
of acute symptomatic seizures (Beghi et al., 2010). Every point marked **[decide]** needs the neurologist's call.

## Three separate fields, never merged

- `patient_epilepsy_status`: the diagnosis, decided per patient.
- `reason_for_EEG`: why the EEG was ordered (referral, "rule out", first seizure).
- `recording_findings`: what that EEG showed (seizure captured, discharges, normal or abnormal).

A captured seizure, an anti-seizure medication, an abnormal EEG or a report-search hit is **evidence to review**,
never a label by itself. A blank cell or an N is **never** a negative diagnosis.

## Decision for one patient

**Include as EPILEPSY** when at least one statement:
- explicitly affirms an epilepsy diagnosis in the patient's history or assessment ("known epilepsy", "history of
  temporal lobe epilepsy", "JME on levetiracetam"), or "seizure disorder" with recurrent unprovoked seizures
  **[decide: accept "seizure disorder" alone?]**;
- is about the patient (not family) and is not negated; and
- no session contains an explicit statement that epilepsy was excluded.
Record when the diagnosis was established relative to each EEG **[decide: are EEGs recorded before the diagnosis
included, or only sessions at or after it?]**.

**Include as NO EPILEPSY (comparison)** only with an explicit assessment:
- "does not have epilepsy" or "epilepsy was ruled out"; or a confirmed alternative diagnosis for the events
  (e.g. syncope, PNES confirmed on video-EEG, migraine) **[decide: is a documented alternative enough without the
  words "no epilepsy"?]**; and
- no session suggests epilepsy, an epilepsy indication for medication, or epileptiform discharges
  **[decide: exclude comparators whose report notes interictal discharges?]**.
N in every session is **not** sufficient.

**UNRESOLVED (left out of the diagnosis cohort, kept in a separate list):**
- referral or suspicion only ("r/o seizure", "spells", "staring", "first seizure");
- acute symptomatic seizures only (hypoglycemia, withdrawal, acute stroke, trauma, CNS infection, anoxia), unless
  epilepsy is also documented;
- status epilepticus without a documented epilepsy history **[decide]**;
- family history only; PNES with possible coexisting epilepsy;
- sessions that conflict without a later clear diagnosis;
- spreadsheet fields only, with nothing explicit.

## Patient separation (unchanged by any label)

Every session, EDF and window of one person stays on one side of a split. Patients in the official TUSZ
development or evaluation sets stay out of training in every dataset, including their TUEP records. Maps and
models are fitted on training patients only.

## Pilot: the 16 keyword hits

Purpose: measure how many of the 16 can be decided **from the spreadsheet fields alone**, and what evidence is
missing for the rest. That fraction decides how much we depend on NEDC's report search and further verification.
Use `review_form.xlsx` (one row per person; keep it on the authorized PC). Report back only the counts per decision
and per "evidence missing" reason.
