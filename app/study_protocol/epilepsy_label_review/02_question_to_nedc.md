# Draft email to NEDC (help@nedcdata.org)

**Subject:** Epilepsy column in metadata_v00r / v06r: how N was assigned, and report-search output format

Dear NEDC team,

Thank you for TUEP v3.1.0 and TUSZ v2.0.6. We are using TUEP's patient groups for a patient-level epilepsy /
no-epilepsy study and would like to screen additional TUSZ and TUEG patients. The Annotation Guidelines (v04, May
2026, "The EEG Metadata Format") say the Epilepsy column is "Y" if the report indicated the condition and "N"
otherwise. Could you clarify the following?

1. Was **N** assigned in the same way when epilepsy was (a) not mentioned, (b) uncertain ("possible", "rule out",
   "evaluate for", "spells"), and (c) explicitly ruled out? Is there any record that separates these cases?
2. How were these coded in the Epilepsy column: "seizure disorder", "history of seizures" with anti-seizure
   medication, status epilepticus, provoked or acute symptomatic seizures, non-epileptic events/PNES, and family
   history of epilepsy?
3. Was each session coded **only from that session's report**, or did annotators use other sessions of the same
   patient?
4. Were the columns coded by trained annotators only, or reviewed by a clinician? Was agreement between annotators
   measured?
5. For TUEP, is the **per-patient basis** of the neurologist-reviewed group assignment available (for example,
   which criterion decided each patient)? Could that review process be applied to additional patients we identify?
6. **Report search:** the example `query.py` prints matching filenames. May our script print one line per match as
   `rule_id <TAB> report path`, so that several search rules can run in one request? Could you also return a short
   de-identified excerpt (the matching sentence) to let a neurologist verify each hit, or is filename-only output
   the limit?

(Minor: the example `query.py` has a `break` inside the line loop, so it tests only the first line of each report.)

Thank you for your help.

Best regards,
Mohammed T. AlQarni
NeuroState project
