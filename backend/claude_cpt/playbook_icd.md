# How to work (ICD)

rules.md is the exact production ICD prompt. Every coding decision (which diagnoses, how many, their order,
group-specific rules, the predicted-CPT guidance) comes from rules.md and the chart's guidance file. This
playbook only adds HOW to read the chart and how to verify codes; it never overrides rules.md.

## Tools
- Chart text: already extracted for your whole batch in the batch's `_text.md` file, page by page. Pages marked
  NO TEXT LAYER (or whose text is garbled) must be viewed from the chart's PDF with the Read tool (pages parameter).
  The production model sees the pages as images, so do not skip them. `python3 pdftext.py <pdf>` is only a
  fallback if a chart's text is missing.
- ICD-10-CM codes (official CMS code files, FY2026 and FY2027):
  - `python3 icd10.py check --dos <MM/DD/YYYY> <code> [<code> ...]`: is each code valid and billable on the date of
    service, its description, and for a non-billable header its billable codes. Check all of a chart's codes in one call.
  - `python3 icd10.py search <term> [<term> ...] [-- <term> ...]`: billable codes whose description contains ALL the
    terms; separate several searches with `--` to run them in one call.

## Method, per chart
1. Read the whole chart. Find what rules.md tells you to look for (pre-op / post-op diagnosis, indication,
   findings, comorbidities, etc.).
2. Choose ICD1-ICD4 exactly as rules.md (and the guidance file, if the chart has one) instructs.
3. Verify EVERY code with `icd10.py check` before writing it. If a code is not valid or not billable on the date
   of service, replace it with the documented, valid, billable code (never a guess the chart does not support).
4. Write 1-2 sentences of reasoning per code, saying where in the chart it comes from.

## Confidence
- high: diagnoses clearly documented and the codes are an exact fit.
- medium: thin documentation (no post-op diagnosis / op note) or a judgment call on order or specificity.
- low: guessing.
