# How to work (ICD)

rules.md is the exact production ICD prompt. Every coding decision (which diagnoses, how many, their order,
group-specific rules, the predicted-CPT guidance) comes from rules.md and the chart's guidance file. This
playbook only adds HOW to read the chart and how to verify codes; it never overrides rules.md.

## Tools
- `python3 pdftext.py <pdf>`: the chart's text, page by page. Pages marked NO TEXT LAYER (or garbled) must be
  viewed with the Read tool (pages parameter). The production model sees the pages as images, so do not skip them.
- ICD-10 Codes connector (tools named mcp__claude_ai_ICD-10_Codes__*):
  - `search_codes`: find a code by description (or by code prefix)
  - `lookup_code`: confirm the description and billability of a code
  - `validate_code`: confirm the code is valid and billable; pass as_of = the date of service when the chart shows it
  - `get_hierarchy` / `get_category`: find the specific billable child of a header code

## Method, per chart
1. Read the whole chart. Find what rules.md tells you to look for (pre-op / post-op diagnosis, indication,
   findings, comorbidities, etc.).
2. Choose ICD1-ICD4 exactly as rules.md (and the guidance file, if the chart has one) instructs.
3. Verify EVERY code with the connector before writing it. If a code is not valid or not billable on the date
   of service, replace it with the documented, valid, billable code (never a guess the chart does not support).
4. Write 1-2 sentences of reasoning per code, saying where in the chart it comes from.

## Confidence
- high: diagnoses clearly documented and the codes are an exact fit.
- medium: thin documentation (no post-op diagnosis / op note) or a judgment call on order or specificity.
- low: guessing.
