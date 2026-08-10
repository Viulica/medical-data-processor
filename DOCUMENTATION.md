# Documentation

## Per-Group CPT Prediction Routing (hardcoded)

CPT (anesthesia/ASA code) prediction in the unified pipeline is **hardcoded per
worktracker group**. The routing lives in `process_unified_background` in
`backend/main.py` (a `CPT_GROUP_ROUTING` table near the top of the function).
It overrides whatever model/instruction/mode the caller requests.

| Group        | Model                       | Instruction template            | Mode            |
|--------------|-----------------------------|---------------------------------|-----------------|
| CHA-HDH / CHA| `google/gemini-3.5-flash`   | #7 (CHA)                        | Crosswalk agent |
| ANA-GMCE     | `google/gemini-3.1-pro-preview` | #225 (ANA-GMCE-CPT-NEW)     | Vision          |
| ANA-SPS      | `google/gemini-3.1-pro-preview` | #226 (ANA-SPS-CPT-NEW)      | Vision          |
| any other    | —                           | —                               | **CPT disabled**|

- Groups not in the table have CPT prediction **forced off** regardless of the
  `enable_cpt` flag, so no CPT logic runs on unsupported groups.
- For agent-routed groups (CHA), the combined CPT+ICD vision path is disabled
  (`enable_combined_cpt_icd = False`) so CPT always runs through the agent; ICD
  still runs on its own vision path.

### How it's wired

- **DB instructions:** `prediction_instructions` rows
  - **#225 `ANA-GMCE-CPT-NEW`** — GMCE rules + data-mined procedure→code reference
    table + I&D / ortho sub-code fixes (the "v4" ruleset).
  - **#226 `ANA-SPS-CPT-NEW`** — ANA-SPS urology+eye rules incl. the lithotripsy
    00910/00918 discriminator ("v2" ruleset). (CHA uses existing #7.)
- **Agent implementation:** `predict_codes_from_pdfs_agent` in
  `backend/general-coding/predict_general.py` runs `cpt_agent.run_agent`
  (vision + crosswalk tools) per PDF and writes the SAME CSV schema
  (`Patient Filename, ASA Code, Procedure Code, Code Explanation, Model Source,
  Error Message`) as the vision predictor, so the pipeline consumes it
  interchangeably. It's invoked at all three CPT-vision call sites in the
  unified pipeline, branched on the `cpt_use_agent` flag set by the routing.
- The agent path depends on `cpt_agent_eval/` (2025 crosswalk.xlsx + embeddings);
  ensure those ship to the backend runtime.

### Why these choices (evidence, Aug 2026)

These groups have **handwritten** anesthesia records. Production Gemini 3 Flash
hallucinated codes because it could not read the handwriting (not a data mixup —
off-by-one shift and PDF-split-overlap hypotheses were both tested and disproven).

Model comparison on the same handwritten ANA-GMCE cases (110 cases, same rules):

| Model            | Accuracy |
|------------------|----------|
| Gemini 3.1 Pro   | ~93%     |
| Gemini 3 Flash   | 83%      |
| gpt-5.6-luna     | 80%      |
| qwen 3.7 flash   | 62%      |

Validated accuracy with the hardcoded model + rules:
- **ANA-SPS:** ~94% (84 cases) with 3.1 Pro + #226 (litho rule).
- **ANA-GMCE:** ~93% (110 cases) with 3.1 Pro + #225 (reference + sub-code fixes).

Residual errors are mostly the irreducible colonoscopy 00811/00812 coin-flip
(billers themselves inconsistent) plus a few one-off sub-codes.

### Known limitations (handwritten ANA batches)

Even with 3.1 Pro, **non-CPT fields on these handwritten forms remain error-prone**
and need human review:
- **DOS:** the handwritten date field is often wrong (surgeons mis-date the form).
  The correct DOS is the **batch/filename date** (matched billed truth 15/15 on
  ANA-SPS batch 54) — prefer that source over the handwritten date.
- **Times / Physical Status (P-mod):** genuinely illegible; even human coders
  guess (~60–67% by any model). Route to manual review.
