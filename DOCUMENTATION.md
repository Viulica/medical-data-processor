# Documentation

## Per-Group CPT Prediction Routing (hardcoded)

CPT (anesthesia/ASA code) prediction in the unified pipeline is **hardcoded per
worktracker group**. The routing lives in `process_unified_background` in
`backend/main.py` (a `CPT_GROUP_ROUTING` table near the top of the function).
It overrides whatever model/instruction/mode the caller requests.

The table has **two kinds of entries**:

**1. Template-based routing** — a specific model + a group-specific instruction
template. Two of these run through the crosswalk agent (CHA), two run plain
vision predict (ANA-*).

| Group        | Model                           | Template                | Mode            |
|--------------|---------------------------------|-------------------------|-----------------|
| CHA-HDH / CHA| `google/gemini-3.5-flash`       | #7 (CHA)                | Crosswalk agent |
| ANA-GMCE     | `google/gemini-3.1-pro-preview` | #225 (ANA-GMCE-CPT-NEW) | Vision          |
| ANA-SPS      | `google/gemini-3.1-pro-preview` | #226 (ANA-SPS-CPT-NEW)  | Vision          |

**2. Crosswalk-agent cohort (GENERIC, no template)** — a set of groups pinned to
the crosswalk agent with its **default/generic** instructions (`template_id =
None`) on a single shared model `AGENT_GENERIC_MODEL` (`google/gemini-3.7-flash`).
Each was validated to **beat production CPT accuracy** on the same batches.

| Group    | prod → agent | Group    | prod → agent |
|----------|--------------|----------|--------------|
| APO-UTP  | 83% → 97%    | DUN      | 89% → 94%    |
| APO-UPM  | (APO, +13)   | APO-ORA  | 93% → 100%   |
| UNI-ROB  | 88% → 97%    | PAC-STE  | 88% → 94%    |
| IAS-FVO  | 88% → 95%    | EAP-PHS  | 92% → 97%    |
| KAP-ASC  | 92% → 97%    | PDK      | 91% → 94%    |
| KAP-CYP  | 92% → 97%    | ANA-ORA  | 89% → 92%    |
|          |              | EAP-SSC  | 92% → 94%    |

- **Any group NOT in the table is left UNTOUCHED** — CPT runs exactly as the
  caller configured (its own model/template). CPT is **not** disabled for
  unlisted groups. (An earlier version incorrectly force-disabled CPT for
  unlisted groups; that was hotfixed — see commit `5fac9ad`.)
- `template_id = None` means: use the crosswalk agent's **built-in generic
  instructions**, and explicitly clear any caller-provided instructions so
  nothing group-specific leaks in. This exactly matches the validation setup
  (the agent was measured generic, without each group's own DB template).
- For agent-routed groups, the combined CPT+ICD vision path is disabled
  (`enable_combined_cpt_icd = False`) so CPT always runs through the agent; ICD
  still runs on its own vision path.

### ⚠️ The complicated part — why this looks inconsistent

The routing mixes THREE distinct configurations, on purpose, because they were
each validated separately and the "obvious" simplifications were measured to be
WRONG:

1. **Template groups run WITHOUT the agent's generic prompt; agent-cohort groups
   run WITHOUT their own template.** These are not interchangeable. Feeding a
   group's custom template into the agent, or dropping a template group onto the
   generic agent, is an **untested** config and may regress accuracy.
2. **The agent does NOT help every group.** It reliably improves groups whose
   production CPT accuracy sits in roughly the **85–93%** band (crosswalk-fixable
   errors). It **HURTS** groups already at ~96–100% (it second-guesses correct
   answers — e.g. UNI-GOLD 100%→94%, IAS-MOR 100%→94%) and does **not rescue**
   sub-80% groups (their errors are illegible-chart / judgment calls the
   crosswalk can't fix — e.g. ACW-GVS, EAP-LGS). So routing is **selective and
   measured per group**, never blanket.
3. **The UNI family is inconsistent:** UNI-ROB improves with the generic agent,
   but UNI-GOLD and UNI-RSC regress with it. Those two carry large custom CPT
   templates (#178, #199) that the generic agent lacks, which is the leading
   suspect — a template-aware re-test is the open follow-up before routing them.
4. **Model is group-dependent.** 3.7-flash won on APO; 3.5-flash won on the KAP /
   IAS-FVO / UNI-ROB / ANA-GAS cohort by small margins. The cohort is currently
   standardized on **3.7-flash** (via `AGENT_GENERIC_MODEL`) for operational
   simplicity + its zero-blank behavior; per-group model overrides are possible
   by replacing `_agent_generic()` with an explicit dict.

**Before adding/moving a group:** measure agent-vs-production on that group's
recent batches (score predicted CPT against the coder-billed CPT from the Changes
API `allCharges=true`, excluding 00811↔00812 colonoscopy swaps). Only route it if
the agent wins by a clear margin (≥ ~+3 pts) on a real sample.

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
