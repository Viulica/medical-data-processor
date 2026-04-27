# Blocks & A-Lines — Companion Charge Analysis

Analysis of which main-line anesthesia CPT codes commonly carry companion charges:
- **Blocks** (6-series: 62xxx epidurals, 64xxx nerve blocks)
- **A-lines / vascular** (3-series: 36620 arterial line, 36556 central venous catheter)

## Methodology

Fetched `allCharges=true` from the Account Log Export API across 14 groups, 50 batches prior to the 04-21-2026 integration set, wide date range (2024-01-01 → 2026-04-25).

**Sample size:**
- 714 batch queries → **397 batches with data**
- **9,403 unique charge rows**
- **7,562 unique cases** (grouped by `AccountNumber`)
- **1,596 multi-line cases** (primary + companion charge)

One case = multiple rows in the API (one per charge line). Primary anesthesia line is the row with CPT `00xxx`/`01xxx`; block companions start with `6`, A-line/vascular companions start with `3`.

Raw data: `/tmp/all_charges_big.csv` (9,403 rows). Script: `/tmp/analyze_many_more.py`.

---

## Block pairings (aggregate, n≥20)

| Primary | Procedure | n | %blk | Dominant block code(s) |
|---|---|---:|---:|---|
| **01630** | Shoulder / upper arm | 271 | **78%** | 64415 interscalene (100% of blocks) |
| **01638** | Shoulder, open reduction | 74 | **76%** | 64415 interscalene |
| **01714** | Forearm / elbow open | 14 | **64%** | 64415 brachial plexus |
| **01961** | C-section | 58 | **64%** | 64488 TAP *or* 62322 epidural |
| **01402** | Knee arthroplasty (TKA) | 236 | **63%** | 64447 adductor ± 64473 lumbar plexus |
| **01392** | Hip / pelvis open | 18 | **67%** | 64447 |
| **01740** | Elbow | 18 | **56%** | 64415 |
| **01464** | Foot / ankle | 11 | **45%** | 64447 + 64445 combo |
| **01400** | Knee (closed) | 297 | **37%** | 64447 |
| **01830** | Wrist / forearm | 146 | **33%** | 64415 ± 64417 |
| **00840** | Lower abdomen | 193 | **24%** | 64488 TAP |
| **01230** | Hip, closed | 29 | **24%** | 64473 lumbar plexus |
| **01480** | Lower leg, open | 214 | **23%** | 64445 sciatic + 64447 |
| **01214** | Hip arthroplasty (THA) | 150 | **11%** | 64473 / 64450 |
| **01202** | Pelvic / hip open (IAS-FVO only) | 11 | **82%** | 64473 lumbar plexus |

## Never-blocked primary CPTs (high-n confirmers)

| Primary | Procedure | n | %blk |
|---|---|---:|---:|
| 00811 / 00812 / 00813 | Colonoscopy | 2,351 | **0%** |
| 00731 | Upper GI endoscopy | 639 | **0%** |
| 00142 | Cataract | 286 | **0%** |
| 00910 | Urology / cystoscopy | 173 | **0%** |
| 00902 | Anal / rectal | 167 | **0%** |
| 00170 | Mouth / throat | 126 | **0%** |
| 01810 | Upper extremity closed | 357 | 3% |
| 00952, 00918, 00921, 00920, 00160, 00126 | — | 410 combined | **0%** |

---

## A-line (3-series) patterns

A-lines cluster almost exclusively in **IAS-BHS cardiac/vascular cases**.

| Primary | Procedure | n | %aline | Code(s) | Groups |
|---|---|---:|---:|---|---|
| **00537** | Intrathoracic cardiac | 28 | **64%** | 36620 | IAS-BHS only |
| **00350** | Major vascular | 3 | 100% | 36620 | IAS-BHS |
| **00567** | Heart / pericardial | 3 | 67% | 36620 ± 36556 | IAS-BHS |
| **00541** | Cardiac CABG-adjacent | 2 | 50% | 36620/36556 | IAS-BHS |
| **00862** | Intrathoracic vascular | 21 | 5% | 36620 | IAS-BHS |
| **00790** | Major abdominal | 112 | 4% | 36620/36556 | IAS-BHS dominant |
| **00300** | Head / neck | 62 | 2% | 36620/36556 | IAS-BHS dominant |
| **00670** | Major spine | 57 | 2% | 36620 | spread |

**Key rule:** An A-line outside IAS-BHS is almost certainly a false positive. An A-line on any primary CPT other than the ones above is almost certainly a false positive.

---

## Group-specific quirks

### IAS-FVO (orthopedic-heavy)
Almost every ortho case gets a block.
- `01630` 107/109 = **98%** → 64415 interscalene
- `01638` 25/25 = **100%** → 64415
- `01202` 9/9 = **100%** → 64473 lumbar plexus
- `01402` 48% → almost always 64447 **+** 64473 combo (unlike other groups that bill just 64447)
- `01392` 86%, `01740` 71%, `01400` 20%, `01480` 32%

### MKI (peripheral ortho)
Block-heavy across almost all ortho CPTs.
- `01400` 54/60 = **90%** → 64447
- `01630` 42/44 = **95%** → 64415
- `01638` 13/13 = **100%** → 64415
- `01480` 91%, `01830` 77%, `01714` 88%, `01740` 80%

### GAP
- `01402` 64/69 = **93%** → 64447 (knee TKA near-universal)
- `01961` 20/24 = **83%** → 64488 TAP
- `01638` 80% → 64415

### SIO-PSS (TAP-heavy for lower abdomen)
- `00840` 32/35 = **91%** → 64488 TAP (vs GAP at 10%, DUN at 14%)
- `01480` 44%, `01630` 70%

### RIV-ANDREW ADAMS
- `00840` 5/5 = **100%** TAP (small n but consistent)

### IAS-BHS — C-section uses epidural, not TAP
- `01961` 22 cases, 45% blocked → **9 of 10 blocks were 62322 (epidural)**, only 1 was TAP. Every other group defaults to TAP 64488.
- `01230` hip-closed 64% → 64473 (lumbar plexus)
- `01214` hip THA 50% → 64473 (vs IAS-FVO at 3%)
- The **only group billing A-lines**

### IAS-BMH anomaly
- `01402` knee TKA only **10%** blocked (vs 93% at GAP, 100% at MKI/AIP)
- Worth investigating — either IAS-BMH hospital doesn't do blocks for knees, or there's a different billing convention at that facility.

---

## Validation rules for `peripheral_blocks` extraction

### EXPECTED BLOCKS (extraction should contain ≥1 of these)

```
01402        → 64447 (always); 64473 common (mandatory in IAS-FVO); 64445 sometimes
01630, 01638 → 64415
01202        → 64473                     (IAS-FVO only)
01400        → 64447
01480        → 64445 + 64447 combo
01714, 01740 → 64415
01392        → 64447
01961 [IAS-BHS]    → 62322 epidural
01961 [not IAS-BHS]→ 64488 TAP bilateral
00840        → 64488 TAP                 (strongly expected for SIO-PSS / RIV-AA)
01230        → 64473                     (hip closed)
01214        → 64473 / 64450             (THA — variable, only ~11% overall)
01464        → 64447 + 64445
01320        → 64447
```

### LIKELY FALSE POSITIVES (extraction of a block on these = red flag)

```
00811, 00812, 00813  (colonoscopy)
00731                (upper GI endoscopy)
00142                (cataract)
00902, 00910, 00918, 00921, 00920
00160, 00170
00126, 00952, 00145
```

### A-LINE (36620 / 36556) RULES

```
Likely valid primary CPTs:
  00537, 00350, 00541, 00562, 00567   (cardiac)
  00790                                (major abdominal)
  00670                                (major spine, rarely)
  00300, 00862                         (neck/thorax, rarely)

A-line on any other primary CPT         → almost certainly a false positive
A-line extracted on a non-IAS-BHS case  → almost certainly a false positive
```

---

## Most striking group/CPT combinations (for targeted prompt tuning)

| Group | Primary | Rate | Action |
|---|---|---:|---|
| IAS-FVO | 01630 | 98% | Hard-prior for 64415 |
| IAS-FVO | 01638 | 100% | Hard-prior for 64415 |
| IAS-FVO | 01202 | 100% | Hard-prior for 64473 (group-unique CPT) |
| IAS-FVO | 01402 | 48% w/ 64447+64473 combo | Teach the combo pattern |
| MKI | 01400 | 90% | Hard-prior for 64447 |
| MKI | 01630 | 95% | Hard-prior for 64415 |
| MKI | 01638 | 100% | Hard-prior for 64415 |
| GAP | 01402 | 93% | Hard-prior for 64447 |
| GAP | 01961 | 83% | Hard-prior for 64488 TAP |
| SIO-PSS | 00840 | 91% | Hard-prior for 64488 TAP |
| RIV-AA | 00840 | 100% | Hard-prior for 64488 TAP |
| IAS-BHS | 01961 | 45% | Prior for **62322 epidural** (not TAP!) |
| IAS-BHS | 00537 | 64% (A-line) | Expect 36620 companion |
