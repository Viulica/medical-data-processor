# CPT Precision Analysis

## SIO-PSS

**Data:** 37 batches (batch 263–299), 1,666 charge lines
**Overall CPT Accuracy:** 95.3% (1,588 / 1,666)

### High Precision CPT Codes (100%, min 5 cases)

Covers **59.5%** of all cases (991 / 1,666). With 00812 (96.4%), coverage rises to **77.6%**.

| CPT | Cases | Precision | Description |
|-----|-------|-----------|-------------|
| 00731 | 287 | 100% | Upper GI endoscopy |
| 00813 | 281 | 100% | Colonoscopy (screening→diagnostic w/ polyp) |
| 00170 | 91 | 100% | Intraoral procedures |
| 00126 | 51 | 100% | Cataract surgery |
| 01480 | 41 | 100% | Lower leg/ankle/foot |
| 00160 | 40 | 100% | Nose/sinus |
| 00952 | 31 | 100% | Lumbar epidural labor |
| 64488 | 30 | 100% | TAP block continuous |
| 00145 | 21 | 100% | Eye (vitrectomy etc.) |
| 00300 | 16 | 100% | Head/neck vessels |
| 64445 | 16 | 100% | Sciatic nerve block |
| 00940 | 15 | 100% | Vaginal delivery |
| 64415 | 12 | 100% | Brachial plexus block |
| 01400 | 12 | 100% | Knee arthroscopy |
| 00120 | 10 | 100% | Eye (external) |
| 01470 | 9 | 100% | Upper leg embolectomy |
| 01630 | 9 | 100% | Shoulder open |
| 01965 | 7 | 100% | C-section |
| 64447 | 6 | 100% | Femoral nerve block |
| 00400 | 6 | 100% | Integumentary (breast) |
| 00812 | 302 | 96.4% | Colonoscopy screening (6 errors → 00811) |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00811 | 215 | 76.3% | 33x → 00812 (screening vs diagnostic colonoscopy) |
| 00840 | 57 | 87.7% | 3x → 00851 |
| 01810 | 32 | 90.6% | 2x → 00400 |
| 01830 | 32 | 93.8% | 1x → 01820 |
| 00402 | 11 | 90.9% | 1x → 00802 |

### Key Takeaway

Colonoscopy 00811↔00812 swaps account for **39 / 78 total errors (50%)**. Excluding colonoscopy confusion, CPT accuracy is ~97.5%.

---

## IAS-BHS

**Data:** 47 batches (batch 219–268), 1,429 charge lines
**Overall CPT Accuracy:** 93.1% (1,330 / 1,429)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **62.9%** of all cases (899 / 1,429).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00813 | 183 | Colonoscopy (screening→diagnostic w/ polyp) |
| 00731 | 136 | Upper GI endoscopy |
| 00952 | 34 | Lumbar epidural labor |
| 00537 | 26 | Open heart (cardiac bypass) |
| 76937 | 24 | Ultrasound guidance (vascular access) |
| 64447 | 22 | Femoral nerve block |
| 00902 | 21 | Anorectal surgery |
| 01961 | 21 | C-section (emergency) |
| 00873 | 20 | Lithotripsy/cystoscopy |
| 64473 | 15 | Lumbar plexus block |
| 00300 | 14 | Head/neck vessels |
| 00520 | 14 | Chest procedures (closed) |
| 00918 | 14 | Transurethral procedures |
| 01630 | 14 | Shoulder open |
| 00912 | 13 | Transurethral resection |
| 64415 | 13 | Brachial plexus block |
| 62322 | 13 | Lumbar epidural single shot |
| 01402 | 10 | Knee arthroplasty |
| 00160 | 9 | Nose/sinus |
| 64445 | 9 | Sciatic nerve block |
| 01830 | 8 | Open femur/hip fracture |
| 01214 | 8 | Total hip arthroplasty |
| 01230 | 7 | Upper leg (femur) |
| 01638 | 7 | Shoulder arthroplasty |
| 00921 | 6 | Bladder surgery |
| 00940 | 6 | Vaginal delivery |
| 01922 | 5 | CT/interventional radiology |
| 64488 | 5 | TAP block continuous |
| 01968 | 5 | C-section + hysterectomy |

**95–99% precision:**

| CPT | Cases | Precision | Description |
|-----|-------|-----------|-------------|
| 00790 | 56 | 98.2% | Intra-abdominal upper (1x → 00752) |
| 36620 | 31 | 96.8% | Arterial line (1x → 93312) |
| 01480 | 48 | 95.8% | Lower leg/ankle/foot (1x → 01392, 1x → 01474) |
| 00840 | 82 | 95.1% | Intra-abdominal lower (2x → 00752, 1x → 00790, 1x → 00700) |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00860 | 16 | 18.8% | 7x → 00902, 4x → 00400, 1x → 00300 |
| 01942 | 6 | 33.3% | 2x → 00630, 2x → 00300 |
| 01210 | 11 | 45.5% | 4x → 01230, 1x → 01250, 1x → 01220 |
| 00630 | 7 | 57.1% | 3x → 00670 |
| 01470 | 15 | 60.0% | 3x → 00400, 2x → 00300, 1x → 01250 |
| 00811 | 75 | 90.7% | 7x → 00812 |
| 01810 | 15 | 80.0% | 3x → 01830 |
| 00400 | 40 | 87.5% | 2x → 01470, 2x → 01610, 1x → 00300 |
| 00812 | 124 | 86.3% | 17x → 00811 |
| 00862 | 15 | 86.7% | 2x → 00918 |

### Key Takeaways

- **00860 (hip surgery)** is the worst code at 18.8% — the model confuses it with 00902, 00400, and 00300. Only 16 cases but 13 wrong.
- **01210 (hip open)** at 45.5% — confused with 01230/01250/01220 (different hip procedure variants).
- **Colonoscopy 00811↔00812** still a factor: 24 swaps out of 99 total errors (24%).
- More diverse procedure mix than SIO-PSS — 100+ distinct CPT codes vs ~37, which spreads errors across more codes.

---

## IAS-NSC

**Data:** 14 batches (batch 83–122), 225 charge lines
**Overall CPT Accuracy:** 96.4% (217 / 225)

### High Precision CPT Codes (100%, min 5 cases)

Covers **84.0%** of all cases (189 / 225). Very concentrated procedure mix — orthopedic surgery center.

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 51 | Lens procedures (cataract) |
| 64447 | 27 | Femoral nerve block |
| 01402 | 26 | Knee arthroplasty |
| 64415 | 25 | Brachial plexus block |
| 01630 | 19 | Shoulder open |
| 01400 | 17 | Knee arthroscopy |
| 01214 | 12 | Total hip arthroplasty |
| 64473 | 7 | Lumbar plexus block |
| 01942 | 5 | CT/interventional radiology |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00630 | 3 | 66.7% | 1x → 01942 |
| 00140 | 3 | 66.7% | 1x → 00142 |
| 01810 | 7 | 85.7% | 1x → 01830 |

### Key Takeaways

- Very clean group — only 8 errors total across 225 charge lines.
- Highly concentrated procedure mix (ortho + eye) means the 9 high-precision codes cover 84% of volume.
- No colonoscopy codes — this is a surgical center, not GI.

---

## DUN

**Data:** 5 batches (batch 684–713), 147 charge lines
**Overall CPT Accuracy:** 88.4% (130 / 147)

### High Precision CPT Codes (100%, min 5 cases)

Covers only **15.0%** of all cases (22 / 147). Small sample size limits reliable precision estimates.

| CPT | Cases | Description |
|-----|-------|-------------|
| 01967 | 10 | C-section (planned) |
| 64488 | 6 | TAP block continuous |
| 01480 | 6 | Lower leg/ankle/foot |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00532 | 3 | 33.3% | 2x → 36569 (PICC line misclassified) |
| 01810 | 4 | 50.0% | 1x → 01830, 1x → 00400 |
| 00790 | 3 | 66.7% | 1x → 00750 |
| 00811 | 6 | 83.3% | 1x → 00812 |

### Key Takeaways

- Lowest accuracy of all groups analyzed (88.4%) but very small sample — only 5 batches returned data out of 30 requested.
- Highly diverse procedure mix — 48 distinct CPT codes across 147 cases, so most codes have <5 occurrences.
- **00532 → 36569** error is unique to DUN — the model predicts venipuncture (00532) when the actual procedure is PICC line insertion (36569).
- **01960 → 01967** (2 cases): model predicts vaginal hysterectomy anesthesia when it should be planned C-section.
- Need more batches to draw reliable conclusions for this group.

---

## KAP-ASC

**Data:** 50 batches (batch 192–241), 742 charge lines
**Overall CPT Accuracy:** 95.1% (706 / 742)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **91.2%** of all cases (677 / 742).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00160 | 99 | Nose/sinus |
| 00126 | 85 | Cataract surgery |
| 01630 | 42 | Shoulder open |
| 01400 | 39 | Knee arthroscopy |
| 01810 | 29 | Hip/femur procedures |
| 00811 | 22 | Colonoscopy diagnostic |
| 00402 | 12 | Breast biopsy (reconstructive) |
| 00813 | 8 | Colonoscopy screening→diagnostic |
| 00300 | 7 | Head/neck vessels |
| 00100 | 6 | Salivary gland/lip |
| 00731 | 5 | Upper GI endoscopy |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00170 | 157 | 97.5% | 2x → 00160, 2x → 00190 |
| 00812 | 134 | 97.0% | 4x → 00811 |
| 00320 | 32 | 96.9% | 1x → 00160 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00400 | 23 | 21.7% | **17x → 00402** (breast — wrong modifier code) |
| 01710 | 9 | 55.6% | 3x → 01810 |
| 00802 | 13 | 69.2% | 3x → 00400, 1x → 00402 |

### Key Takeaways

- Highest high-precision coverage at **91.2%** — very predictable ASC procedure mix.
- **00400 → 00402** is the dominant error (17 cases): model picks generic breast (00400) when coders want reconstructive breast (00402). Systematic, fixable with CPT instruction template.
- No colonoscopy confusion — 00811 is 100% and 00812 is 97%.

---

## KAP-CYP

**Data:** 50 batches (batch 200–249), 1,374 charge lines
**Overall CPT Accuracy:** 95.4% (1,311 / 1,374)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **66.5%** of all cases (914 / 1,374).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 01400 | 77 | Knee arthroscopy |
| 00952 | 73 | Lumbar epidural labor |
| 00140 | 64 | Eye (not lens) |
| 00142 | 40 | Lens procedures (cataract) |
| 00300 | 31 | Head/neck vessels |
| 00851 | 30 | Lower abdominal (hernia) |
| 00731 | 25 | Upper GI endoscopy |
| 00402 | 25 | Breast biopsy (reconstructive) |
| 00920 | 20 | Bladder |
| 00103 | 16 | Blepharoplasty |
| 00802 | 10 | Panniculectomy |
| 00813 | 10 | Colonoscopy screening→diagnostic |
| 00532 | 9 | Gastrostomy |
| 00790 | 9 | Intra-abdominal upper |
| 00100 | 5 | Salivary gland/lip |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00170 | 280 | 99.6% | 1x → 00126 |
| 00940 | 54 | 98.1% | 1x → 01965 |
| 00160 | 92 | 97.8% | 1x → 00300, 1x → 00170 |
| 00320 | 44 | 97.7% | 1x → 00300 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00400 | 46 | 63.0% | **13x → 00402**, 2x → 00940 |
| 00811 | 20 | 80.0% | 3x → 00812, 1x → 00731 |
| 00944 | 7 | 85.7% | 1x → 00840 |

### Key Takeaways

- Same **00400 → 00402** problem as KAP-ASC (13 cases) — likely a KAP-group-wide issue.
- Most diverse high-precision set — 19 codes covering 66.5%.
- 00812 → 00811 colonoscopy swap: 13 errors (91% precision), moderate issue.

---

## NTA-WGS

**Data:** 49 batches (batch 162–211), 660 charge lines
**Overall CPT Accuracy:** 96.8% (639 / 660)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **89.2%** of all cases (589 / 660).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00910 | 114 | Bladder procedures |
| 00902 | 107 | Anorectal surgery |
| 00921 | 57 | Bladder surgery |
| 01630 | 50 | Shoulder open |
| 00918 | 44 | Transurethral procedures |
| 01400 | 31 | Knee arthroscopy |
| 00920 | 22 | Bladder |
| 64447 | 15 | Femoral nerve block |
| 00912 | 12 | Transurethral resection |
| 00914 | 12 | Cystoscopy with stent |
| 01480 | 8 | Lower leg/ankle/foot |
| 01942 | 8 | CT/interventional radiology |
| 00300 | 5 | Head/neck vessels |
| 01710 | 5 | Elbow/forearm |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 01810 | 57 | 98.2% | 1x → 00400 |
| 64415 | 42 | 97.6% | 1x → 64447 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00860 | 8 | 0.0% | **7x → 00902**, 1x → 00300 — model always wrong |
| 00862 | 3 | 33.3% | 2x → 00918 |
| 00670 | 9 | 66.7% | 3x → 01942 |

### Key Takeaways

- Heavy urology center — 00910/00902/00921/00918/00920 dominate.
- **00860 at 0%** — all 8 predictions wrong (7x should be 00902). Same pattern as IAS-BHS. The model systematically misapplies 00860 (hip surgery) when it should be 00902 (anorectal).
- Very high coverage (89.2%) with high-precision codes.

---

## PCE-CAS

**Data:** 48 batches (batch 206–255), 693 charge lines
**Overall CPT Accuracy:** 95.1% (659 / 693)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **88.6%** of all cases (614 / 693).

| CPT | Cases | Precision | Description |
|-----|-------|-----------|-------------|
| 00142 | 529 | 99.8% | Lens procedures (cataract) |
| 00103 | 51 | 100% | Blepharoplasty |
| 00300 | 34 | 97.1% | Head/neck vessels |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00140 | 76 | 57.9% | **31x → 00142** (eye non-lens vs lens) |

### Key Takeaways

- Almost entirely an **eye surgery center** — 00142 (cataract) is 76% of volume.
- Only 6 distinct CPT codes across 693 charge lines.
- **00140 → 00142** is the sole significant error: model predicts generic eye (00140) when coders want lens-specific (00142). 31 cases, very systematic.

---

## PCE-WWMG

**Data:** 48 batches (batch 291–340), 1,307 charge lines
**Overall CPT Accuracy:** 88.1% (1,152 / 1,307)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **45.8%** of all cases (599 / 1,307).

| CPT | Cases | Precision | Description |
|-----|-------|-----------|-------------|
| 00731 | 237 | 100% | Upper GI endoscopy |
| 00813 | 69 | 100% | Colonoscopy screening→diagnostic |
| 00811 | 293 | 96.6% | Colonoscopy diagnostic |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00812 | 708 | 79.5% | **145x → 00811** (screening vs diagnostic) |

### Key Takeaways

- **Pure GI center** — only 4 CPT codes (00731, 00811, 00812, 00813).
- **00812 → 00811 is catastrophic**: 145 errors out of 708 predictions (20.5% error rate). The model over-predicts screening colonoscopy (00812) when the coder wants diagnostic (00811).
- This single error type accounts for **all** 155 errors in the group. Without it, accuracy would be 99.2%.
- Lowest high-precision coverage (45.8%) because the dominant code (00812) is unreliable.

---

## PRM-WHT

**Data:** 47 batches (batch 552–601), 850 charge lines
**Overall CPT Accuracy:** 99.2% (843 / 850)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **89.1%** of all cases (757 / 850).

| CPT | Cases | Precision | Description |
|-----|-------|-----------|-------------|
| 00142 | 713 | 100% | Lens procedures (cataract) |
| 00103 | 44 | 100% | Blepharoplasty |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00140 | 70 | 92.9% | 4x → 00142, 1x → 00103 |

### Key Takeaways

- **Best accuracy of all groups analyzed at 99.2%**.
- Almost entirely cataract surgery (00142 = 84% of volume).
- Only 7 total errors across 850 charge lines.
- Same 00140 → 00142 confusion as PCE-CAS but much milder (5.7% vs 42.1% error rate).

---

## IAS-MOR

**Data:** 46 batches (batch 379–424), 1,034 charge lines
**Overall CPT Accuracy:** 96.6% (999 / 1,034)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **78.8%** of all cases (815 / 1,034).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 124 | Upper GI endoscopy |
| 00790 | 67 | Intra-abdominal upper |
| 00840 | 62 | Intra-abdominal lower |
| 01967 | 48 | C-section (planned) |
| 00902 | 19 | Anorectal surgery |
| 01961 | 18 | C-section (emergency) |
| 01480 | 18 | Lower leg/ankle/foot |
| 64415 | 15 | Brachial plexus block |
| 00520 | 14 | Chest procedures (closed) |
| 00952 | 14 | Lumbar epidural labor |
| 01402 | 13 | Knee arthroplasty |
| 01638 | 12 | Shoulder arthroplasty |
| 01214 | 11 | Total hip arthroplasty |
| 64488 | 11 | TAP block continuous |
| 00120 | 8 | Eye (external) |
| 00160 | 7 | Nose/sinus |
| 00752 | 7 | Intra-abdominal hernia repair |
| 01810 | 7 | Hip/femur procedures |
| 01630 | 6 | Shoulder open |
| 00732 | 5 | Upper GI endoscopy (with biopsy) |
| 01210 | 5 | Hip open |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00812 | 180 | 98.3% | 3x → 00811 |
| 00813 | 123 | 96.7% | 2x → 00731, 1x → 00790, 1x → 00812 |
| 00830 | 21 | 95.2% | 1x → 00840 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00600 | 5 | 40.0% | 3x → 00670 |
| 01968 | 8 | 50.0% | **4x → 01967** (C-section+hysterectomy vs planned C-section) |
| 00400 | 21 | 85.7% | 2x → 01610, 1x → 00404 |
| 00811 | 87 | 86.2% | 11x → 00812 |

### Key Takeaways

- Large full-service hospital — diverse procedure mix (59 distinct CPTs).
- **01968 → 01967** at 50%: model confuses C-section+hysterectomy with planned C-section.
- **00600 → 00670** at 40%: cervical spine vs upper abdominal confusion.
- Colonoscopy 00811↔00812 moderate (14 swaps).

---

## TQA-ARSC

**Data:** 49 batches (batch 511–560), 475 charge lines
**Overall CPT Accuracy:** 95.6% (454 / 475)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **64.0%** of all cases (304 / 475).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 71 | Lens procedures (cataract) |
| 00813 | 17 | Colonoscopy screening→diagnostic |
| 64445 | 16 | Sciatic nerve block |
| 00952 | 15 | Lumbar epidural labor |
| 64447 | 14 | Femoral nerve block |
| 00731 | 12 | Upper GI endoscopy |
| 00170 | 12 | Intraoral procedures |
| 01400 | 11 | Knee arthroscopy |
| 64415 | 9 | Brachial plexus block |
| 01830 | 9 | Open femur/hip fracture |
| 00145 | 8 | Eye (vitrectomy) |
| 01470 | 7 | Upper leg embolectomy |
| 00160 | 6 | Nose/sinus |
| 01740 | 5 | Elbow arthroplasty |
| 00126 | 5 | Cataract surgery |
| 00400 | 5 | Integumentary (breast) |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00812 | 82 | 96.3% | 3x → 00811 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 01810 | 34 | 85.3% | 4x → 01830, 1x → 00400 |
| 00811 | 43 | 88.4% | 5x → 00812 |
| 00140 | 9 | 88.9% | 1x → 00142 |

### Key Takeaways

- Mixed ASC — eye, GI, ortho, OB all present.
- **01810 → 01830** (hip procedure variant confusion) is the top issue at 85.3%.
- Colonoscopy 00811↔00812: 8 swaps total, moderate.

---

## GOS-GOH

**Data:** 18 batches (batch 403–420), 358 charge lines
**Overall CPT Accuracy:** 89.9% (322 / 358)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **52.5%** of all cases (188 / 358).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 01967 | 23 | C-section (planned) |
| 01961 | 15 | C-section (emergency) |
| 00813 | 14 | Colonoscopy screening→diagnostic |
| 64488 | 10 | TAP block continuous |
| 00910 | 9 | Bladder procedures |
| 00902 | 8 | Anorectal surgery |
| 01480 | 7 | Lower leg/ankle/foot |
| 00811 | 7 | Colonoscopy diagnostic |
| 00940 | 7 | Vaginal delivery |
| 00732 | 6 | Upper GI (with biopsy) |
| 76937 | 6 | Ultrasound guidance |
| 00952 | 5 | Lumbar epidural labor |
| 01810 | 5 | Hip/femur procedures |
| 36620 | 5 | Arterial line |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00731 | 61 | 98.4% | 1x → 00813 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 01210 | 3 | 0.0% | 3x → 01230 |
| 01926 | 3 | 33.3% | 2x → 01924 |
| 00812 | 11 | 45.5% | **5x → 00811**, 1x → 00813 |
| 64448 | 4 | 50.0% | 2x → 64447 |
| 00918 | 11 | 72.7% | 3x → 00910 |

### Key Takeaways

- Full-service hospital — OB, GI, urology, ortho all present.
- **00812 at 45.5%** — worst colonoscopy screening precision of any group (5 of 11 wrong).
- **00918 → 00910** (transurethral vs bladder): 3 errors, systematic confusion.
- Only 18 batches returned data — relatively small sample.

---

## SIO-STL

**Data:** 50 batches (batch 445–494), 901 charge lines
**Overall CPT Accuracy:** 98.0% (883 / 901)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **87.9%** of all cases (792 / 901).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 01967 | 94 | C-section (planned) |
| 00170 | 86 | Intraoral procedures |
| 00790 | 64 | Intra-abdominal upper |
| 01480 | 42 | Lower leg/ankle/foot |
| 00731 | 39 | Upper GI endoscopy |
| 00532 | 20 | Gastrostomy |
| 00732 | 18 | Upper GI (with biopsy) |
| 00910 | 18 | Bladder procedures |
| 00813 | 18 | Colonoscopy screening→diagnostic |
| 01965 | 17 | C-section (vaginal birth after) |
| 01968 | 15 | C-section + hysterectomy |
| 00520 | 15 | Chest procedures (closed) |
| 00410 | 13 | Heart catheterization |
| 00400 | 12 | Integumentary (breast) |
| 01922 | 12 | CT/interventional radiology |
| 00537 | 12 | Open heart (cardiac bypass) |
| 00952 | 10 | Lumbar epidural labor |
| 00940 | 9 | Vaginal delivery |
| 00812 | 9 | Colonoscopy screening |
| 00851 | 8 | Lower abdominal (hernia) |
| 64445 | 7 | Sciatic nerve block |
| 36620 | 6 | Arterial line |
| 01844 | 6 | Vascular lower leg |
| 00902 | 5 | Anorectal surgery |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 64488 | 98 | 99.0% | 1x → 64486 |
| 01961 | 42 | 97.6% | 1x → 01967 |
| 00840 | 97 | 95.9% | 2x → 00851 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00800 | 5 | 40.0% | 1x → 01250, 1x → 00840 |
| 00756 | 3 | 66.7% | 1x → 00790 |
| 01210 | 4 | 75.0% | 1x → 01230 |
| 01470 | 9 | 77.8% | 1x → 00400 |

### Key Takeaways

- Sister group to SIO-PSS — second best accuracy overall at **98.0%**.
- **64488 (TAP block)** is a huge volume code here (98 cases, 99% precision).
- Very few problem codes — 00800 (panniculectomy) is the worst but only 5 cases.
- No significant colonoscopy issues — 00812 is 100%, 00811 is 92.3%.

---

## STA-HLX

**Data:** 14 batches (batch 255–268), 183 charge lines
**Overall CPT Accuracy:** 96.7% (177 / 183)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **80.9%** of all cases (148 / 183).

| CPT | Cases | Precision | Description |
|-----|-------|-----------|-------------|
| 00142 | 148 | 99.3% | Lens procedures (cataract) |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00140 | 31 | 83.9% | 4x → 00142, 1x → 00144 |

### Key Takeaways

- **Pure cataract center** — 00142 is 81% of volume. Only 3 distinct CPT codes.
- Same 00140 → 00142 confusion seen in PCE-CAS and PRM-WHT.
- Small sample (14 batches, 183 cases).

---

## WPA

**Data:** 49 batches (batch 295–344), 712 charge lines
**Overall CPT Accuracy:** 98.6% (702 / 712)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **94.0%** of all cases (669 / 712).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 328 | Lens procedures (cataract) |
| 00170 | 96 | Intraoral procedures |
| 00812 | 63 | Colonoscopy screening |
| 00103 | 24 | Blepharoplasty |
| 00731 | 21 | Upper GI endoscopy |
| 00126 | 20 | Cataract surgery |
| 00145 | 18 | Eye (vitrectomy) |
| 00300 | 14 | Head/neck vessels |
| 00813 | 14 | Colonoscopy screening→diagnostic |
| 00160 | 9 | Nose/sinus |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00140 | 62 | 95.2% | 2x → 00142, 1x → 00144 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00120 | 5 | 60.0% | 2x → 00124 |
| 00811 | 16 | 68.8% | 5x → 00812 |

### Key Takeaways

- **Highest high-precision coverage at 94.0%** — best of all groups.
- Second-best accuracy at **98.6%** behind PRM-WHT.
- Eye + ENT + GI ASC — very predictable procedure mix.
- **00811 at 68.8%** is the main weak spot (5 of 16 wrong → 00812).

---

## GII-ASC

**Data:** 50 batches (batch 171–220), 1,151 charge lines
**Overall CPT Accuracy:** 97.0% (1,116 / 1,151)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **70.5%** of all cases (811 / 1,151).

| CPT | Cases | Precision | Description |
|-----|-------|-----------|-------------|
| 00731 | 174 | 100% | Upper GI endoscopy |
| 00813 | 161 | 100% | Colonoscopy screening→diagnostic |
| 00812 | 476 | 99.6% | Colonoscopy screening (2x → 00811) |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00811 | 340 | 90.3% | 33x → 00812 |

### Key Takeaways

- **Pure GI center** — only 4 CPT codes, all colonoscopy/endoscopy.
- Much better 00812 precision than PCE-WWMG (99.6% vs 79.5%) despite same procedure mix.
- **00811 at 90.3%** — 33 cases changed to 00812. This is the reverse of PCE-WWMG (where 00812 was the problem).
- High volume (1,151 charge lines) makes these precision numbers very reliable.

---

## GAP

**Data:** 50 batches (batch 706–755), 1,228 charge lines
**Overall CPT Accuracy:** 96.6% (1,186 / 1,228)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **80.0%** of all cases (983 / 1,228).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 232 | Lens procedures (cataract) |
| 64447 | 64 | Femoral nerve block |
| 01967 | 60 | C-section (planned) |
| 00170 | 44 | Intraoral procedures |
| 01214 | 43 | Total hip arthroplasty |
| 01480 | 39 | Lower leg/ankle/foot |
| 64488 | 34 | TAP block continuous |
| 00790 | 31 | Intra-abdominal upper |
| 00410 | 29 | Heart catheterization |
| 01810 | 27 | Hip/femur procedures |
| 64415 | 26 | Brachial plexus block |
| 01961 | 24 | C-section (emergency) |
| 01630 | 17 | Shoulder open |
| 00731 | 17 | Upper GI endoscopy |
| 00811 | 17 | Colonoscopy diagnostic |
| 00532 | 15 | Gastrostomy |
| 01922 | 15 | CT/interventional radiology |
| 01638 | 15 | Shoulder arthroplasty |
| 00400 | 14 | Integumentary (breast) |
| 01830 | 14 | Open femur/hip fracture |
| 00160 | 14 | Nose/sinus |
| 00910 | 14 | Bladder procedures |
| 00126 | 12 | Cataract surgery |
| 00140 | 12 | Eye (not lens) |
| 00912 | 11 | Transurethral resection |
| 64473 | 10 | Lumbar plexus block |
| 01942 | 8 | CT/interventional radiology |
| 00952 | 8 | Lumbar epidural labor |
| 64450 | 6 | Other peripheral nerve block |
| 01482 | 5 | Lower leg (open) |
| 01965 | 5 | C-section (vaginal birth after) |
| 31500 | 5 | Intubation/tracheostomy |
| 01230 | 5 | Upper leg (femur) |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 01402 | 68 | 98.5% | 1x → 01400 |
| 01400 | 23 | 95.7% | 1x → 01392 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00620 | 5 | 0.0% | 3x → 01942, 2x → 01941 |
| 00920 | 3 | 33.3% | 2x → 00902 |
| 00600 | 7 | 57.1% | 3x → 00670 |
| 01210 | 6 | 66.7% | 2x → 01230 |
| 00670 | 28 | 85.7% | 2x → 01941, 1x → 00300, 1x → 01942 |

### Key Takeaways

- **Most diverse high-precision set: 35 codes at 100%** covering 80% of volume.
- Full-service group — eye, OB, ortho, GI, urology, cardiac, nerve blocks all represented.
- **00620 at 0%**: model always wrong — predicts cervical spine when it should be interventional radiology (01942/01941).
- **00600 → 00670** (cervical spine → upper abdominal): same pattern as IAS-MOR.
- Very strong nerve block precision: 64447 (64 cases), 64488 (34), 64415 (26), 64473 (10) all at 100%.

---

## CHA-HDH

**Data:** 46 batches (batch 428–477), 901 charge lines
**Overall CPT Accuracy:** 91.9% (828 / 901)

### High Precision CPT Codes (>=95%, min 5 cases)

Covers **60.4%** of all cases (544 / 901).

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 64 | Upper GI endoscopy |
| 00142 | 61 | Lens procedures (cataract) |
| 64488 | 33 | TAP block continuous |
| 00790 | 27 | Intra-abdominal upper |
| 64473 | 24 | Lumbar plexus block |
| 01967 | 23 | C-section (planned) |
| 00840 | 19 | Intra-abdominal lower |
| 64447 | 17 | Femoral nerve block |
| 00952 | 16 | Lumbar epidural labor |
| 01961 | 16 | C-section (emergency) |
| 36620 | 15 | Arterial line |
| 01922 | 15 | CT/interventional radiology |
| 01214 | 14 | Total hip arthroplasty |
| 01400 | 12 | Knee arthroscopy |
| 00902 | 11 | Anorectal surgery |
| 01402 | 11 | Knee arthroplasty |
| 64415 | 8 | Brachial plexus block |
| 00940 | 8 | Vaginal delivery |
| 00126 | 8 | Cataract surgery |
| 00170 | 8 | Intraoral procedures |
| 00300 | 7 | Head/neck vessels |
| 76937 | 7 | Ultrasound guidance |
| 00851 | 7 | Lower abdominal (hernia) |
| 31500 | 5 | Intubation/tracheostomy |
| 00914 | 5 | Cystoscopy with stent |
| 01830 | 5 | Open femur/hip fracture |
| 01965 | 5 | C-section (vaginal birth after) |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00812 | 71 | 97.2% | 2x → 00811 |
| 00813 | 22 | 95.5% | 1x → 00731 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 64449 | 3 | 0.0% | 3x → 64450 (continuous vs single-shot lumbar plexus) |
| 00862 | 7 | 14.3% | **5x → 00918**, 1x → 00910 (renal→transurethral confusion) |
| 00532 | 8 | 25.0% | **4x → 36410**, 2x → 36573 (venipuncture vs blood draw/PICC) |
| 01926 | 3 | 33.3% | 2x → 01924 |
| 01360 | 3 | 33.3% | 2x → 01230 |
| 00811 | 65 | 72.3% | **18x → 00812** (diagnostic vs screening colonoscopy) |
| 00910 | 38 | 84.2% | 6x → 00918 (bladder vs transurethral) |
| 00918 | 19 | 89.5% | 2x → 00910 |

### Key Takeaways

- Full-service hospital with very diverse mix — 89 distinct CPT codes.
- **00862 at 14.3%**: model predicts renal procedures when coders want transurethral (00918/00910). Systematic confusion.
- **00532 at 25.0%**: model predicts venipuncture but actual procedures are blood draws (36410) or PICC lines (36573). Same pattern as DUN.
- **00910↔00918 bidirectional confusion**: 6x 00910→00918 and 2x 00918→00910. Bladder vs transurethral is a consistent weak spot.
- **00811 at 72.3%**: 18 colonoscopy errors — third worst after PCE-WWMG and GOS-GOH.
- Nerve blocks are solid: 64488 (33), 64473 (24), 64447 (17), 64415 (8) all at 100%.

---

## PCE-PMC

**Data:** 49 batches, 1,367 charge lines
**Overall CPT Accuracy:** 94.9% (1,297 / 1,367) | **HP Coverage:** 80.8%

Pure GI center (4 codes). 00812 at 97.3% (879 cases), 00813 100% (95), 00731 100% (130). Problem: 00811 at 82.9% (45x → 00812).

---

## INJE-CLIFW

**Data:** 40 batches, 1,151 charge lines
**Overall CPT Accuracy:** 97.9% (1,127 / 1,151) | **HP Coverage:** 93.7%

Eye center. 00142 at 99.6% (1,079 cases). Problem: 00140 at 71.4% (18x → 00142).

---

## INJE-CSCG

**Data:** 32 batches, 632 charge lines
**Overall CPT Accuracy:** 85.3% (539 / 632) | **HP Coverage:** 81.0%

Eye center. 00142 at 99.8% (512 cases). Problem: **00140 at 22.9%** (92x → 00142) — worst 00140 precision of any group.

---

## INJE-NRSC

**Data:** 46 batches, 288 charge lines
**Overall CPT Accuracy:** 97.9% (282 / 288) | **HP Coverage:** 82.6%

Ortho center. 11 codes at 100%: 64415 (46), 64447 (32), 01400 (31), 01402 (30), 01630 (26), 01214 (17), 01830 (17), 00170 (13), 00160 (12), 01638 (8), 01480 (6). Problem: 01810 at 86.4%.

---

## INJE-CLIK

**Data:** 24 batches, 570 charge lines
**Overall CPT Accuracy:** 96.7% (551 / 570) | **HP Coverage:** 93.7%

Eye center. 00142 at 97.6% (534 cases). Problem: 00140 at 82.4% (6x → 00142).

---

## INJE-CSC

**Data:** 14 batches, 340 charge lines
**Overall CPT Accuracy:** 99.1% (337 / 340) | **HP Coverage:** 91.2%

Eye center. 00142 at 99.7% (310 cases). Only 3 errors total.

---

## INJE-CHB

**Data:** 26 batches, 159 charge lines
**Overall CPT Accuracy:** 95.0% (151 / 159) | **HP Coverage:** 31.4%

Mixed ASC — small volume, diverse mix. 6 HP codes: 01480 (18), 64445 (11), 00731 (6), 64488 (5), 01961 (5), 01402 (5). Low HP coverage due to many codes under 5 cases.

---

## LOV

**Data:** 40 batches, 1,038 charge lines
**Overall CPT Accuracy:** 99.4% (1,032 / 1,038) | **HP Coverage:** 97.2%

Eye center — **third best accuracy overall**. 00142 100% (875 cases), 00103 100% (17), 00140 96.6% (117). Only 6 errors across 1,038 cases.

---

## APO-ORA

**Data:** 26 batches, 438 charge lines
**Overall CPT Accuracy:** 92.7% (406 / 438) | **HP Coverage:** 76.5%

Ortho center. 12 HP codes. Problem: **01938 at 0%** (11x → 01940), 64999 at 0% (4x → 64473), 01710 at 60%.

---

## APO-UTP

**Data:** 44 batches, 1,155 charge lines
**Overall CPT Accuracy:** 94.0% (1,086 / 1,155) | **HP Coverage:** 69.2%

Full-service. 34 HP codes at 100%. Strong nerve blocks: 64447 (88, 97.7%), 64473 (71, 97.2%), 64415 (29, 100%). Problems: 00600 0%, 00862 0%, 01968 0%, 01462 16.7%.

---

## APO-UPM

**Data:** 48 batches, 524 charge lines
**Overall CPT Accuracy:** 93.3% (489 / 524) | **HP Coverage:** 63.4%

Mixed. 15 HP codes. 00812 at 99.4% (163 cases). Problems: 01850 at 0% (4x → 36410), 00830 77.8%.

---

## APO-CVO

**Data:** 40 batches, 549 charge lines
**Overall CPT Accuracy:** 98.7% (542 / 549) | **HP Coverage:** 88.7%

Ortho center. 15 HP codes — 64415 (78), 01402 (64), 01400 (55), 01630 (54), 01940 (38), 01214 (34). Only problem: 01710 at 81.8%.

---

## ANA-ORA

**Data:** 25 batches, 293 charge lines
**Overall CPT Accuracy:** 93.9% (275 / 293) | **HP Coverage:** 78.8%

Ortho center. 9 HP codes. Problem: **64450 at 0%** (7x → 64473), 01710 at 60%.

---

## ANA-CVO

**Data:** 38 batches, 522 charge lines
**Overall CPT Accuracy:** 99.0% (517 / 522) | **HP Coverage:** 95.2%

Ortho center — **excellent**. 18 HP codes covering 95.2%. Only 5 errors across 522 cases.

---

## EAP-CMI

**Data:** 24 batches, 145 charge lines
**Overall CPT Accuracy:** 96.6% (140 / 145) | **HP Coverage:** 60.7%

Mixed ASC. 8 HP codes. Small volume. Problem: 01938 at 84.6%.

---

## EAP-JSC

**Data:** 9 batches, 56 charge lines
**Overall CPT Accuracy:** 96.4% (54 / 56) | **HP Coverage:** 39.3%

Ortho center. Small sample — 3 HP codes: 01630 (8), 01214 (8), 01402 (6).

---

## PRI-RSL

**Data:** 30 batches, 127 charge lines
**Overall CPT Accuracy:** 98.4% (125 / 127) | **HP Coverage:** 92.9%

Eye center. 00103 (52), 00145 (46), 00142 (20) all 100%. Problem: 00140 at 60% (only 5 cases).

---

## PRI-CRS

**Data:** 38 batches, 333 charge lines
**Overall CPT Accuracy:** 97.9% (326 / 333) | **HP Coverage:** 85.3%

Ortho center. 12 HP codes — 01400 (53), 64447 (36), 01402 (33), 01938 (32), 64473 (31), 01214 (23), 01630 (21), 64415 (13). Problem: 64450 at 80% (3x → 64473).

---

## NTA-ASCOV

**Data:** 43 batches, 622 charge lines
**Overall CPT Accuracy:** 99.0% (616 / 622) | **HP Coverage:** 86.3%

Mixed — eye + ortho + urology. 18 HP codes. 00142 (118), 00145 (97), 01402 (41), 01400 (38). Only 6 errors.

---

## MKI

**Data:** 47 batches, 546 charge lines
**Overall CPT Accuracy:** 96.7% (528 / 546) | **HP Coverage:** 87.4%

Ortho center. 14 HP codes — 64415 (105), 01810 (91, 96.7%), 64447 (63), 01400 (56, 98.2%), 01630 (43). Problem: **01710 at 33.3%** (4x → 01810, 3x → 01714).

---

## PAC-MHI

**Data:** 28 batches, 1,125 charge lines
**Overall CPT Accuracy:** 95.5% (1,074 / 1,125) | **HP Coverage:** 71.7%

Full-service hospital. 22 HP codes — 00812 (258, 96.9%), 00731 (113), 01967 (51), 00790 (45), 00813 (43). Problems: 01968 at 61.5% (4x → 01967), 00534 at 60%, 00811 at 87.1%.

---

## AHG

**Data:** 24 batches, 525 charge lines
**Overall CPT Accuracy:** 91.8% (482 / 525) | **HP Coverage:** 53.0%

Mixed. 14 HP codes — 00813 (79), 00731 (48), 00902 (25), 00300 (22). Problem: **01992 at 34.8%** (13x → 01938) — systematic confusion between pain procedure codes.

---

## IAS-FVO

**Data:** 50 batches, 1,110 charge lines
**Overall CPT Accuracy:** 97.7% (1,084 / 1,110) | **HP Coverage:** 91.1%

Ortho center — **excellent**. 17 HP codes — 01400 (149), 64415 (144), 64447 (116, 99.1%), 01402 (109), 01630 (109), 64473 (70, 98.6%), 01214 (63), 01810 (89, 98.9%), 01830 (47, 97.9%). Problem: 64445 at 69.2% (4x → 64473).

---

## IAS-BMH

**Data:** 19 batches, 346 charge lines
**Overall CPT Accuracy:** 91.3% (316 / 346) | **HP Coverage:** 63.6%

Mixed hospital. 15 HP codes. Problem: **00811 at 40.7%** (16x → 00812) — worst colonoscopy diagnostic precision. Also 01961 at 57.1% (3x → 01968).

---

## TAN-ESC

**Data:** 50 batches, 861 charge lines
**Overall CPT Accuracy:** 99.2% (854 / 861) | **HP Coverage:** 98.4%

Eye center — **tied for best accuracy (99.2%) and highest HP coverage (98.4%)**. 00142 (579), 00140 (123, 96.7%), 00103 (96), 00144 (29), 00145 (20). Only 7 errors.

---

## RIV (combined)

**Data:** 10 batches, 214 charge lines
**Overall CPT Accuracy:** 96.7% (207 / 214) | **HP Coverage:** 67.3%

Combined across all RIV providers. 9 HP codes — 00812 (46, 95.7%), 64488 (23), 00731 (19), 00811 (15), 00790 (11), 00840 (9), 64447 (8), 01480 (7), 00918 (6). Small sample.

---

## UNI-GOLD

**Data:** 45 batches (batch 979–1028), 1,132 charge lines
**Overall CPT Accuracy:** 96.1% (1,088 / 1,132) | **HP Coverage:** 79.1%

### High Precision CPT Codes (>=95%, min 5 cases)

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 01630 | 168 | Shoulder open |
| 64415 | 130 | Brachial plexus block |
| 64447 | 53 | Femoral nerve block |
| 01202 | 49 | Hip arthroplasty (posterior) |
| 00630 | 40 | Lumbar spine |
| 01638 | 27 | Shoulder arthroplasty |
| 00670 | 15 | Upper abdominal/spine |
| 01472 | 9 | Tibia/fibula repair |
| 01214 | 8 | Total hip arthroplasty |
| 01464 | 8 | Ankle repair |
| 01714 | 8 | Elbow arthroplasty |
| 01402 | 6 | Knee arthroplasty |
| 01320 | 6 | Knee disarticulation |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 01400 | 295 | 98.6% | 2x → 01320 |
| 01830 | 73 | 98.6% | 1x → 01810 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 01710 | 25 | 52.0% | 5x → 01714, 1x → 01712 |
| 01250 | 9 | 66.7% | 2x → 01320 |
| 00600 | 13 | 69.2% | 2x → 00670 |
| 01610 | 11 | 72.7% | 2x → 01630 |
| 01810 | 59 | 84.7% | 5x → 01830 |

### Key Takeaways

- Pure orthopedic center — shoulder (01630, 01638) + knee (01400, 01402) + hip (01830, 01214) dominate.
- **01400 at 98.6% with 295 cases** — largest single high-precision code by volume.
- 01710 (elbow) at 52% is the main weak spot — confused with 01714/01712 variants.

---

## UNI-ROB

**Data:** 31 batches (batch 1480–1529), 2,206 charge lines
**Overall CPT Accuracy:** 91.3% (2,014 / 2,206) | **HP Coverage:** 74.2%

### High Precision CPT Codes (>=95%, min 5 cases)

**100% precision:**

| CPT | Cases | Description |
|-----|-------|-------------|
| 01630 | 154 | Shoulder open |
| 00813 | 118 | Colonoscopy screening→diagnostic |
| 64415 | 107 | Brachial plexus block |
| 00731 | 98 | Upper GI endoscopy |
| 64447 | 89 | Femoral nerve block |
| 01214 | 76 | Total hip arthroplasty |
| 01202 | 46 | Hip arthroplasty (posterior) |
| 01638 | 38 | Shoulder arthroplasty |
| 01940 | 23 | CT/interventional radiology |
| 01939 | 20 | Vertebroplasty/kyphoplasty |
| 00902 | 16 | Anorectal surgery |
| 00120 | 12 | Eye (external) |
| 00160 | 11 | Nose/sinus |
| 01380 | 9 | Knee disarticulation (open) |
| 64445 | 8 | Sciatic nerve block |
| 01392 | 8 | Knee ligament repair |
| 01250 | 8 | Upper arm/elbow |
| 01941 | 5 | Percutaneous spine |
| 00402 | 5 | Breast biopsy (reconstructive) |

**95–99% precision:**

| CPT | Cases | Precision | Notes |
|-----|-------|-----------|-------|
| 00812 | 402 | 98.8% | 3x → 00811 |
| 01400 | 246 | 98.0% | 2x → 01392, 1x → 01402 |
| 01402 | 93 | 97.8% | 1x → 01400 |
| 01480 | 45 | 95.6% | 1x → 01464 |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00600 | 20 | 20.0% | 7x → 00670 |
| 01938 | 67 | 25.4% | **25x → 01992**, 5x → 01940 (pain procedure confusion) |
| 01470 | 14 | 42.9% | 5x → 01464 |
| 01937 | 38 | 55.3% | 11x → 01992 |
| 01810 | 73 | 72.6% | 7x → 01830, 4x → 00400 |
| 01710 | 26 | 76.9% | 2x → 01714 |

### Key Takeaways

- Largest volume of the three UNI groups — 2,206 charge lines.
- Full-service: ortho + GI + pain management + eye + ENT.
- **01938↔01992 confusion** is the #1 error source (36 cases) — pain procedure code mix-ups.
- 00812 at 98.8% with 402 cases — excellent colonoscopy screening precision.
- 00600 → 00670 (cervical→lumbar spine): same pattern as IAS-MOR, GAP.

---

## UNI-RSC

**Data:** 32 batches (batch 1405–1454), 1,000 charge lines
**Overall CPT Accuracy:** 69.2% (692 / 1,000) | **HP Coverage:** 31.0%

### High Precision CPT Codes (>=95%, min 5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 166 | Lens procedures (cataract) |
| 01939 | 27 | Vertebroplasty/kyphoplasty |
| 01472 | 16 | Tibia/fibula repair |
| 00160 | 16 | Nose/sinus |
| 64447 | 11 | Femoral nerve block |
| 01400 | 11 | Knee arthroscopy |
| 00952 | 10 | Lumbar epidural labor |
| 64445 | 10 | Sciatic nerve block |
| 00920 | 9 | Bladder |
| 00145 | 9 | Eye (vitrectomy) |
| 01830 | 7 | Open femur/hip fracture |
| 00921 | 7 | Bladder surgery |
| 64415 | 6 | Brachial plexus block |
| 01630 | 5 | Shoulder open |

### Problem Codes

| CPT | Cases | Precision | Primary Error |
|-----|-------|-----------|---------------|
| 00630 | 23 | 0.0% | 7x → 00300, 4x → 01938, 2x → 01941 (all wrong) |
| 01320 | 20 | 5.0% | **13x → 01991** |
| 01942 | 20 | 10.0% | 6x → 01941, 5x → 00300 |
| 01938 | 230 | 37.4% | **73x → 01992** (massive pain procedure confusion) |
| 01937 | 50 | 48.0% | 15x → 01992 |
| 00300 | 46 | 67.4% | 4x → 01941, 3x → 01991, 2x → 01992 |
| 01940 | 24 | 70.8% | 4x → 01992 |

### Key Takeaways

- **Worst accuracy of ALL groups at 69.2%** — driven almost entirely by pain procedure code confusion.
- **01938 at 37.4% with 230 cases** — the model predicts lumbar epidural injection when coders want 01992 (pain management). This single code accounts for 144 of the 308 total errors.
- 01937, 01320, 01942 also heavily confused with 01991/01992.
- The eye/ortho codes are fine (00142 100%, nerve blocks 100%) — it's specifically the pain/spine codes that are broken.
- HP coverage only 31% — too many error-prone codes dominating the mix.

---

## No Data Groups

IAS-WAG, OHJ-MEC, MED, TQA-CFM, TQA-GCCA, GOS-GOSC, INJE-SOU, INJE-CFS, INJE-LAF — returned no data from the API for the batch ranges queried.
