# CPT Precision Analysis — March + April 2026 (All Batches)

**Methodology:** Every billed CPT line counted as 1 data point (primary + nerve blocks + A-lines + ultrasound guidance + central lines + epidurals + every other secondary/additional line). Ground truth = `allCharges=true` API response. Predictions deemed wrong when the change log records `CPT changed from X to Y` matching that account+line.

**Coverage:** 1481 batches with data (out of 1848 requested) across 54 groups.


## Summary Across All Groups

HP Coverage = % of billed lines that could be auto-coded by codes with ≥95% precision (and ≥5 cases). 100% Coverage = subset that's never been wrong.

| Group | Lines | Changes | Accuracy | HP Coverage | 100% Coverage |
|-------|-------|---------|----------|-------------|---------------|
| UNI-INTEG | 6061 | 684 | 88.71% | 59.1% (3581/6061) | 25.9% (1569/6061) |
| SIO-PSS | 1965 | 84 | 95.73% | 79.9% (1571/1965) | 60.8% (1195/1965) |
| IAS-BHS | 1352 | 113 | 91.64% | 55.1% (745/1352) | 46.2% (625/1352) |
| GOS-GOH | 1256 | 132 | 89.49% | 52.4% (658/1256) | 36.7% (461/1256) |
| APO-UTP | 1181 | 104 | 91.19% | 66.9% (790/1181) | 43.2% (510/1181) |
| SIO-STL | 1155 | 60 | 94.81% | 55.0% (635/1155) | 40.4% (467/1155) |
| PCE-PMC | 1146 | 73 | 93.63% | 82.1% (941/1146) | 9.2% (105/1146) |
| KAP-CYP | 1140 | 64 | 94.39% | 64.8% (739/1140) | 29.4% (335/1140) |
| GII-ASC | 1032 | 36 | 96.51% | 72.3% (746/1032) | 29.4% (303/1032) |
| IAS-MOR | 1005 | 61 | 93.93% | 73.7% (741/1005) | 31.5% (317/1005) |
| IAS-FVO | 947 | 32 | 96.62% | 87.9% (832/947) | 59.0% (559/947) |
| PCE-WWMG | 947 | 160 | 83.10% | 22.9% (217/947) | 22.9% (217/947) |
| GAP | 938 | 44 | 95.31% | 75.3% (706/938) | 69.6% (653/938) |
| CHA-HDH | 765 | 94 | 87.71% | 57.8% (442/765) | 48.4% (370/765) |
| INJE-CLIFW | 765 | 24 | 96.86% | 93.3% (714/765) | 0.0% (0/765) |
| PRM-WHT | 689 | 10 | 98.55% | 89.4% (616/689) | 89.4% (616/689) |
| PAC-MHI | 666 | 34 | 94.89% | 72.4% (482/666) | 44.0% (293/666) |
| IAS-BMH | 618 | 55 | 91.10% | 55.0% (340/618) | 49.4% (305/618) |
| TAN-ESC | 614 | 9 | 98.53% | 82.2% (505/614) | 82.2% (505/614) |
| KAP-ASC | 613 | 39 | 93.64% | 71.3% (437/613) | 48.3% (296/613) |
| WPA | 559 | 10 | 98.21% | 83.7% (468/559) | 83.7% (468/559) |
| APO-ORA | 549 | 44 | 91.99% | 67.6% (371/549) | 52.8% (290/549) |
| LOV | 549 | 6 | 98.91% | 86.7% (476/549) | 86.7% (476/549) |
| NTA-WGS | 545 | 27 | 95.05% | 86.4% (471/545) | 54.9% (299/545) |
| INJE-CSCG | 542 | 93 | 82.84% | 77.9% (422/542) | 0.0% (0/542) |
| GOS-GOSC | 508 | 104 | 79.53% | 27.4% (139/508) | 2.0% (10/508) |
| PRM | 502 | 39 | 92.23% | 55.2% (277/502) | 55.2% (277/502) |
| PCE-CAS | 475 | 35 | 92.63% | 87.2% (414/475) | 8.4% (40/475) |
| RIV | 449 | 23 | 94.88% | 43.2% (194/449) | 33.9% (152/449) |
| MKI | 389 | 22 | 94.34% | 71.5% (278/389) | 44.0% (171/389) |
| INJE-CLIK | 388 | 19 | 95.10% | 92.8% (360/388) | 0.0% (0/388) |
| AHG | 380 | 52 | 86.32% | 42.4% (161/380) | 42.4% (161/380) |
| DUN | 373 | 46 | 87.67% | 47.2% (176/373) | 28.4% (106/373) |
| ANA-ORA | 348 | 18 | 94.83% | 80.2% (279/348) | 60.9% (212/348) |
| TQA-ARSC | 344 | 26 | 92.44% | 42.4% (146/344) | 42.4% (146/344) |
| APO-UPM | 264 | 35 | 86.74% | 47.7% (126/264) | 20.5% (54/264) |
| INJE-NRSC | 249 | 6 | 97.59% | 83.1% (207/249) | 83.1% (207/249) |
| INJE-CSC | 248 | 6 | 97.58% | 90.7% (225/248) | 0.0% (0/248) |
| STA-HLX | 236 | 9 | 96.19% | 82.6% (195/236) | 3.0% (7/236) |
| GAP-UMSC | 220 | 9 | 95.91% | 70.5% (155/220) | 45.9% (101/220) |
| IAS-NSC | 215 | 12 | 94.42% | 80.9% (174/215) | 80.9% (174/215) |
| EAP-CMI | 176 | 6 | 96.59% | 60.8% (107/176) | 60.8% (107/176) |
| PRI-CRS | 176 | 6 | 96.59% | 90.3% (159/176) | 75.6% (133/176) |
| NTA-ASCOV | 151 | 5 | 96.69% | 70.2% (106/151) | 70.2% (106/151) |
| EAP-TIN | 150 | 5 | 96.67% | 74.7% (112/150) | 60.0% (90/150) |
| APO-CVO | 117 | 7 | 94.02% | 49.6% (58/117) | 49.6% (58/117) |
| ANA-CVO | 112 | 5 | 95.54% | 45.5% (51/112) | 45.5% (51/112) |
| EAP-SCA | 110 | 12 | 89.09% | 48.2% (53/110) | 48.2% (53/110) |
| INJE-CHB | 109 | 9 | 91.74% | 14.7% (16/109) | 14.7% (16/109) |
| EAP-JSC | 98 | 4 | 95.92% | 74.5% (73/98) | 51.0% (50/98) |
| EAP-PSC | 77 | 3 | 96.10% | 66.2% (51/77) | 66.2% (51/77) |
| PRI-RSL | 72 | 2 | 97.22% | 91.7% (66/72) | 91.7% (66/72) |
| ARKMETH | 53 | 12 | 77.36% | 0.0% (0/53) | 0.0% (0/53) |
| AIP | 47 | 4 | 91.49% | 55.3% (26/47) | 55.3% (26/47) |


---

## UNI-INTEG

**Lines:** 6061  **CPT Changes:** 684  **Accuracy:** 88.71%

**HP Coverage (≥95% codes, ≥5 cases):** 3581/6061 = **59.1%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 1569/6061 = **25.9%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 36620 | 196 | Arterial line |
| 01922 | 162 | CT/interventional radiology |
| 00537 | 152 | Open heart bypass |
| 01402 | 143 | Knee arthroplasty |
| 76937 | 98 | Ultrasound guidance |
| 00670 | 86 | Upper abdominal/spine |
| 00320 | 75 | Anterior neck |
| 01967 | 54 | C-section (planned) |
| 01214 | 43 | Total hip arthroplasty |
| 00912 | 42 | Transurethral resection |
| 93312 | 40 | TEE |
| 64468 | 39 |  |
| 00120 | 34 | Eye (external) |
| 00410 | 33 | Heart catheterization |
| 00830 | 32 | Hernia lower abd |
| 00534 | 29 | Cardioversion |
| 00635 | 28 | Diagnostic lumbar |
| 01638 | 28 | Shoulder arthroplasty |
| 00797 | 27 | Bariatric morbid |
| 64466 | 27 |  |
| 64416 | 20 | Brachial plexus continuous |
| 00541 | 19 |  |
| 64415 | 17 | Brachial plexus block |
| 00865 | 17 | Prostatectomy |
| 00567 | 16 | CABG w CPB |
| 01270 | 13 | Upper leg vessels |
| 36556 | 12 | Central venous catheter |
| 00906 | 12 | Vulvectomy |
| 01230 | 11 | Upper leg (femur) |
| 00126 | 10 | Cataract surgery |
| 00868 | 9 | Renal transplant |
| 64469 | 9 |  |
| 01232 | 9 | Femur amputation |
| 01938 | 8 | Pain mgmt epidural injection |
| 01380 | 7 | Knee disarticulation (open) |
| 00930 | 6 | Orchiopexy |
| 00220 | 6 | Cranial open |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00731 | 378 | 97.6% | 9x → 00732 | Upper GI endoscopy |
| 00812 | 317 | 95.6% | 14x → 00811 | Colonoscopy screening |
| 00902 | 173 | 98.8% | 1x → 00300, 1x → 00790 | Anorectal surgery |
| 00813 | 117 | 98.3% | 2x → 00731 | Colonoscopy screening→diagnostic |
| 00520 | 112 | 98.2% | 2x → 00541 | Chest procedures (closed) |
| 00142 | 108 | 95.4% | 3x → 00300, 1x → 00160, 1x → 00400 | Lens procedures (cataract) |
| 00104 | 107 | 96.3% | 4x → 01924 | ECT |
| 00732 | 103 | 99.0% | 1x → 00731 | Upper GI w biopsy |
| 00402 | 98 | 95.9% | 2x → 00400, 2x → 00840 | Breast biopsy/recon |
| 00952 | 91 | 98.9% | 1x → 01966 | Lumbar epidural labor |
| 00160 | 81 | 97.5% | 2x → 00320 | Nose/sinus |
| 00170 | 69 | 95.7% | 2x → 00300, 1x → 00160 | Intraoral procedures |
| 00910 | 64 | 96.9% | 2x → 00914 | Bladder procedures |
| 00530 | 45 | 95.6% | 2x → 00534 | Pacemaker/cardio |
| 64447 | 44 | 97.7% | 1x → 64473 | Femoral nerve block |
| 01844 | 38 | 97.4% | 1x → 01770 | Vascular lower leg |
| 00532 | 34 | 97.1% | 1x → 00520 | Vascular access |
| 00210 | 33 | 97.0% | 1x → 00300 | Intracranial |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01120 | 5 | 0.0% | 5x → 00300 | Pelvis bone |
| 00702 | 6 | 0.0% | 4x → 01922, 2x → 00520 | Liver biopsy |
| 00000 | 6 | 0.0% | 2x → 01991, 2x → 01830, 1x → 00300, 1x → 00320 |  |
| 00404 | 6 | 0.0% | 4x → 00400, 2x → 00402 | Mastectomy radical |
| 00162 | 44 | 4.5% | 40x → 00160, 1x → 00300, 1x → 00170 | Ethmoid surgery |
| 00542 | 15 | 6.7% | 12x → 00541, 1x → 00540, 1x → 00790 | Decortication |
| 00326 | 11 | 9.1% | 8x → 00320, 1x → 00520, 1x → 00300 |  |
| 00851 | 5 | 20.0% | 4x → 00840 | Lower abdominal (hernia) |
| 00792 | 7 | 28.6% | 4x → 00790, 1x → 00840 | Liver hemorrhage |
| 00846 | 34 | 29.4% | 23x → 00840, 1x → 00944 | Radical hysterectomy |
| 00192 | 13 | 38.5% | 3x → 00300, 3x → 01922, 1x → 00940, 1x → 00320 | Radical/major facial |
| 00176 | 5 | 40.0% | 3x → 00320 |  |
| 01470 | 21 | 42.9% | 7x → 01480, 3x → 00400, 2x → 00300 | Upper leg embolectomy |
| 01916 | 46 | 43.5% | 11x → 01926, 10x → 01924, 3x → 01930, 2x → 01925 | Diagnostic angiography |
| 01250 | 13 | 46.2% | 5x → 00400, 2x → 00300 | Upper arm/elbow |
| 01320 | 13 | 46.2% | 2x → 00300, 2x → 00400, 2x → 01250, 1x → 01400 | Knee disarticulation |
| 01925 | 18 | 50.0% | 8x → 01926, 1x → 01922 | Carotid angio |
| 01930 | 25 | 52.0% | 6x → 01924, 2x → 00532, 2x → 01522, 2x → 01931 | Veins central |
| 00860 | 23 | 52.2% | 5x → 00910, 2x → 00790, 2x → 00940, 1x → 01922 | Hip/upper leg/kidney |
| 00800 | 17 | 52.9% | 5x → 00910, 2x → 00840, 1x → 00400 | Lower abdominal |
| 00750 | 9 | 55.6% | 4x → 00840 | Hernia upper abdomen |
| 00350 | 9 | 55.6% | 4x → 00320 |  |
| 00500 | 5 | 60.0% | 2x → 00790 | Esophagus |
| 00214 | 5 | 60.0% | 2x → 00210 | Burr holes |
| 00926 | 8 | 62.5% | 2x → 01250, 1x → 00920 | Radical orchiectomy |
| 01966 | 19 | 63.2% | 5x → 01965, 2x → 00952 | Induced abortion |
| 01941 | 19 | 63.2% | 7x → 01942 | Percutaneous spine |
| 00145 | 85 | 64.7% | 28x → 00142, 1x → 00300, 1x → 00140 | Eye (vitrectomy) |
| 01810 | 6 | 66.7% | 2x → 00400 | Hip/femur procedures |
| 01610 | 24 | 66.7% | 7x → 00402, 1x → 00300 | Upper arm shoulder |
| 01215 | 9 | 66.7% | 2x → 01214, 1x → 01230 | Hip revision |
| 00944 | 12 | 66.7% | 2x → 00846, 2x → 00840 | Vaginal hysterectomy |
| 00600 | 6 | 66.7% | 1x → 00300, 1x → 00630 | Cervical spine |
| 00630 | 34 | 67.6% | 7x → 00670, 3x → 00600, 1x → 00300 | Lumbar laminectomy |
| 00100 | 14 | 71.4% | 2x → 00300, 2x → 00170 | Salivary gland/lip |
| 01991 | 7 | 71.4% | 2x → 00790 | Daily mgmt regional |
| 00918 | 67 | 73.1% | 16x → 00910, 2x → 00912 | Transurethral procedures |
| 00802 | 19 | 73.7% | 3x → 00400, 2x → 00402 | Panniculectomy |
| 00700 | 61 | 73.8% | 4x → 01922, 4x → 00730, 3x → 00800, 2x → 01844 | Upper abdominal |
| 01486 | 8 | 75.0% | 2x → 01480 | Total ankle |
| 01482 | 9 | 77.8% | 1x → 00400, 1x → 01320 | Lower leg (open) |
| 00942 | 14 | 78.6% | 2x → 00940, 1x → 00906 | Colpotomy/colpectomy |
| 00140 | 29 | 79.3% | 3x → 00300, 2x → 01480, 1x → 00160 | Eye (not lens) |
| 00862 | 44 | 79.5% | 5x → 01922, 3x → 00790, 1x → 01340 | Renal procedures |
| 00811 | 299 | 79.6% | 56x → 00812, 4x → 00731, 1x → 00902 | Colonoscopy diagnostic |
| 01931 | 25 | 80.0% | 3x → 01924, 2x → 01930 | TIPSS |
| 93503 | 5 | 80.0% | 1x → 36556 | Swan-Ganz |
| 01924 | 92 | 80.4% | 12x → 01926, 4x → 01930, 2x → 01916 | Therapeutic radiology |
| 00300 | 88 | 80.7% | 9x → 00400, 3x → 00192, 3x → 00320, 1x → 00160 | Head/neck vessels |
| 01961 | 38 | 81.6% | 7x → 01967 | C-section (emergency) |
| 00866 | 11 | 81.8% | 1x → 00840, 1x → 01926 | Adrenalectomy |
| 01965 | 18 | 83.3% | 3x → 01966 | Vaginal birth after C |
| 00190 | 6 | 83.3% | 1x → 00300 | Facial bone |
| 00103 | 18 | 83.3% | 3x → 00300 | Blepharoplasty |
| 00790 | 164 | 85.4% | 17x → 00840, 4x → 01922, 2x → 00400, 1x → 00700 | Intra-abdominal upper |
| 01400 | 23 | 87.0% | 2x → 01380, 1x → 01402 | Knee arthroscopy |
| 01926 | 31 | 87.1% | 4x → 01924 | Therapeutic radiology vascular |
| 00940 | 32 | 87.5% | 3x → 01922, 1x → 00904 | Vaginal delivery |
| 01210 | 8 | 87.5% | 1x → 01230 | Hip open |
| 01392 | 8 | 87.5% | 1x → 01480 | Knee ligament repair |
| 00562 | 8 | 87.5% | 1x → 00560 |  |
| 01830 | 9 | 88.9% | 1x → 00400 | Open femur/hip fracture |
| 01992 | 29 | 89.7% | 3x → 00300 | Pain mgmt continuous regional |
| 01630 | 10 | 90.0% | 1x → 01922 | Shoulder open |
| 00752 | 32 | 90.6% | 2x → 00790, 1x → 00750 | Lap upper hernia |
| 64445 | 24 | 91.7% | 2x → 64446 | Sciatic nerve block |
| 00914 | 49 | 91.8% | 3x → 00910, 1x → 00865 | Cystoscopy with stent |
| 00400 | 124 | 91.9% | 3x → 01250, 2x → 01630, 1x → 01830, 1x → 00300 | Integumentary (extremity/torso) |
| 00920 | 75 | 92.0% | 4x → 00910, 2x → 01120 | Bladder |
| 00840 | 141 | 92.2% | 4x → 00830, 3x → 00851, 2x → 00846, 2x → 00790 | Intra-abdominal lower |
| 00144 | 13 | 92.3% | 1x → 00140 | Eye corneal transplant |
| 01480 | 60 | 93.3% | 2x → 01470, 2x → 00400 | Lower leg/ankle/foot |
| 64488 | 32 | 93.8% | 2x → 64468 | TAP block continuous |
| 00560 | 17 | 94.1% | 1x → 00541 | Heart w/CPB age 1+ |
| 00904 | 19 | 94.7% | 1x → 00840 | Radical perineal |


---

## SIO-PSS

**Lines:** 1965  **CPT Changes:** 84  **Accuracy:** 95.73%

**HP Coverage (≥95% codes, ≥5 cases):** 1571/1965 = **79.9%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 1195/1965 = **60.8%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00813 | 344 | Colonoscopy screening→diagnostic |
| 00731 | 303 | Upper GI endoscopy |
| 00170 | 125 | Intraoral procedures |
| 00126 | 66 | Cataract surgery |
| 01480 | 53 | Lower leg/ankle/foot |
| 00160 | 51 | Nose/sinus |
| 00952 | 45 | Lumbar epidural labor |
| 64488 | 37 | TAP block continuous |
| 00145 | 29 | Eye (vitrectomy) |
| 64445 | 20 | Sciatic nerve block |
| 00940 | 18 | Vaginal delivery |
| 01400 | 15 | Knee arthroscopy |
| 64415 | 13 | Brachial plexus block |
| 00120 | 13 | Eye (external) |
| 01470 | 13 | Upper leg embolectomy |
| 00300 | 12 | Head/neck vessels |
| 01630 | 9 | Shoulder open |
| 64447 | 9 | Femoral nerve block |
| 01965 | 8 | Vaginal birth after C |
| 00320 | 6 | Anterior neck |
| 00400 | 6 | Integumentary (extremity/torso) |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 348 | 95.7% | 15x → 00811 | Colonoscopy screening |
| 01810 | 28 | 96.4% | 1x → 00400 | Hip/femur procedures |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01474 | 7 | 57.1% | 2x → 01480, 1x → 01470 | Lower leg amputation |
| 00811 | 243 | 78.2% | 53x → 00812 | Colonoscopy diagnostic |
| 00840 | 70 | 90.0% | 5x → 00851, 2x → 00944 | Intra-abdominal lower |
| 00402 | 15 | 93.3% | 1x → 00802 | Breast biopsy/recon |
| 01830 | 37 | 94.6% | 2x → 01820 | Open femur/hip fracture |


---

## IAS-BHS

**Lines:** 1352  **CPT Changes:** 113  **Accuracy:** 91.64%

**HP Coverage (≥95% codes, ≥5 cases):** 745/1352 = **55.1%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 625/1352 = **46.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00813 | 159 | Colonoscopy screening→diagnostic |
| 00731 | 127 | Upper GI endoscopy |
| 00952 | 38 | Lumbar epidural labor |
| 00537 | 26 | Open heart bypass |
| 76937 | 25 | Ultrasound guidance |
| 64447 | 24 | Femoral nerve block |
| 00902 | 19 | Anorectal surgery |
| 01961 | 18 | C-section (emergency) |
| 62322 | 18 | Lumbar epidural single shot |
| 64473 | 16 | Lumbar plexus block |
| 00873 | 16 | Lithotripsy/cystoscopy |
| 00520 | 15 | Chest procedures (closed) |
| 64415 | 15 | Brachial plexus block |
| 01630 | 11 | Shoulder open |
| 01214 | 10 | Total hip arthroplasty |
| 00912 | 10 | Transurethral resection |
| 64445 | 9 | Sciatic nerve block |
| 01402 | 9 | Knee arthroplasty |
| 00160 | 9 | Nose/sinus |
| 01638 | 8 | Shoulder arthroplasty |
| 00914 | 7 | Cystoscopy with stent |
| 01830 | 7 | Open femur/hip fracture |
| 01230 | 6 | Upper leg (femur) |
| 00404 | 6 | Mastectomy radical |
| 00918 | 6 | Transurethral procedures |
| 64488 | 6 | TAP block continuous |
| 01968 | 5 | C-section + hysterectomy |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 01480 | 44 | 95.5% | 1x → 01392, 1x → 01474 | Lower leg/ankle/foot |
| 00790 | 43 | 97.7% | 1x → 00752 | Intra-abdominal upper |
| 36620 | 33 | 97.0% | 1x → 93312 | Arterial line |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00860 | 19 | 15.8% | 7x → 00902, 4x → 00400, 4x → 00300, 1x → 00862 | Hip/upper leg/kidney |
| 01942 | 7 | 28.6% | 3x → 00630, 2x → 00300 | CT/interventional radiology |
| 01210 | 9 | 33.3% | 4x → 01230, 1x → 01250, 1x → 01220 | Hip open |
| 00630 | 6 | 50.0% | 3x → 00670 | Lumbar laminectomy |
| 01470 | 14 | 57.1% | 3x → 00400, 2x → 00300, 1x → 01250 | Upper leg embolectomy |
| 01810 | 19 | 63.2% | 5x → 01830, 2x → 00400 | Hip/femur procedures |
| 01710 | 6 | 66.7% | 1x → 01810, 1x → 01712 | Elbow/forearm |
| 00920 | 10 | 70.0% | 3x → 00902 | Bladder |
| 01400 | 9 | 77.8% | 1x → 01392, 1x → 01402 | Knee arthroscopy |
| 00832 | 5 | 80.0% | 1x → 00752 | Lap lower hernia |
| 00320 | 5 | 80.0% | 1x → 00300 | Anterior neck |
| 00400 | 30 | 83.3% | 2x → 01470, 2x → 01610, 1x → 00300 | Integumentary (extremity/torso) |
| 00940 | 6 | 83.3% | 1x → 00902 | Vaginal delivery |
| 36556 | 6 | 83.3% | 1x → 93312 | Central venous catheter |
| 00532 | 7 | 85.7% | 1x → 00400 | Vascular access |
| 00812 | 130 | 86.9% | 17x → 00811 | Colonoscopy screening |
| 00811 | 70 | 87.1% | 9x → 00812 | Colonoscopy diagnostic |
| 00862 | 17 | 88.2% | 2x → 00918 | Renal procedures |
| 00300 | 11 | 90.9% | 1x → 00400 | Head/neck vessels |
| 00910 | 42 | 92.9% | 3x → 00918 | Bladder procedures |
| 00840 | 85 | 94.1% | 3x → 00752, 1x → 00700, 1x → 00790 | Intra-abdominal lower |


---

## GOS-GOH

**Lines:** 1256  **CPT Changes:** 132  **Accuracy:** 89.49%

**HP Coverage (≥95% codes, ≥5 cases):** 658/1256 = **52.4%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 461/1256 = **36.7%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00813 | 61 | Colonoscopy screening→diagnostic |
| 01967 | 61 | C-section (planned) |
| 01961 | 33 | C-section (emergency) |
| 36620 | 30 | Arterial line |
| 76937 | 29 | Ultrasound guidance |
| 00520 | 26 | Chest procedures (closed) |
| 00952 | 23 | Lumbar epidural labor |
| 00732 | 22 | Upper GI w biopsy |
| 00902 | 21 | Anorectal surgery |
| 01402 | 21 | Knee arthroplasty |
| 64488 | 19 | TAP block continuous |
| 00940 | 14 | Vaginal delivery |
| 00320 | 12 | Anterior neck |
| 00300 | 11 | Head/neck vessels |
| 36556 | 10 | Central venous catheter |
| 00532 | 10 | Vascular access |
| 64445 | 9 | Sciatic nerve block |
| 00914 | 8 | Cystoscopy with stent |
| 01942 | 8 | CT/interventional radiology |
| 64466 | 7 |  |
| 01400 | 6 | Knee arthroscopy |
| 01638 | 5 | Shoulder arthroplasty |
| 64416 | 5 | Brachial plexus continuous |
| 01214 | 5 | Total hip arthroplasty |
| 01482 | 5 | Lower leg (open) |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00731 | 155 | 99.4% | 1x → 00813 | Upper GI endoscopy |
| 00910 | 22 | 95.5% | 1x → 00902 | Bladder procedures |
| 01480 | 20 | 95.0% | 1x → 01392 | Lower leg/ankle/foot |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01210 | 11 | 0.0% | 10x → 01230, 1x → 00400 | Hip open |
| 00860 | 8 | 12.5% | 5x → 00942, 1x → 00902, 1x → 00944 | Hip/upper leg/kidney |
| 01926 | 5 | 40.0% | 3x → 01924 | Therapeutic radiology vascular |
| 00812 | 42 | 42.9% | 23x → 00811, 1x → 00813 | Colonoscopy screening |
| 00830 | 7 | 57.1% | 2x → 00840, 1x → 00750 | Hernia lower abd |
| 01968 | 7 | 71.4% | 2x → 01967 | C-section + hysterectomy |
| 64448 | 18 | 72.2% | 5x → 64447 | Femoral nerve continuous |
| 01610 | 11 | 72.7% | 2x → 00400, 1x → 01630 | Upper arm shoulder |
| 00918 | 36 | 77.8% | 7x → 00910, 1x → 00914 | Transurethral procedures |
| 00912 | 5 | 80.0% | 1x → 00910 | Transurethral resection |
| 01941 | 5 | 80.0% | 1x → 01942 | Percutaneous spine |
| 00920 | 5 | 80.0% | 1x → 00860 | Bladder |
| 01810 | 56 | 82.1% | 10x → 01830 | Hip/femur procedures |
| 01965 | 7 | 85.7% | 1x → 00940 | Vaginal birth after C |
| 00400 | 29 | 86.2% | 2x → 01250, 1x → 01630, 1x → 01810 | Integumentary (extremity/torso) |
| 64415 | 15 | 86.7% | 2x → 64416 | Brachial plexus block |
| 00840 | 75 | 86.7% | 8x → 00790, 2x → 00752 | Intra-abdominal lower |
| 64473 | 18 | 88.9% | 1x → 64450, 1x → 64474 | Lumbar plexus block |
| 64447 | 9 | 88.9% | 1x → 64448 | Femoral nerve block |
| 00790 | 78 | 89.7% | 2x → 00750, 2x → 00752, 2x → 00840, 1x → 00902 | Intra-abdominal upper |
| 00402 | 12 | 91.7% | 1x → 00400 | Breast biopsy/recon |
| 01830 | 16 | 93.8% | 1x → 01820 | Open femur/hip fracture |
| 00811 | 33 | 93.9% | 2x → 00813 | Colonoscopy diagnostic |


---

## APO-UTP

**Lines:** 1181  **CPT Changes:** 104  **Accuracy:** 91.19%

**HP Coverage (≥95% codes, ≥5 cases):** 790/1181 = **66.9%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 510/1181 = **43.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01402 | 77 | Knee arthroplasty |
| 01967 | 65 | C-section (planned) |
| 01214 | 45 | Total hip arthroplasty |
| 01961 | 39 | C-section (emergency) |
| 64445 | 34 | Sciatic nerve block |
| 64415 | 30 | Brachial plexus block |
| 01400 | 25 | Knee arthroscopy |
| 00670 | 25 | Upper abdominal/spine |
| 01940 | 18 | Pain mgmt continuous |
| 01630 | 17 | Shoulder open |
| 00902 | 17 | Anorectal surgery |
| 00952 | 17 | Lumbar epidural labor |
| 01470 | 14 | Upper leg embolectomy |
| 01810 | 13 | Hip/femur procedures |
| 01638 | 12 | Shoulder arthroplasty |
| 01610 | 8 | Upper arm shoulder |
| 64488 | 8 | TAP block continuous |
| 00813 | 8 | Colonoscopy screening→diagnostic |
| 01965 | 8 | Vaginal birth after C |
| 00126 | 7 | Cataract surgery |
| 01486 | 6 | Total ankle |
| 01939 | 6 | Vertebroplasty/kyphoplasty |
| 00731 | 6 | Upper GI endoscopy |
| 01472 | 5 | Tibia/fibula repair |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 64447 | 96 | 95.8% | 2x → 64450, 2x → 64473 | Femoral nerve block |
| 64473 | 71 | 97.2% | 2x → 64447 | Lumbar plexus block |
| 01480 | 51 | 98.0% | 1x → 01474 | Lower leg/ankle/foot |
| 00812 | 32 | 96.9% | 1x → 00811 | Colonoscopy screening |
| 99140 | 30 | 96.7% | 1x → 00904 |  |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00600 | 7 | 0.0% | 7x → 00670 | Cervical spine |
| 01462 | 7 | 14.3% | 6x → 01480 | Lower leg closed |
| 01938 | 7 | 14.3% | 5x → 01940, 1x → 01992 | Pain mgmt epidural injection |
| 01210 | 7 | 42.9% | 3x → 01230, 1x → 01220 | Hip open |
| 01320 | 5 | 60.0% | 2x → 01991 | Knee disarticulation |
| 01710 | 7 | 71.4% | 2x → 01810 | Elbow/forearm |
| 00400 | 11 | 72.7% | 2x → 00402, 1x → 00902 | Integumentary (extremity/torso) |
| 00940 | 9 | 77.8% | 1x → 01965, 1x → 00860 | Vaginal delivery |
| 00840 | 75 | 80.0% | 5x → 00851, 2x → 00790, 2x → 00944, 2x → 00750 | Intra-abdominal lower |
| 00320 | 10 | 80.0% | 1x → 00170, 1x → 00300 | Anterior neck |
| 00532 | 10 | 80.0% | 2x → 36410 | Vascular access |
| 00170 | 17 | 82.4% | 1x → 00300, 1x → 00120, 1x → 00160 | Intraoral procedures |
| 01392 | 7 | 85.7% | 1x → 00400 | Knee ligament repair |
| 00910 | 17 | 88.2% | 1x → 00918, 1x → 00902 | Bladder procedures |
| 00300 | 9 | 88.9% | 1x → 00400 | Head/neck vessels |
| 00790 | 39 | 89.7% | 4x → 00840 | Intra-abdominal upper |
| 62322 | 25 | 92.0% | 1x → 36410, 1x → 01968 | Lumbar epidural single shot |
| 00811 | 13 | 92.3% | 1x → 00812 | Colonoscopy diagnostic |


---

## SIO-STL

**Lines:** 1155  **CPT Changes:** 60  **Accuracy:** 94.81%

**HP Coverage (≥95% codes, ≥5 cases):** 635/1155 = **55.0%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 467/1155 = **40.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00170 | 93 | Intraoral procedures |
| 00790 | 85 | Intra-abdominal upper |
| 01480 | 42 | Lower leg/ankle/foot |
| 00731 | 40 | Upper GI endoscopy |
| 00732 | 22 | Upper GI w biopsy |
| 00910 | 21 | Bladder procedures |
| 00813 | 18 | Colonoscopy screening→diagnostic |
| 00520 | 17 | Chest procedures (closed) |
| 01965 | 16 | Vaginal birth after C |
| 00410 | 16 | Heart catheterization |
| 00537 | 13 | Open heart bypass |
| 00851 | 12 | Lower abdominal (hernia) |
| 00902 | 10 | Anorectal surgery |
| 01922 | 10 | CT/interventional radiology |
| 64445 | 8 | Sciatic nerve block |
| 36620 | 8 | Arterial line |
| 00952 | 7 | Lumbar epidural labor |
| 01844 | 7 | Vascular lower leg |
| 36410 | 7 | Venipuncture |
| 00918 | 5 | Transurethral procedures |
| 76937 | 5 | Ultrasound guidance |
| 64447 | 5 | Femoral nerve block |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 64488 | 139 | 99.3% | 1x → 64486 | TAP block continuous |
| 00532 | 29 | 96.6% | 1x → 00400 | Vascular access |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01470 | 11 | 54.5% | 3x → 01480, 2x → 00400 | Upper leg embolectomy |
| 01968 | 23 | 73.9% | 6x → 01967 | C-section + hysterectomy |
| 00812 | 12 | 75.0% | 3x → 00811 | Colonoscopy screening |
| 01210 | 10 | 80.0% | 2x → 01230 | Hip open |
| 00400 | 26 | 84.6% | 2x → 00700, 1x → 01610, 1x → 00300 | Integumentary (extremity/torso) |
| 00750 | 7 | 85.7% | 1x → 00790 | Hernia upper abdomen |
| 00940 | 7 | 85.7% | 1x → 00860 | Vaginal delivery |
| 00840 | 128 | 92.2% | 4x → 00851, 2x → 00904, 2x → 00830, 2x → 00790 | Intra-abdominal lower |
| 00300 | 14 | 92.9% | 1x → 00400 | Head/neck vessels |
| 00811 | 15 | 93.3% | 1x → 00813 | Colonoscopy diagnostic |
| 01961 | 65 | 93.8% | 3x → 01967, 1x → 01968 | C-section (emergency) |
| 01967 | 113 | 94.7% | 6x → 01968 | C-section (planned) |


---

## PCE-PMC

**Lines:** 1146  **CPT Changes:** 73  **Accuracy:** 93.63%

**HP Coverage (≥95% codes, ≥5 cases):** 941/1146 = **82.1%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 105/1146 = **9.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 105 | Upper GI endoscopy |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 757 | 96.3% | 28x → 00811 | Colonoscopy screening |
| 00813 | 79 | 98.7% | 1x → 00812 | Colonoscopy screening→diagnostic |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00811 | 205 | 78.5% | 44x → 00812 | Colonoscopy diagnostic |


---

## KAP-CYP

**Lines:** 1140  **CPT Changes:** 64  **Accuracy:** 94.39%

**HP Coverage (≥95% codes, ≥5 cases):** 739/1140 = **64.8%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 335/1140 = **29.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01400 | 69 | Knee arthroscopy |
| 00952 | 58 | Lumbar epidural labor |
| 00140 | 42 | Eye (not lens) |
| 00142 | 30 | Lens procedures (cataract) |
| 00731 | 24 | Upper GI endoscopy |
| 00851 | 24 | Lower abdominal (hernia) |
| 00300 | 23 | Head/neck vessels |
| 00920 | 16 | Bladder |
| 00402 | 14 | Breast biopsy/recon |
| 00532 | 9 | Vascular access |
| 00103 | 8 | Blepharoplasty |
| 00813 | 7 | Colonoscopy screening→diagnostic |
| 00802 | 6 | Panniculectomy |
| 00790 | 5 | Intra-abdominal upper |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00170 | 233 | 99.6% | 1x → 00126 | Intraoral procedures |
| 00160 | 72 | 97.2% | 1x → 00170, 1x → 00300 | Nose/sinus |
| 00940 | 56 | 98.2% | 1x → 01965 | Vaginal delivery |
| 00320 | 43 | 97.7% | 1x → 00300 | Anterior neck |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00400 | 46 | 63.0% | 13x → 00402, 2x → 00940, 1x → 01250, 1x → 00300 | Integumentary (extremity/torso) |
| 00811 | 16 | 75.0% | 3x → 00812, 1x → 00731 | Colonoscopy diagnostic |
| 00120 | 8 | 87.5% | 1x → 00124 | Eye (external) |
| 00126 | 75 | 89.3% | 6x → 00120, 1x → 00124, 1x → 00160 | Cataract surgery |
| 00812 | 123 | 89.4% | 13x → 00811 | Colonoscopy screening |
| 01610 | 11 | 90.9% | 1x → 00400 | Upper arm shoulder |
| 00840 | 65 | 93.8% | 3x → 00944, 1x → 00952 | Intra-abdominal lower |
| 01965 | 18 | 94.4% | 1x → 00952 | Vaginal birth after C |


---

## GII-ASC

**Lines:** 1032  **CPT Changes:** 36  **Accuracy:** 96.51%

**HP Coverage (≥95% codes, ≥5 cases):** 746/1032 = **72.3%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 303/1032 = **29.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 170 | Upper GI endoscopy |
| 00813 | 133 | Colonoscopy screening→diagnostic |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 443 | 99.5% | 2x → 00811 | Colonoscopy screening |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00811 | 286 | 88.1% | 34x → 00812 | Colonoscopy diagnostic |


---

## IAS-MOR

**Lines:** 1005  **CPT Changes:** 61  **Accuracy:** 93.93%

**HP Coverage (≥95% codes, ≥5 cases):** 741/1005 = **73.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 317/1005 = **31.5%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 113 | Upper GI endoscopy |
| 01967 | 51 | C-section (planned) |
| 01961 | 19 | C-section (emergency) |
| 01480 | 18 | Lower leg/ankle/foot |
| 00952 | 14 | Lumbar epidural labor |
| 00520 | 14 | Chest procedures (closed) |
| 01402 | 14 | Knee arthroplasty |
| 00902 | 14 | Anorectal surgery |
| 64415 | 12 | Brachial plexus block |
| 64488 | 9 | TAP block continuous |
| 01630 | 8 | Shoulder open |
| 01214 | 7 | Total hip arthroplasty |
| 01638 | 7 | Shoulder arthroplasty |
| 00120 | 7 | Eye (external) |
| 00160 | 5 | Nose/sinus |
| 01810 | 5 | Hip/femur procedures |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 164 | 96.3% | 6x → 00811 | Colonoscopy screening |
| 00813 | 141 | 97.2% | 2x → 00731, 1x → 00812, 1x → 00790 | Colonoscopy screening→diagnostic |
| 00840 | 65 | 95.4% | 2x → 00851, 1x → 00750 | Intra-abdominal lower |
| 00790 | 54 | 98.1% | 1x → 00750 | Intra-abdominal upper |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00600 | 11 | 27.3% | 8x → 00670 | Cervical spine |
| 01968 | 8 | 37.5% | 5x → 01967 | C-section + hysterectomy |
| 01210 | 11 | 72.7% | 2x → 01230, 1x → 00300 | Hip open |
| 00400 | 19 | 73.7% | 2x → 01610, 1x → 00940, 1x → 00300, 1x → 01480 | Integumentary (extremity/torso) |
| 36620 | 8 | 75.0% | 2x → 99140 | Arterial line |
| 00811 | 94 | 83.0% | 15x → 00812, 1x → 00902 | Colonoscopy diagnostic |
| 00752 | 9 | 88.9% | 1x → 00790 | Lap upper hernia |
| 00670 | 15 | 93.3% | 1x → 00630 | Upper abdominal/spine |
| 00300 | 15 | 93.3% | 1x → 00352 | Head/neck vessels |
| 00830 | 16 | 93.8% | 1x → 00840 | Hernia lower abd |


---

## IAS-FVO

**Lines:** 947  **CPT Changes:** 32  **Accuracy:** 96.62%

**HP Coverage (≥95% codes, ≥5 cases):** 832/947 = **87.9%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 559/947 = **59.0%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 64415 | 132 | Brachial plexus block |
| 01400 | 123 | Knee arthroscopy |
| 01630 | 91 | Shoulder open |
| 01402 | 77 | Knee arthroplasty |
| 01214 | 48 | Total hip arthroplasty |
| 01480 | 40 | Lower leg/ankle/foot |
| 01638 | 21 | Shoulder arthroplasty |
| 01202 | 6 | Hip arthroplasty (posterior) |
| 01392 | 6 | Knee ligament repair |
| 01380 | 5 | Knee disarticulation (open) |
| 01472 | 5 | Tibia/fibula repair |
| 01486 | 5 | Total ankle |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 64447 | 102 | 99.0% | 1x → 64473 | Femoral nerve block |
| 01810 | 83 | 98.8% | 1x → 00400 | Hip/femur procedures |
| 01830 | 47 | 97.9% | 1x → 01810 | Open femur/hip fracture |
| 64473 | 41 | 97.6% | 1x → 64447 | Lumbar plexus block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00600 | 5 | 40.0% | 3x → 00670 | Cervical spine |
| 01710 | 9 | 44.4% | 2x → 01714, 2x → 01810, 1x → 01712 | Elbow/forearm |
| 01470 | 7 | 57.1% | 2x → 00400, 1x → 01464 | Upper leg embolectomy |
| 01320 | 6 | 66.7% | 1x → 01470, 1x → 01400 | Knee disarticulation |
| 64445 | 33 | 72.7% | 5x → 64473, 2x → 64447, 2x → 64450 | Sciatic nerve block |
| 00400 | 6 | 83.3% | 1x → 01810 | Integumentary (extremity/torso) |
| 00630 | 17 | 94.1% | 1x → 00670 | Lumbar laminectomy |


---

## PCE-WWMG

**Lines:** 947  **CPT Changes:** 160  **Accuracy:** 83.10%

**HP Coverage (≥95% codes, ≥5 cases):** 217/947 = **22.9%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 217/947 = **22.9%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 164 | Upper GI endoscopy |
| 00813 | 53 | Colonoscopy screening→diagnostic |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00812 | 558 | 73.3% | 149x → 00811 | Colonoscopy screening |
| 00811 | 172 | 93.6% | 11x → 00812 | Colonoscopy diagnostic |


---

## GAP

**Lines:** 938  **CPT Changes:** 44  **Accuracy:** 95.31%

**HP Coverage (≥95% codes, ≥5 cases):** 706/938 = **75.3%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 653/938 = **69.6%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 186 | Lens procedures (cataract) |
| 64447 | 53 | Femoral nerve block |
| 01967 | 47 | C-section (planned) |
| 01214 | 33 | Total hip arthroplasty |
| 00170 | 32 | Intraoral procedures |
| 64488 | 25 | TAP block continuous |
| 00790 | 24 | Intra-abdominal upper |
| 01480 | 23 | Lower leg/ankle/foot |
| 01810 | 23 | Hip/femur procedures |
| 00410 | 22 | Heart catheterization |
| 64415 | 20 | Brachial plexus block |
| 01961 | 18 | C-section (emergency) |
| 01922 | 15 | CT/interventional radiology |
| 00400 | 13 | Integumentary (extremity/torso) |
| 01630 | 13 | Shoulder open |
| 01830 | 12 | Open femur/hip fracture |
| 01638 | 12 | Shoulder arthroplasty |
| 00126 | 12 | Cataract surgery |
| 00910 | 11 | Bladder procedures |
| 00160 | 10 | Nose/sinus |
| 00811 | 8 | Colonoscopy diagnostic |
| 00532 | 7 | Vascular access |
| 00731 | 7 | Upper GI endoscopy |
| 00140 | 6 | Eye (not lens) |
| 64473 | 6 | Lumbar plexus block |
| 00912 | 5 | Transurethral resection |
| 01482 | 5 | Lower leg (open) |
| 01942 | 5 | CT/interventional radiology |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 01402 | 53 | 98.1% | 1x → 01400 | Knee arthroplasty |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00620 | 5 | 0.0% | 3x → 01942, 2x → 01941 | Lumbar spine |
| 00600 | 8 | 50.0% | 4x → 00670 | Cervical spine |
| 01210 | 5 | 60.0% | 2x → 01230 | Hip open |
| 00670 | 18 | 77.8% | 2x → 01941, 1x → 00300, 1x → 01942 | Upper abdominal/spine |
| 00812 | 10 | 80.0% | 2x → 00811 | Colonoscopy screening |
| 00300 | 5 | 80.0% | 1x → 01942 | Head/neck vessels |
| 01610 | 7 | 85.7% | 1x → 00400 | Upper arm shoulder |
| 00320 | 9 | 88.9% | 1x → 00300 | Anterior neck |
| 00630 | 10 | 90.0% | 1x → 01942 | Lumbar laminectomy |
| 01968 | 10 | 90.0% | 1x → 01967 | C-section + hysterectomy |
| 00840 | 46 | 91.3% | 2x → 00800, 1x → 00851, 1x → 00790 | Intra-abdominal lower |
| 01400 | 18 | 94.4% | 1x → 01392 | Knee arthroscopy |


---

## CHA-HDH

**Lines:** 765  **CPT Changes:** 94  **Accuracy:** 87.71%

**HP Coverage (≥95% codes, ≥5 cases):** 442/765 = **57.8%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 370/765 = **48.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 49 | Lens procedures (cataract) |
| 00731 | 45 | Upper GI endoscopy |
| 64488 | 28 | TAP block continuous |
| 00790 | 23 | Intra-abdominal upper |
| 64473 | 20 | Lumbar plexus block |
| 00840 | 18 | Intra-abdominal lower |
| 64447 | 17 | Femoral nerve block |
| 01967 | 17 | C-section (planned) |
| 01961 | 16 | C-section (emergency) |
| 36620 | 14 | Arterial line |
| 01400 | 12 | Knee arthroscopy |
| 01922 | 12 | CT/interventional radiology |
| 00952 | 12 | Lumbar epidural labor |
| 01214 | 12 | Total hip arthroplasty |
| 01402 | 11 | Knee arthroplasty |
| 00902 | 9 | Anorectal surgery |
| 64415 | 8 | Brachial plexus block |
| 00940 | 7 | Vaginal delivery |
| 00912 | 7 | Transurethral resection |
| 64466 | 6 |  |
| 01830 | 6 | Open femur/hip fracture |
| 00126 | 6 | Cataract surgery |
| 01965 | 5 | Vaginal birth after C |
| 00914 | 5 | Cystoscopy with stent |
| 00170 | 5 | Intraoral procedures |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 52 | 98.1% | 1x → 00811 | Colonoscopy screening |
| 00813 | 20 | 95.0% | 1x → 00731 | Colonoscopy screening→diagnostic |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00862 | 8 | 12.5% | 6x → 00918, 1x → 00910 | Renal procedures |
| 00532 | 7 | 14.3% | 4x → 36410, 2x → 36573 | Vascular access |
| 01210 | 8 | 50.0% | 4x → 01230 | Hip open |
| 00752 | 5 | 60.0% | 2x → 00790 | Lap upper hernia |
| 00918 | 10 | 60.0% | 4x → 00910 | Transurethral procedures |
| 00910 | 35 | 62.9% | 12x → 00918, 1x → 00914 | Bladder procedures |
| 00830 | 11 | 63.6% | 2x → 00840, 1x → 00750, 1x → 64425 | Hernia lower abd |
| 01942 | 6 | 66.7% | 1x → 01941, 1x → 00300 | CT/interventional radiology |
| 00811 | 58 | 69.0% | 18x → 00812 | Colonoscopy diagnostic |
| 00670 | 33 | 78.8% | 2x → 00300, 2x → 01942, 1x → 00600, 1x → 00630 | Upper abdominal/spine |
| 00400 | 22 | 81.8% | 2x → 01610, 1x → 00300, 1x → 00904 | Integumentary (extremity/torso) |
| 00630 | 10 | 90.0% | 1x → 00300 | Lumbar laminectomy |
| 01810 | 11 | 90.9% | 1x → 01830 | Hip/femur procedures |
| 01480 | 13 | 92.3% | 1x → 01470 | Lower leg/ankle/foot |


---

## INJE-CLIFW

**Lines:** 765  **CPT Changes:** 24  **Accuracy:** 96.86%

**HP Coverage (≥95% codes, ≥5 cases):** 714/765 = **93.3%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 0/765 = **0.0%** of lines code-perfect.

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00142 | 714 | 99.4% | 3x → 00140, 1x → 00145 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00140 | 45 | 60.0% | 18x → 00142 | Eye (not lens) |
| 00144 | 6 | 66.7% | 2x → 00142 | Eye corneal transplant |


---

## PRM-WHT

**Lines:** 689  **CPT Changes:** 10  **Accuracy:** 98.55%

**HP Coverage (≥95% codes, ≥5 cases):** 616/689 = **89.4%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 616/689 = **89.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 597 | Lens procedures (cataract) |
| 00103 | 19 | Blepharoplasty |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00300 | 15 | 86.7% | 2x → 00103 | Head/neck vessels |
| 00140 | 56 | 87.5% | 6x → 00142, 1x → 00103 | Eye (not lens) |


---

## PAC-MHI

**Lines:** 666  **CPT Changes:** 34  **Accuracy:** 94.89%

**HP Coverage (≥95% codes, ≥5 cases):** 482/666 = **72.4%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 293/666 = **44.0%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 62 | Upper GI endoscopy |
| 01967 | 32 | C-section (planned) |
| 00790 | 29 | Intra-abdominal upper |
| 00813 | 28 | Colonoscopy screening→diagnostic |
| 01402 | 25 | Knee arthroplasty |
| 01922 | 17 | CT/interventional radiology |
| 00410 | 17 | Heart catheterization |
| 00537 | 16 | Open heart bypass |
| 01214 | 12 | Total hip arthroplasty |
| 00530 | 12 | Pacemaker/cardio |
| 00120 | 10 | Eye (external) |
| 00170 | 9 | Intraoral procedures |
| 00300 | 7 | Head/neck vessels |
| 00400 | 6 | Integumentary (extremity/torso) |
| 64415 | 6 | Brachial plexus block |
| 01638 | 5 | Shoulder arthroplasty |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 159 | 95.6% | 7x → 00811 | Colonoscopy screening |
| 00840 | 30 | 96.7% | 1x → 00944 | Intra-abdominal lower |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01968 | 7 | 57.1% | 3x → 01967 | C-section + hysterectomy |
| 01810 | 5 | 80.0% | 1x → 01830 | Hip/femur procedures |
| 00811 | 91 | 85.7% | 12x → 00812, 1x → 00813 | Colonoscopy diagnostic |
| 01480 | 8 | 87.5% | 1x → 00400 | Lower leg/ankle/foot |
| 00670 | 17 | 94.1% | 1x → 00300 | Upper abdominal/spine |


---

## IAS-BMH

**Lines:** 618  **CPT Changes:** 55  **Accuracy:** 91.10%

**HP Coverage (≥95% codes, ≥5 cases):** 340/618 = **55.0%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 305/618 = **49.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 39 | Upper GI endoscopy |
| 01810 | 29 | Hip/femur procedures |
| 01214 | 26 | Total hip arthroplasty |
| 01402 | 26 | Knee arthroplasty |
| 00142 | 24 | Lens procedures (cataract) |
| 01480 | 22 | Lower leg/ankle/foot |
| 01630 | 22 | Shoulder open |
| 00840 | 22 | Intra-abdominal lower |
| 01400 | 15 | Knee arthroscopy |
| 00790 | 14 | Intra-abdominal upper |
| 01830 | 14 | Open femur/hip fracture |
| 01638 | 14 | Shoulder arthroplasty |
| 00952 | 12 | Lumbar epidural labor |
| 00912 | 8 | Transurethral resection |
| 01230 | 7 | Upper leg (femur) |
| 01965 | 6 | Vaginal birth after C |
| 00300 | 5 | Head/neck vessels |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00813 | 35 | 97.1% | 1x → 00731 | Colonoscopy screening→diagnostic |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01470 | 5 | 40.0% | 1x → 00400, 1x → 01464, 1x → 01480 | Upper leg embolectomy |
| 00811 | 33 | 48.5% | 17x → 00812 | Colonoscopy diagnostic |
| 01210 | 5 | 60.0% | 2x → 01230 | Hip open |
| 01961 | 8 | 62.5% | 3x → 01968 | C-section (emergency) |
| 00400 | 12 | 66.7% | 1x → 01930, 1x → 00470, 1x → 01610, 1x → 00300 | Integumentary (extremity/torso) |
| 00830 | 18 | 72.2% | 3x → 00750, 1x → 00752, 1x → 00790 | Hernia lower abd |
| 01710 | 5 | 80.0% | 1x → 01714 | Elbow/forearm |
| 00918 | 11 | 90.9% | 1x → 00862 | Transurethral procedures |
| 00812 | 135 | 91.1% | 12x → 00811 | Colonoscopy screening |


---

## TAN-ESC

**Lines:** 614  **CPT Changes:** 9  **Accuracy:** 98.53%

**HP Coverage (≥95% codes, ≥5 cases):** 505/614 = **82.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 505/614 = **82.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 403 | Lens procedures (cataract) |
| 00103 | 66 | Blepharoplasty |
| 00144 | 21 | Eye corneal transplant |
| 00145 | 15 | Eye (vitrectomy) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00300 | 8 | 87.5% | 1x → 00103 | Head/neck vessels |
| 00140 | 97 | 94.8% | 2x → 00142, 2x → 00145, 1x → 00144 | Eye (not lens) |


---

## KAP-ASC

**Lines:** 613  **CPT Changes:** 39  **Accuracy:** 93.64%

**HP Coverage (≥95% codes, ≥5 cases):** 437/613 = **71.3%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 296/613 = **48.3%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00160 | 88 | Nose/sinus |
| 00126 | 81 | Cataract surgery |
| 01400 | 33 | Knee arthroscopy |
| 01630 | 33 | Shoulder open |
| 01810 | 22 | Hip/femur procedures |
| 00811 | 15 | Colonoscopy diagnostic |
| 00813 | 7 | Colonoscopy screening→diagnostic |
| 00731 | 6 | Upper GI endoscopy |
| 00300 | 6 | Head/neck vessels |
| 00402 | 5 | Breast biopsy/recon |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00170 | 119 | 96.6% | 2x → 00160, 2x → 00190 | Intraoral procedures |
| 00320 | 22 | 95.5% | 1x → 00160 | Anterior neck |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00400 | 27 | 33.3% | 17x → 00402, 1x → 00300 | Integumentary (extremity/torso) |
| 01710 | 7 | 42.9% | 3x → 01810, 1x → 01714 | Elbow/forearm |
| 00802 | 15 | 73.3% | 3x → 00400, 1x → 00402 | Panniculectomy |
| 00812 | 112 | 94.6% | 6x → 00811 | Colonoscopy screening |


---

## WPA

**Lines:** 559  **CPT Changes:** 10  **Accuracy:** 98.21%

**HP Coverage (≥95% codes, ≥5 cases):** 468/559 = **83.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 468/559 = **83.7%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 244 | Lens procedures (cataract) |
| 00170 | 79 | Intraoral procedures |
| 00812 | 55 | Colonoscopy screening |
| 00103 | 17 | Blepharoplasty |
| 00126 | 17 | Cataract surgery |
| 00731 | 16 | Upper GI endoscopy |
| 00813 | 16 | Colonoscopy screening→diagnostic |
| 00145 | 14 | Eye (vitrectomy) |
| 00300 | 10 | Head/neck vessels |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00811 | 18 | 77.8% | 4x → 00812 | Colonoscopy diagnostic |
| 00140 | 49 | 93.9% | 2x → 00142, 1x → 00144 | Eye (not lens) |


---

## APO-ORA

**Lines:** 549  **CPT Changes:** 44  **Accuracy:** 91.99%

**HP Coverage (≥95% codes, ≥5 cases):** 371/549 = **67.6%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 290/549 = **52.8%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 64447 | 56 | Femoral nerve block |
| 01400 | 52 | Knee arthroscopy |
| 01630 | 50 | Shoulder open |
| 01402 | 35 | Knee arthroplasty |
| 01940 | 26 | Pain mgmt continuous |
| 01480 | 22 | Lower leg/ankle/foot |
| 01638 | 14 | Shoulder arthroplasty |
| 01939 | 13 | Vertebroplasty/kyphoplasty |
| 01740 | 9 | Elbow arthroplasty |
| 01392 | 7 | Knee ligament repair |
| 64445 | 6 | Sciatic nerve block |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 64415 | 81 | 98.8% | 1x → 64417 | Brachial plexus block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 64999 | 5 | 0.0% | 5x → 64473 | Unlisted nerve block |
| 01938 | 14 | 7.1% | 13x → 01940 | Pain mgmt epidural injection |
| 01320 | 7 | 42.9% | 2x → 01991, 1x → 01470, 1x → 00400 | Knee disarticulation |
| 00630 | 6 | 50.0% | 2x → 01941, 1x → 00300 | Lumbar laminectomy |
| 01710 | 15 | 60.0% | 3x → 01714, 2x → 01810, 1x → 01712 | Elbow/forearm |
| 01810 | 59 | 91.5% | 4x → 01830, 1x → 01710 | Hip/femur procedures |
| 01830 | 36 | 94.4% | 2x → 01810 | Open femur/hip fracture |


---

## LOV

**Lines:** 549  **CPT Changes:** 6  **Accuracy:** 98.91%

**HP Coverage (≥95% codes, ≥5 cases):** 476/549 = **86.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 476/549 = **86.7%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 476 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00144 | 17 | 88.2% | 2x → 00142 | Eye corneal transplant |
| 00140 | 53 | 92.5% | 3x → 00144, 1x → 00142 | Eye (not lens) |


---

## NTA-WGS

**Lines:** 545  **CPT Changes:** 27  **Accuracy:** 95.05%

**HP Coverage (≥95% codes, ≥5 cases):** 471/545 = **86.4%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 299/545 = **54.9%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00902 | 86 | Anorectal surgery |
| 00910 | 82 | Bladder procedures |
| 01630 | 42 | Shoulder open |
| 01400 | 25 | Knee arthroscopy |
| 00920 | 19 | Bladder |
| 00912 | 14 | Transurethral resection |
| 00914 | 12 | Cystoscopy with stent |
| 64447 | 9 | Femoral nerve block |
| 00300 | 5 | Head/neck vessels |
| 01710 | 5 | Elbow/forearm |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00921 | 51 | 98.0% | 1x → 00920 | Bladder surgery |
| 01810 | 44 | 97.7% | 1x → 00400 | Hip/femur procedures |
| 00918 | 39 | 97.4% | 1x → 00910 | Transurethral procedures |
| 64415 | 38 | 97.4% | 1x → 64447 | Brachial plexus block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00860 | 8 | 0.0% | 7x → 00902, 1x → 00300 | Hip/upper leg/kidney |
| 00670 | 7 | 57.1% | 3x → 01942 | Upper abdominal/spine |
| 01942 | 10 | 70.0% | 2x → 00300, 1x → 00400 | CT/interventional radiology |
| 01830 | 9 | 77.8% | 1x → 01810, 1x → 00400 | Open femur/hip fracture |
| 00630 | 8 | 87.5% | 1x → 01942 | Lumbar laminectomy |


---

## INJE-CSCG

**Lines:** 542  **CPT Changes:** 93  **Accuracy:** 82.84%

**HP Coverage (≥95% codes, ≥5 cases):** 422/542 = **77.9%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 0/542 = **0.0%** of lines code-perfect.

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00142 | 422 | 99.8% | 1x → 00140 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00140 | 119 | 23.5% | 91x → 00142 | Eye (not lens) |

### Recent Performance — Post April 8, 2026 (after 00140 rule fix)

A CPT instruction template fix for the 00140↔00142 confusion took effect at batch #176 (integrated 2026-04-08). The numbers above include pre-fix March/early-April batches and no longer reflect current behavior. Restricted to batches #176-#187:

**Lines:** 219  **CPT Changes:** 0  **Accuracy:** 100.00%

**HP Coverage (≥95% codes, ≥5 cases):** 215/219 = **98.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 215/219 = **98.2%** of lines code-perfect.

Batches included: #176 through #187 (10 batches between 2026-04-08 and 2026-04-29).

#### 100% Precision (≥5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 215 | Lens procedures (cataract) |

The previously systematic `00140 → 00142` correction (74-100% error rate in March) has been fully eliminated. Across these 219 lines, **zero CPT corrections** were made by coders.


---

## GOS-GOSC

**Lines:** 508  **CPT Changes:** 104  **Accuracy:** 79.53%

**HP Coverage (≥95% codes, ≥5 cases):** 139/508 = **27.4%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 10/508 = **2.0%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00160 | 10 | Nose/sinus |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00731 | 75 | 98.7% | 1x → 00813 | Upper GI endoscopy |
| 00813 | 54 | 98.1% | 1x → 00731 | Colonoscopy screening→diagnostic |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00812 | 241 | 68.9% | 58x → 00811, 17x → 00813 | Colonoscopy screening |
| 00811 | 75 | 69.3% | 14x → 00813, 9x → 00812 | Colonoscopy diagnostic |
| 00170 | 13 | 92.3% | 1x → 00100 | Intraoral procedures |
| 00126 | 13 | 92.3% | 1x → 00124 | Cataract surgery |


---

## PRM

**Lines:** 502  **CPT Changes:** 39  **Accuracy:** 92.23%

**HP Coverage (≥95% codes, ≥5 cases):** 277/502 = **55.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 277/502 = **55.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00902 | 37 | Anorectal surgery |
| 01810 | 26 | Hip/femur procedures |
| 00142 | 25 | Lens procedures (cataract) |
| 00731 | 23 | Upper GI endoscopy |
| 00952 | 20 | Lumbar epidural labor |
| 00790 | 19 | Intra-abdominal upper |
| 01402 | 15 | Knee arthroplasty |
| 64488 | 15 | TAP block continuous |
| 00813 | 14 | Colonoscopy screening→diagnostic |
| 01630 | 12 | Shoulder open |
| 01214 | 12 | Total hip arthroplasty |
| 00532 | 10 | Vascular access |
| 00160 | 10 | Nose/sinus |
| 00940 | 8 | Vaginal delivery |
| 01965 | 7 | Vaginal birth after C |
| 01400 | 7 | Knee arthroscopy |
| 01638 | 6 | Shoulder arthroplasty |
| 00400 | 6 | Integumentary (extremity/torso) |
| 00921 | 5 | Bladder surgery |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00918 | 14 | 50.0% | 7x → 00910 | Transurethral procedures |
| 01210 | 5 | 60.0% | 2x → 01230 | Hip open |
| 00920 | 6 | 66.7% | 2x → 00902 | Bladder |
| 01961 | 7 | 71.4% | 2x → 01968 | C-section (emergency) |
| 00811 | 12 | 75.0% | 3x → 00812 | Colonoscopy diagnostic |
| 00750 | 7 | 85.7% | 1x → 00790 | Hernia upper abdomen |
| 00300 | 7 | 85.7% | 1x → 01610 | Head/neck vessels |
| 00812 | 29 | 86.2% | 4x → 00811 | Colonoscopy screening |
| 00873 | 9 | 88.9% | 1x → 00918 | Lithotripsy/cystoscopy |
| 00170 | 11 | 90.9% | 1x → 00300 | Intraoral procedures |
| 00840 | 36 | 91.7% | 3x → 00851 | Intra-abdominal lower |
| 01480 | 14 | 92.9% | 1x → 01392 | Lower leg/ankle/foot |
| 00830 | 16 | 93.8% | 1x → 00750 | Hernia lower abd |
| 00910 | 17 | 94.1% | 1x → 00918 | Bladder procedures |


---

## PCE-CAS

**Lines:** 475  **CPT Changes:** 35  **Accuracy:** 92.63%

**HP Coverage (≥95% codes, ≥5 cases):** 414/475 = **87.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 40/475 = **8.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00103 | 40 | Blepharoplasty |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00142 | 350 | 99.7% | 1x → 00140 | Lens procedures (cataract) |
| 00300 | 24 | 95.8% | 1x → 00103 | Head/neck vessels |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00140 | 58 | 44.8% | 31x → 00142, 1x → 00144 | Eye (not lens) |


---

## RIV

**Lines:** 449  **CPT Changes:** 23  **Accuracy:** 94.88%

**HP Coverage (≥95% codes, ≥5 cases):** 194/449 = **43.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 152/449 = **33.9%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 37 | Upper GI endoscopy |
| 00811 | 32 | Colonoscopy diagnostic |
| 64447 | 15 | Femoral nerve block |
| 00790 | 14 | Intra-abdominal upper |
| 01638 | 8 | Shoulder arthroplasty |
| 01830 | 8 | Open femur/hip fracture |
| 00918 | 7 | Transurethral procedures |
| 01402 | 6 | Knee arthroplasty |
| 01214 | 5 | Total hip arthroplasty |
| 00300 | 5 | Head/neck vessels |
| 64466 | 5 |  |
| 64445 | 5 | Sciatic nerve block |
| 01630 | 5 | Shoulder open |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 64488 | 42 | 97.6% | 1x → 64486 | TAP block continuous |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01810 | 12 | 66.7% | 2x → 01830, 2x → 01710 | Hip/femur procedures |
| 00902 | 5 | 80.0% | 1x → 00812 | Anorectal surgery |
| 00400 | 9 | 88.9% | 1x → 00402 | Integumentary (extremity/torso) |
| 00840 | 19 | 89.5% | 1x → 00752, 1x → 00851 | Intra-abdominal lower |
| 01480 | 12 | 91.7% | 1x → 99202 | Lower leg/ankle/foot |
| 00813 | 31 | 93.5% | 1x → 00811, 1x → 00731 | Colonoscopy screening→diagnostic |
| 64415 | 17 | 94.1% | 1x → 64417 | Brachial plexus block |
| 00812 | 86 | 94.2% | 5x → 00811 | Colonoscopy screening |


---

## MKI

**Lines:** 389  **CPT Changes:** 22  **Accuracy:** 94.34%

**HP Coverage (≥95% codes, ≥5 cases):** 278/389 = **71.5%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 171/389 = **44.0%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 64415 | 66 | Brachial plexus block |
| 64447 | 47 | Femoral nerve block |
| 01630 | 30 | Shoulder open |
| 00170 | 12 | Intraoral procedures |
| 00160 | 9 | Nose/sinus |
| 01638 | 7 | Shoulder arthroplasty |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 01810 | 66 | 95.5% | 2x → 00400, 1x → 01830 | Hip/femur procedures |
| 01400 | 41 | 97.6% | 1x → 01320 | Knee arthroscopy |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01710 | 11 | 18.2% | 4x → 01810, 4x → 01714, 1x → 01712 | Elbow/forearm |
| 00126 | 9 | 88.9% | 1x → 00170 | Cataract surgery |
| 01480 | 10 | 90.0% | 1x → 64445 | Lower leg/ankle/foot |
| 00630 | 11 | 90.9% | 1x → 00670 | Lumbar laminectomy |
| 64445 | 17 | 94.1% | 1x → 64473 | Sciatic nerve block |
| 01830 | 19 | 94.7% | 1x → 01810 | Open femur/hip fracture |


---

## INJE-CLIK

**Lines:** 388  **CPT Changes:** 19  **Accuracy:** 95.10%

**HP Coverage (≥95% codes, ≥5 cases):** 360/388 = **92.8%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 0/388 = **0.0%** of lines code-perfect.

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00142 | 360 | 96.4% | 12x → 00140, 1x → 00103 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00140 | 26 | 76.9% | 6x → 00142 | Eye (not lens) |


---

## AHG

**Lines:** 380  **CPT Changes:** 52  **Accuracy:** 86.32%

**HP Coverage (≥95% codes, ≥5 cases):** 161/380 = **42.4%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 161/380 = **42.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00813 | 52 | Colonoscopy screening→diagnostic |
| 00731 | 41 | Upper GI endoscopy |
| 00902 | 16 | Anorectal surgery |
| 01939 | 12 | Vertebroplasty/kyphoplasty |
| 01940 | 12 | Pain mgmt continuous |
| 00300 | 11 | Head/neck vessels |
| 01937 | 11 | Pain mgmt percutaneous |
| 01400 | 6 | Knee arthroscopy |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01992 | 16 | 6.2% | 13x → 01938, 1x → 01940, 1x → 01937 | Pain mgmt continuous regional |
| 00400 | 8 | 62.5% | 2x → 00300, 1x → 01942 | Integumentary (extremity/torso) |
| 01938 | 27 | 81.5% | 2x → 01992, 1x → 01939, 1x → 01940, 1x → 01937 | Pain mgmt epidural injection |
| 00811 | 6 | 83.3% | 1x → 00812 | Colonoscopy diagnostic |
| 01470 | 6 | 83.3% | 1x → 01480 | Upper leg embolectomy |
| 00812 | 96 | 88.5% | 11x → 00811 | Colonoscopy screening |
| 00790 | 9 | 88.9% | 1x → 00750 | Intra-abdominal upper |
| 01480 | 12 | 91.7% | 1x → 01470 | Lower leg/ankle/foot |


---

## DUN

**Lines:** 373  **CPT Changes:** 46  **Accuracy:** 87.67%

**HP Coverage (≥95% codes, ≥5 cases):** 176/373 = **47.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 106/373 = **28.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01967 | 21 | C-section (planned) |
| 64488 | 11 | TAP block continuous |
| 01480 | 9 | Lower leg/ankle/foot |
| 00400 | 8 | Integumentary (extremity/torso) |
| 01402 | 8 | Knee arthroplasty |
| 64415 | 7 | Brachial plexus block |
| 64447 | 7 | Femoral nerve block |
| 01112 | 7 | Bone marrow aspirate |
| 00902 | 6 | Anorectal surgery |
| 00126 | 6 | Cataract surgery |
| 01961 | 6 | C-section (emergency) |
| 00522 | 5 | Pleural needle |
| 01400 | 5 | Knee arthroscopy |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00731 | 36 | 97.2% | 1x → 00532 | Upper GI endoscopy |
| 00813 | 34 | 97.1% | 1x → 00811 | Colonoscopy screening→diagnostic |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01520 | 6 | 0.0% | 6x → 01930 | Lower leg vein |
| 00532 | 8 | 37.5% | 4x → 36569, 1x → 00522 | Vascular access |
| 00300 | 6 | 66.7% | 1x → 00520, 1x → 00862 | Head/neck vessels |
| 00840 | 16 | 68.8% | 3x → 00944, 1x → 00752, 1x → 00790 | Intra-abdominal lower |
| 01810 | 8 | 75.0% | 1x → 00400, 1x → 01830 | Hip/femur procedures |
| 00790 | 14 | 78.6% | 2x → 00750, 1x → 00752 | Intra-abdominal upper |
| 01470 | 6 | 83.3% | 1x → 00400 | Upper leg embolectomy |
| 00812 | 19 | 84.2% | 2x → 00811, 1x → 00300 | Colonoscopy screening |
| 64450 | 9 | 88.9% | 1x → 64488 | Other peripheral nerve block |
| 00811 | 18 | 94.4% | 1x → 00812 | Colonoscopy diagnostic |


---

## ANA-ORA

**Lines:** 348  **CPT Changes:** 18  **Accuracy:** 94.83%

**HP Coverage (≥95% codes, ≥5 cases):** 279/348 = **80.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 212/348 = **60.9%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01402 | 52 | Knee arthroplasty |
| 64415 | 51 | Brachial plexus block |
| 01630 | 32 | Shoulder open |
| 01400 | 24 | Knee arthroscopy |
| 01214 | 20 | Total hip arthroplasty |
| 01638 | 11 | Shoulder arthroplasty |
| 64445 | 9 | Sciatic nerve block |
| 01740 | 7 | Elbow arthroplasty |
| 01470 | 6 | Upper leg embolectomy |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 64447 | 67 | 95.5% | 3x → 64473 | Femoral nerve block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 64450 | 7 | 0.0% | 7x → 64473 | Other peripheral nerve block |
| 01710 | 5 | 60.0% | 2x → 01810 | Elbow/forearm |
| 01810 | 11 | 81.8% | 2x → 01830 | Hip/femur procedures |
| 01480 | 16 | 93.8% | 1x → 01464 | Lower leg/ankle/foot |


---

## TQA-ARSC

**Lines:** 344  **CPT Changes:** 26  **Accuracy:** 92.44%

**HP Coverage (≥95% codes, ≥5 cases):** 146/344 = **42.4%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 146/344 = **42.4%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 55 | Lens procedures (cataract) |
| 64445 | 13 | Sciatic nerve block |
| 00813 | 12 | Colonoscopy screening→diagnostic |
| 64447 | 11 | Femoral nerve block |
| 00731 | 10 | Upper GI endoscopy |
| 00952 | 10 | Lumbar epidural labor |
| 00170 | 7 | Intraoral procedures |
| 64415 | 6 | Brachial plexus block |
| 01400 | 6 | Knee arthroscopy |
| 00145 | 6 | Eye (vitrectomy) |
| 01830 | 5 | Open femur/hip fracture |
| 00400 | 5 | Integumentary (extremity/torso) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01810 | 20 | 75.0% | 4x → 01830, 1x → 00400 | Hip/femur procedures |
| 00160 | 5 | 80.0% | 1x → 00170 | Nose/sinus |
| 00811 | 30 | 83.3% | 5x → 00812 | Colonoscopy diagnostic |
| 01480 | 47 | 89.4% | 3x → 01474, 2x → 01470 | Lower leg/ankle/foot |
| 00812 | 71 | 90.1% | 6x → 00811, 1x → 01480 | Colonoscopy screening |


---

## APO-UPM

**Lines:** 264  **CPT Changes:** 35  **Accuracy:** 86.74%

**HP Coverage (≥95% codes, ≥5 cases):** 126/264 = **47.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 54/264 = **20.5%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00813 | 19 | Colonoscopy screening→diagnostic |
| 00731 | 9 | Upper GI endoscopy |
| 01214 | 8 | Total hip arthroplasty |
| 00902 | 7 | Anorectal surgery |
| 00170 | 6 | Intraoral procedures |
| 64488 | 5 | TAP block continuous |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 72 | 98.6% | 1x → 00811 | Colonoscopy screening |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01810 | 5 | 80.0% | 1x → 01710 | Hip/femur procedures |
| 00811 | 28 | 82.1% | 5x → 00812 | Colonoscopy diagnostic |
| 00840 | 17 | 88.2% | 2x → 00851 | Intra-abdominal lower |
| 00790 | 14 | 92.9% | 1x → 00840 | Intra-abdominal upper |


---

## INJE-NRSC

**Lines:** 249  **CPT Changes:** 6  **Accuracy:** 97.59%

**HP Coverage (≥95% codes, ≥5 cases):** 207/249 = **83.1%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 207/249 = **83.1%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 64415 | 35 | Brachial plexus block |
| 01400 | 31 | Knee arthroscopy |
| 64447 | 30 | Femoral nerve block |
| 01402 | 29 | Knee arthroplasty |
| 01630 | 17 | Shoulder open |
| 01830 | 14 | Open femur/hip fracture |
| 01214 | 13 | Total hip arthroplasty |
| 00170 | 13 | Intraoral procedures |
| 00160 | 11 | Nose/sinus |
| 01638 | 8 | Shoulder arthroplasty |
| 01480 | 6 | Lower leg/ankle/foot |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01810 | 17 | 82.4% | 2x → 01830, 1x → 01710 | Hip/femur procedures |


---

## INJE-CSC

**Lines:** 248  **CPT Changes:** 6  **Accuracy:** 97.58%

**HP Coverage (≥95% codes, ≥5 cases):** 225/248 = **90.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 0/248 = **0.0%** of lines code-perfect.

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00142 | 225 | 99.6% | 1x → 00144 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00140 | 22 | 77.3% | 4x → 00142, 1x → 00144 | Eye (not lens) |


---

## STA-HLX

**Lines:** 236  **CPT Changes:** 9  **Accuracy:** 96.19%

**HP Coverage (≥95% codes, ≥5 cases):** 195/236 = **82.6%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 7/236 = **3.0%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00144 | 7 | Eye corneal transplant |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00142 | 188 | 99.5% | 1x → 00140 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00140 | 41 | 80.5% | 7x → 00142, 1x → 00144 | Eye (not lens) |


---

## GAP-UMSC

**Lines:** 220  **CPT Changes:** 9  **Accuracy:** 95.91%

**HP Coverage (≥95% codes, ≥5 cases):** 155/220 = **70.5%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 101/220 = **45.9%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01810 | 26 | Hip/femur procedures |
| 01400 | 23 | Knee arthroscopy |
| 01480 | 11 | Lower leg/ankle/foot |
| 01630 | 11 | Shoulder open |
| 00170 | 8 | Intraoral procedures |
| 00790 | 6 | Intra-abdominal upper |
| 00830 | 6 | Hernia lower abd |
| 00400 | 5 | Integumentary (extremity/torso) |
| 00731 | 5 | Upper GI endoscopy |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00812 | 54 | 98.1% | 1x → 00811 | Colonoscopy screening |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00811 | 18 | 94.4% | 1x → 00812 | Colonoscopy diagnostic |


---

## IAS-NSC

**Lines:** 215  **CPT Changes:** 12  **Accuracy:** 94.42%

**HP Coverage (≥95% codes, ≥5 cases):** 174/215 = **80.9%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 174/215 = **80.9%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 59 | Lens procedures (cataract) |
| 64447 | 25 | Femoral nerve block |
| 01402 | 20 | Knee arthroplasty |
| 01400 | 17 | Knee arthroscopy |
| 64415 | 16 | Brachial plexus block |
| 01214 | 11 | Total hip arthroplasty |
| 64473 | 11 | Lumbar plexus block |
| 01630 | 10 | Shoulder open |
| 01942 | 5 | CT/interventional radiology |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01810 | 6 | 50.0% | 2x → 01830, 1x → 01710 | Hip/femur procedures |


---

## EAP-CMI

**Lines:** 176  **CPT Changes:** 6  **Accuracy:** 96.59%

**HP Coverage (≥95% codes, ≥5 cases):** 107/176 = **60.8%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 107/176 = **60.8%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00160 | 28 | Nose/sinus |
| 01400 | 22 | Knee arthroscopy |
| 01630 | 16 | Shoulder open |
| 00170 | 16 | Intraoral procedures |
| 01480 | 7 | Lower leg/ankle/foot |
| 01937 | 7 | Pain mgmt percutaneous |
| 00126 | 6 | Cataract surgery |
| 00630 | 5 | Lumbar laminectomy |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01938 | 18 | 88.9% | 2x → 01992 | Pain mgmt epidural injection |
| 01810 | 13 | 92.3% | 1x → 01830 | Hip/femur procedures |


---

## PRI-CRS

**Lines:** 176  **CPT Changes:** 6  **Accuracy:** 96.59%

**HP Coverage (≥95% codes, ≥5 cases):** 159/176 = **90.3%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 133/176 = **75.6%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01938 | 19 | Pain mgmt epidural injection |
| 01400 | 19 | Knee arthroscopy |
| 01214 | 17 | Total hip arthroplasty |
| 01402 | 17 | Knee arthroplasty |
| 64473 | 15 | Lumbar plexus block |
| 64415 | 13 | Brachial plexus block |
| 64450 | 7 | Other peripheral nerve block |
| 01810 | 7 | Hip/femur procedures |
| 01940 | 7 | Pain mgmt continuous |
| 01630 | 7 | Shoulder open |
| 01638 | 5 | Shoulder arthroplasty |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 64447 | 26 | 96.2% | 1x → 64473 | Femoral nerve block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01937 | 5 | 80.0% | 1x → 01992 | Pain mgmt percutaneous |


---

## NTA-ASCOV

**Lines:** 151  **CPT Changes:** 5  **Accuracy:** 96.69%

**HP Coverage (≥95% codes, ≥5 cases):** 106/151 = **70.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 106/151 = **70.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 31 | Lens procedures (cataract) |
| 00145 | 26 | Eye (vitrectomy) |
| 01402 | 11 | Knee arthroplasty |
| 01810 | 11 | Hip/femur procedures |
| 01630 | 9 | Shoulder open |
| 01400 | 6 | Knee arthroscopy |
| 00873 | 6 | Lithotripsy/cystoscopy |
| 64447 | 6 | Femoral nerve block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00912 | 5 | 80.0% | 1x → 00910 | Transurethral resection |
| 00140 | 10 | 90.0% | 1x → 00142 | Eye (not lens) |


---

## EAP-TIN

**Lines:** 150  **CPT Changes:** 5  **Accuracy:** 96.67%

**HP Coverage (≥95% codes, ≥5 cases):** 112/150 = **74.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 90/150 = **60.0%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 65 | Lens procedures (cataract) |
| 01480 | 7 | Lower leg/ankle/foot |
| 00812 | 7 | Colonoscopy screening |
| 00170 | 6 | Intraoral procedures |
| 64447 | 5 | Femoral nerve block |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 00140 | 22 | 95.5% | 1x → 00142 | Eye (not lens) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00120 | 5 | 80.0% | 1x → 00124 | Eye (external) |
| 00126 | 12 | 83.3% | 2x → 00120 | Cataract surgery |


---

## APO-CVO

**Lines:** 117  **CPT Changes:** 7  **Accuracy:** 94.02%

**HP Coverage (≥95% codes, ≥5 cases):** 58/117 = **49.6%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 58/117 = **49.6%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 64415 | 13 | Brachial plexus block |
| 01400 | 11 | Knee arthroscopy |
| 01402 | 10 | Knee arthroplasty |
| 01630 | 10 | Shoulder open |
| 01940 | 8 | Pain mgmt continuous |
| 01480 | 6 | Lower leg/ankle/foot |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 01830 | 5 | 80.0% | 1x → 01820 | Open femur/hip fracture |
| 01810 | 8 | 87.5% | 1x → 01830 | Hip/femur procedures |
| 64447 | 18 | 94.4% | 1x → 64473 | Femoral nerve block |


---

## ANA-CVO

**Lines:** 112  **CPT Changes:** 5  **Accuracy:** 95.54%

**HP Coverage (≥95% codes, ≥5 cases):** 51/112 = **45.5%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 51/112 = **45.5%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 64415 | 13 | Brachial plexus block |
| 01214 | 10 | Total hip arthroplasty |
| 01402 | 9 | Knee arthroplasty |
| 01630 | 8 | Shoulder open |
| 01810 | 6 | Hip/femur procedures |
| 64473 | 5 | Lumbar plexus block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 64447 | 22 | 90.9% | 2x → 64473 | Femoral nerve block |
| 01400 | 18 | 94.4% | 1x → 01392 | Knee arthroscopy |


---

## EAP-SCA

**Lines:** 110  **CPT Changes:** 12  **Accuracy:** 89.09%

**HP Coverage (≥95% codes, ≥5 cases):** 53/110 = **48.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 53/110 = **48.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00142 | 53 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00910 | 8 | 62.5% | 3x → 00918 | Bladder procedures |


---

## INJE-CHB

**Lines:** 109  **CPT Changes:** 9  **Accuracy:** 91.74%

**HP Coverage (≥95% codes, ≥5 cases):** 16/109 = **14.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 16/109 = **14.7%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01480 | 9 | Lower leg/ankle/foot |
| 64445 | 7 | Sciatic nerve block |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00811 | 6 | 66.7% | 2x → 00812 | Colonoscopy diagnostic |
| 00812 | 34 | 88.2% | 4x → 00811 | Colonoscopy screening |
| 64447 | 11 | 90.9% | 1x → 64473 | Femoral nerve block |


---

## EAP-JSC

**Lines:** 98  **CPT Changes:** 4  **Accuracy:** 95.92%

**HP Coverage (≥95% codes, ≥5 cases):** 73/98 = **74.5%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 50/98 = **51.0%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 01214 | 13 | Total hip arthroplasty |
| 01402 | 10 | Knee arthroplasty |
| 01630 | 9 | Shoulder open |
| 64445 | 6 | Sciatic nerve block |
| 64415 | 6 | Brachial plexus block |
| 64473 | 6 | Lumbar plexus block |

### 95–99% Precision (>=5 cases)

| CPT | Cases | Precision | Errors | Description |
|-----|-------|-----------|--------|-------------|
| 01400 | 23 | 95.7% | 1x → 01320 | Knee arthroscopy |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 64447 | 9 | 88.9% | 1x → 64473 | Femoral nerve block |


---

## EAP-PSC

**Lines:** 77  **CPT Changes:** 3  **Accuracy:** 96.10%

**HP Coverage (≥95% codes, ≥5 cases):** 51/77 = **66.2%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 51/77 = **66.2%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00731 | 13 | Upper GI endoscopy |
| 00813 | 13 | Colonoscopy screening→diagnostic |
| 00142 | 11 | Lens procedures (cataract) |
| 00811 | 9 | Colonoscopy diagnostic |
| 00160 | 5 | Nose/sinus |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00812 | 18 | 88.9% | 2x → 00811 | Colonoscopy screening |


---

## PRI-RSL

**Lines:** 72  **CPT Changes:** 2  **Accuracy:** 97.22%

**HP Coverage (≥95% codes, ≥5 cases):** 66/72 = **91.7%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 66/72 = **91.7%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00103 | 29 | Blepharoplasty |
| 00145 | 20 | Eye (vitrectomy) |
| 00142 | 17 | Lens procedures (cataract) |

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00140 | 5 | 60.0% | 2x → 00142 | Eye (not lens) |


---

## ARKMETH

**Lines:** 53  **CPT Changes:** 12  **Accuracy:** 77.36%

**HP Coverage (≥95% codes, ≥5 cases):** 0/53 = **0.0%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 0/53 = **0.0%** of lines code-perfect.

### Problem Codes (<95%, >=5 cases)

| CPT | Cases | Precision | Primary Errors | Description |
|-----|-------|-----------|----------------|-------------|
| 00813 | 6 | 0.0% | 5x → 00731, 1x → 00811 | Colonoscopy screening→diagnostic |
| 00812 | 6 | 66.7% | 2x → 00811 | Colonoscopy screening |


---

## AIP

**Lines:** 47  **CPT Changes:** 4  **Accuracy:** 91.49%

**HP Coverage (≥95% codes, ≥5 cases):** 26/47 = **55.3%** of all billed lines could be auto-coded.
**100% Precision Coverage:** 26/47 = **55.3%** of lines code-perfect.

### 100% Precision (>=5 cases)

| CPT | Cases | Description |
|-----|-------|-------------|
| 00812 | 8 | Colonoscopy screening |
| 00918 | 7 | Transurethral procedures |
| 64447 | 6 | Femoral nerve block |
| 01402 | 5 | Knee arthroplasty |


---

## Appendix: UNI Groups (from prior analysis)

> The UNI groups below (UNI-GOLD, UNI-ROB, UNI-RSC) are not part of the daily SharePoint integration list and were not in the March+April fetch. These sections come from the earlier ANALYSIS.md run with their own batch ranges. Included for completeness. UNI-INTEG appears in the main report above.


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

