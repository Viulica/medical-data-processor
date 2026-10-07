# How to code each chart

Same method as the production crosswalk agent (cpt_agent.py), with the crosswalk tools exposed as a CLI.

## Tools (run with Bash from the job folder)
- `python3 xwalk.py search <term> [<term> ...]`: crosswalk rows whose surgical descriptor contains ALL terms (anesthesia code, base units, alternates with descriptors, ASA comments/instructions)
- `python3 xwalk.py lookup <anesthesia code>`: the code's descriptor, ASA coding notes, example surgical procedures
- `python3 xwalk.py surg <surgical CPT>`: exact surgical CPT -> anesthesia mapping (use any surgical CPT printed in the chart)
- `python3 pdftext.py <pdf>`: the chart's text, page by page. Pages marked NO TEXT LAYER (or garbled) must be viewed with the Read tool (pages parameter).

## Precedence (most important first)
1. The CRITICAL CODING RULES in rules.md ALWAYS win. Do not let the crosswalk override an explicit rule.
2. Group-specific custom instructions in rules.md ("If the procedure is X then code is Y") come next.
3. Use the crosswalk for everything not covered by 1-2, and to verify candidates.

## Mandatory crosswalk protocol
Never answer from memory. For every chart:
1. Read the procedure actually PERFORMED (op note, procedure note, anesthesia record), not just what was scheduled. Note site, approach (open/laparoscopic/percutaneous/endoscopic), depth, variant, and obstetric context (labor epidural vs cesarean vs vaginal delivery).
2. Search the crosswalk for the performed procedure (`search`, or `surg` with a printed surgical CPT).
3. Verify the final code AND at least one competing candidate with `lookup`.
4. Cite the crosswalk row you relied on.

## Alternate selection
When a row lists alternates, the primary is not automatically the answer: the choice is determined by the documented SITE, depth, approach and variant. Pick the code whose descriptor matches this case (e.g. breast 00402 not 00400, TURP 00914 not 00910, total knee 01402, instrumented spine 00670 not 00630, upper abdomen 00790 vs lower abdomen 00840). Never default to a "not otherwise specified" code when an alternate names the documented site or procedure.
Stay with the crosswalk primary unless the ASA note or the documented site/approach explicitly directs an alternate.

## Compare competing candidates
Before every answer, compare at least two candidate codes. One region can map to different codes by the TISSUE LAYER operated on: bone vs integumentary/soft tissue (skin, subcutaneous, mass, cyst, lesion, node) vs joint vs nerve/muscle/tendon/fascia. Read what structure the surgeon acted on, look up a candidate from each plausible layer, and state why the losers were rejected.

## Multiple procedures: higher base units wins
When two or more distinct procedures were done in one session, bill ONE code: the procedure with the most base units. Example: tympanostomy 69436 -> 00126 (4 units) plus adenoidectomy 42830 -> 00170 (5 units) gives 00170.

## Disambiguation
- Eye: 00140 eye NOS; 00142 lens (cataract, IOL, phaco); 00145 vitreoretinal. A cataract case is 00142.
- 01968 is an OB add-on code: never the main-line answer. Nerve blocks (64xxx) and lines (36xxx) are never the main line.

## Confidence
- high: the performed procedure is clearly documented and one code clearly fits.
- medium: thin documentation (no op note, coded from schedule/consent), a base-unit tie, or a rule vs crosswalk conflict.
- low: you are guessing between codes.
