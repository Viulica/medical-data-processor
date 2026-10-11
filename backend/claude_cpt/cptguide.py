#!/usr/bin/env python3
"""Print the predicted-CPT block the production ICD prompt appends for a chart (same text as
predict_general._build_cpt_guidance_block), for one or more anesthesia CPT codes.

  python3 cptguide.py <code> [<code> ...]
"""
import re
import sys
from pathlib import Path

DESC = {}
for line in (Path(__file__).resolve().parent / "rules.md").read_text().splitlines():
    m = re.match(r"\s*(\d{5})\s+(.+)", line)
    if m and m.group(1) not in DESC:
        DESC[m.group(1)] = m.group(2).strip()


def block(code):
    description = DESC.get(code, "")
    cpt_line = f"CPT = {code}" + (f"  ({description})" if description else "")
    return (
        "\n\n"
        "================================================================================\n"
        "PREDICTED ANESTHESIA CPT FOR THIS CASE (supporting context only)\n"
        "================================================================================\n"
        f"{cpt_line}\n"
        "\n"
        "NOTE on the description above: it is the GENERIC anesthesia-CPT description pulled\n"
        "from a standard reference list (not specific to this patient). The CPT itself is just\n"
        "the anesthesia code for the broad procedure category — use it only as supporting\n"
        "context to narrow down the procedure type, NOT as the source of truth for the ICD.\n"
        "\n"
        "Your ICD codes must be chosen based on what is actually documented in this specific\n"
        "PDF (pre-op diagnosis, post-op diagnosis, indication, surgeon's note, etc.) — pick\n"
        "the ICD codes that medically justify the procedure performed on this patient.\n"
        "Do not let the generic CPT description override what the chart actually says.\n"
        "\n"
        "Apply any facility-specific CPT→ICD rules from the prompt above using this CPT.\n"
        "Sanity-check: ensure your ICD1 is clinically consistent with the procedure category\n"
        "(e.g., do NOT return a GI ICD when CPT is for an ortho procedure).\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for c in sys.argv[1:]:
        print(f"########## {c} ##########{block(c.strip().zfill(5))}")
