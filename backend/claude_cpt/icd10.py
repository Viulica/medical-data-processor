#!/usr/bin/env python3
"""ICD-10-CM lookup for the Claude coders (official CMS code files, FY2026 + FY2027).

  python3 icd10.py check [--dos MM/DD/YYYY] <code> [<code> ...]   valid? billable on the date of service? description;
                                                                  for a non-billable header, its billable codes
  python3 icd10.py search <term> [<term> ...] [-- <term> ...]     billable codes whose description contains ALL terms;
                                                                  separate several searches with --
FY2027 applies to dates of service from 10/01/2026, FY2026 before that (default: FY2027).
"""
import gzip
import json
import os
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icd10cm_fy26_fy27.tsv.gz")


def load():
    rows = {}
    with gzip.open(DATA, "rt") as f:
        for line in f:
            code, b26, b27, desc = line.rstrip("\n").split("\t", 3)
            rows[code] = (b26, b27, desc)
    return rows


def dotted(code):
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def fiscal_year(dos):
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", dos or "")
    if not m:
        return "27"
    mo, _, yr = (int(x) for x in m.groups())
    return "27" if (yr, mo) >= (2026, 10) else "26"


def main(argv):
    if len(argv) < 3 or argv[1] not in ("check", "search"):
        print(__doc__)
        return 1
    rows = load()
    args = argv[2:]
    if argv[1] == "check":
        fy = "27"
        if args and args[0] == "--dos":
            fy, args = fiscal_year(args[1] if len(args) > 1 else ""), args[2:]
        res = []
        for a in args:
            code = a.strip().upper().replace(".", "")
            r = rows.get(code)
            if not r or not r[0 if fy == "26" else 1]:
                res.append({"code": a, "valid": False, "fiscal_year": f"FY20{fy}",
                            "note": "not in the ICD-10-CM code set for this date of service"})
                continue
            billable = r[0 if fy == "26" else 1] == "1"
            out = {"code": dotted(code), "valid": True, "billable": billable, "fiscal_year": f"FY20{fy}", "description": r[2]}
            if not billable:
                kids = [(c, v) for c, v in rows.items() if c.startswith(code) and c != code and v[0 if fy == "26" else 1] == "1"]
                out["billable_codes"] = [f"{dotted(c)} {v[2]}" for c, v in kids[:20]]
                if len(kids) > 20:
                    out["more_billable_codes"] = len(kids) - 20
            res.append(out)
    else:
        queries, cur = [], []
        for a in args + ["--"]:
            if a == "--":
                if cur:
                    queries.append(cur)
                cur = []
            else:
                cur.append(a.lower())
        res = []
        for terms in queries:
            hits = [f"{dotted(c)} {v[2]}" for c, v in rows.items() if v[1] == "1" and all(t in v[2].lower() for t in terms)]
            res.append({"terms": terms, "n_matches": len(hits), "codes": hits[:25]})
    print(json.dumps(res[0] if len(res) == 1 else res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
