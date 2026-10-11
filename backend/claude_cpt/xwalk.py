#!/usr/bin/env python3
"""Crosswalk CLI for the Claude CPT coders (official 2025 ASA Crosswalk).

  python3 xwalk.py search <term> [<term> ...] [-- <term> ...]   rows whose surgical descriptor contains ALL terms;
                                                               separate several searches with --
  python3 xwalk.py lookup <anesthesia code> [<code> ...]        descriptor, coding notes, example procedures
  python3 xwalk.py surg <surgical CPT> [<CPT> ...]              exact surgical CPT -> anesthesia mapping
"""
import json
import os
import sys

import pandas as pd

XL = os.environ.get("XWALK_XLSX") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "2025 crosswalk.xlsx")


def _clean(v):
    s = "" if v is None else str(v).strip()
    return "" if s.lower() in ("nan", "none", "0.0") else s


def load():
    d = pd.read_excel(XL, sheet_name=0, header=8)
    d = d.rename(columns={'CPT Procedure Code': 'surg', 'CPT Procedure Descriptor': 'surg_desc',
                          'CPT Anesthesia Code': 'anes', 'CPT Anesthesia Descriptor': 'anes_desc',
                          'Base Unit Value': 'base_units', 'Alternates': 'alternates',
                          'Comment': 'comment', 'Instructi/Text': 'instruction'})
    d = d[d['surg_desc'].notna()].reset_index(drop=True)
    for c in ('surg', 'anes'):
        d[c] = d[c].astype(str).str.replace(r'\.0$', '', regex=True)
    d['sd'] = d['surg_desc'].str.lower()
    return d


def anes_desc_map(d):
    m = {}
    for _, r in d.iterrows():
        c, dsc = r['anes'], _clean(r.get('anes_desc'))
        if c and dsc and c not in m and 'NOT A PRIMARY' not in dsc.upper() and 'NOT TYPICALLY' not in dsc.upper():
            m[c] = dsc
    return m


def row(r, dm):
    out = {"surg": r['surg'], "surg_desc": _clean(r['surg_desc'])[:160], "anes": r['anes'],
           "anes_desc": _clean(r.get('anes_desc'))[:120]}
    bu = _clean(r.get('base_units'))
    if bu:
        out["base_units"] = bu
    alt = _clean(r.get('alternates'))
    if alt:
        out["alternates"] = [{"anes": c.strip(), "desc": dm.get(c.strip(), "")[:100]} for c in alt.split(",") if c.strip()]
    for col in ('comment', 'instruction'):
        v = _clean(r.get(col))
        if v:
            out[col] = v[:300]
    return out


def main(argv):
    if len(argv) < 3 or argv[1] not in ("search", "lookup", "surg"):
        print(__doc__)
        return 1
    d = load()
    dm = anes_desc_map(d)
    cmd, args = argv[1], argv[2:]
    if cmd == "search":
        queries, cur = [], []
        for a in args + ["--"]:
            if a == "--":
                if cur:
                    queries.append(cur)
                cur = []
            else:
                cur.append(a)
        res = []
        for terms in queries:
            m = pd.Series(True, index=d.index)
            for t in terms:
                m &= d['sd'].str.contains(t.lower(), regex=False, na=False)
            res.append({"terms": terms, "n_matches": int(m.sum()), "rows": [row(r, dm) for _, r in d[m].head(25).iterrows()]})
    elif cmd == "lookup":
        res = []
        for a in args:
            code = a.zfill(5)
            sub = d[d['anes'] == code]
            notes = []
            for _, r in sub.iterrows():
                for col in ('comment', 'instruction'):
                    v = _clean(r.get(col))
                    if v and v[:300] not in notes:
                        notes.append(v[:300])
            res.append({"anes": code, "found": not sub.empty, "anes_desc": dm.get(code, ""),
                        "num_surgical_procedures": len(sub), "coding_notes": notes[:8],
                        "examples": [f"{r['surg']}: {_clean(r['surg_desc'])[:90]}" for _, r in sub.head(15).iterrows()]})
    else:
        res = []
        for a in args:
            sub = d[d['surg'] == a.strip()]
            res.append({"surg": a, "found": not sub.empty, "rows": [row(r, dm) for _, r in sub.iterrows()]})
    if len(res) == 1:
        res = res[0]
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
