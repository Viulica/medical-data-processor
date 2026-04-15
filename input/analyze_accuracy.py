"""Total CPT and ICD1 accuracy across all cached CHARGE batches.

Logic: if a field wasn't changed, the AI got it right.
- CPT denominator: every charge line in __all.txt files (per account + service code).
- ICD1 denominator: every account (unique AccountNumber across all __all.txt files).
- Numerator (errors): count of matching change-log entries in __changes.txt files.
"""
import glob
import os
import re
from collections import defaultdict

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_cache")


def read_pipe(path):
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8-sig", errors="replace").replace("\r", "")
    lines = [ln for ln in raw.split("\n") if ln]
    if not lines:
        return [], []
    header = lines[0].split("|")
    rows = [dict(zip(header, ln.split("|"))) for ln in lines[1:]]
    return header, rows


# ---- Denominators from __all.txt ----
charge_lines = set()   # (group, batch, account, cpt) — one per charge line
accounts = set()       # (group, batch, account) — one per case

for path in sorted(glob.glob(os.path.join(CACHE, "*__all.txt"))):
    _, rows = read_pipe(path)
    for r in rows:
        key_acc = (r.get("WorktrackerGroupName", ""), r.get("WorktrackerBatchNo", ""), r.get("AccountNumber", ""))
        accounts.add(key_acc)
        charge_lines.add(key_acc + (r.get("CPT", ""),))

# ---- Errors from __changes.txt ----
cpt_changed = set()    # (group, batch, account, service_code)
icd1_changed = set()   # (group, batch, account)

cpt_pat = re.compile(r"\bCPT\b.*changed", re.IGNORECASE)
icd1_pat = re.compile(r"\bIcd1\b.*changed", re.IGNORECASE)

for path in sorted(glob.glob(os.path.join(CACHE, "*__changes.txt"))):
    _, rows = read_pipe(path)
    for r in rows:
        change = r.get("Change", "")
        key_acc = (r.get("WorktrackerGroupName", ""), r.get("WorktrackerBatchNo", ""), r.get("AccountNumber", ""))
        if cpt_pat.search(change):
            cpt_changed.add(key_acc + (r.get("ServiceCode", ""),))
        if icd1_pat.search(change):
            icd1_changed.add(key_acc)

# ---- Report ----
cpt_total = len(charge_lines)
cpt_err = len(cpt_changed)
icd_total = len(accounts)
icd_err = len(icd1_changed)

def pct(n, d):
    return f"{(1 - n/d) * 100:.2f}%" if d else "n/a"

print(f"Total batches cached:    {len(glob.glob(os.path.join(CACHE, '*__all.txt')))}")
print(f"Total unique accounts:   {icd_total:>6}")
print(f"Total charge lines:      {cpt_total:>6}")
print()
print(f"CPT  accuracy:  {pct(cpt_err, cpt_total):>7}  ({cpt_total - cpt_err}/{cpt_total}; {cpt_err} changed)")
print(f"ICD1 accuracy:  {pct(icd_err, icd_total):>7}  ({icd_total - icd_err}/{icd_total}; {icd_err} changed)")

# ---- Per-group breakdown ----
print("\n--- Per-group ---")
group_acc = defaultdict(set)
group_lines = defaultdict(set)
for g, b, a in accounts:
    group_acc[g].add((b, a))
for g, b, a, c in charge_lines:
    group_lines[g].add((b, a, c))
group_cpt_err = defaultdict(set)
group_icd_err = defaultdict(set)
for g, b, a, sc in cpt_changed:
    group_cpt_err[g].add((b, a, sc))
for g, b, a in icd1_changed:
    group_icd_err[g].add((b, a))

print(f"{'Group':<18}{'Cases':>6} {'Lines':>6} {'CPT%':>8} {'ICD1%':>8}")
for g in sorted(group_acc):
    cases = len(group_acc[g])
    lines = len(group_lines[g])
    cpt_a = pct(len(group_cpt_err[g]), lines)
    icd_a = pct(len(group_icd_err[g]), cases)
    print(f"{g:<18}{cases:>6} {lines:>6} {cpt_a:>8} {icd_a:>8}")
