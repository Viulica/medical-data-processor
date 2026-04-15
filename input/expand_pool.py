"""Expand the auto-advance pool using location-aware slicing.

Two views per (group, location, CPT) bucket:
- CPT-safe: CPT code wasn't changed (this is the bet for 'auto-bill CPT')
- Clean:    no CPT/ICD/Modifier changes (strict auto-advance)

Thresholds: rate >= 95%, n >= 3.
"""
import csv
import glob
import os
import re
from collections import defaultdict

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_cache")
RATE = 0.95
MIN_N = 3

CRIT = re.compile(r"^(cpt|icd[1-4]|modifier[1-4])\b", re.IGNORECASE)
CPT_CH = re.compile(r"^cpt\b", re.IGNORECASE)

def read_pipe(path):
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8-sig", errors="replace").replace("\r", "")
    lines = [ln for ln in raw.split("\n") if ln]
    if not lines: return []
    header = lines[0].split("|")
    return [dict(zip(header, ln.split("|"))) for ln in lines[1:]]

cpt_changed = set()
critical_changed = set()
for p in glob.glob(os.path.join(CACHE, "*__changes.txt")):
    for r in read_pipe(p):
        ch = r.get("Change", "").strip()
        k = (r.get("WorktrackerGroupName",""),
             r.get("WorktrackerBatchNo",""),
             r.get("AccountNumber",""))
        if CPT_CH.match(ch): cpt_changed.add(k)
        if CRIT.match(ch):   critical_changed.add(k)

primary = {}
for p in glob.glob(os.path.join(CACHE, "*__all.txt")):
    for r in read_pipe(p):
        k = (r.get("WorktrackerGroupName",""),
             r.get("WorktrackerBatchNo",""),
             r.get("AccountNumber",""))
        try: amt = float(r.get("ChargeAmount","0") or 0)
        except: amt = 0
        cur = primary.get(k)
        if cur is None or amt > cur[0]:
            primary[k] = (amt, r.get("CPT",""), r.get("Location",""))

b = defaultdict(lambda: {"total":0, "cpt_ok":0, "clean":0})
for k, (_, cpt, loc) in primary.items():
    grp = k[0]
    bk = (grp, loc, cpt)
    b[bk]["total"] += 1
    if k not in cpt_changed:      b[bk]["cpt_ok"] += 1
    if k not in critical_changed: b[bk]["clean"]  += 1

def qualify(mode):
    field = "cpt_ok" if mode == "cpt" else "clean"
    out = []
    for k, v in b.items():
        if v["total"] < MIN_N: continue
        rate = v[field] / v["total"]
        if rate >= RATE:
            out.append((k, v["total"], v[field], rate))
    out.sort(key=lambda r: -r[1])
    return out

cpt_pool = qualify("cpt")
clean_pool = qualify("clean")

# Per-batch coverage
def coverage(pool):
    safe_keys = {k for k, *_ in pool}
    per_batch = defaultdict(lambda: {"total":0, "safe":0})
    for acct, (_, cpt, loc) in primary.items():
        bk = (acct[0], acct[1])
        per_batch[bk]["total"] += 1
        if (acct[0], loc, cpt) in safe_keys:
            per_batch[bk]["safe"] += 1
    total = sum(v["total"] for v in per_batch.values())
    safe  = sum(v["safe"]  for v in per_batch.values())
    pcts = sorted(v["safe"]/v["total"] for v in per_batch.values() if v["total"])
    mean = sum(pcts)/len(pcts)*100 if pcts else 0
    med = pcts[len(pcts)//2]*100 if pcts else 0
    over10 = sum(1 for p in pcts if p>=0.10)
    over25 = sum(1 for p in pcts if p>=0.25)
    over50 = sum(1 for p in pcts if p>=0.50)
    return total, safe, mean, med, over10, over25, over50, per_batch

for mode, pool in (("CPT-safe", cpt_pool), ("Critical-clean", clean_pool)):
    total, safe, mean, med, o10, o25, o50, _ = coverage(pool)
    cases = sum(p[1] for p in pool)
    ok = sum(p[2] for p in pool)
    print(f"\n=== {mode} pool (>={int(RATE*100)}% rate, n>={MIN_N}) ===")
    print(f"  Buckets: {len(pool)}")
    print(f"  Historical cases in pool: {cases}  (pool internal accuracy: {ok/cases*100:.1f}%)")
    print(f"  Pool share of total volume: {safe}/{total} = {safe/total*100:.1f}%")
    print(f"  Per-batch coverage: mean={mean:.1f}%  median={med:.1f}%")
    print(f"  Batches >=10% covered: {o10}   >=25%: {o25}   >=50%: {o50}")

# Write CPT-safe pool CSV (bigger pool, useful for auto-CPT-advance)
with open("cpt_safe_pool.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["group", "location", "cpt", "total", "cpt_ok", "rate"])
    for (g, loc, cpt), tot, ok, rate in cpt_pool:
        w.writerow([g, loc, cpt, tot, ok, f"{rate:.4f}"])
print(f"\nWrote cpt_safe_pool.csv")

# Per-group expansion comparison
print("\n=== Per-group coverage (CPT-safe pool) — top 25 by total ===")
_, _, _, _, _, _, _, per_batch = coverage(cpt_pool)
grp = defaultdict(lambda: {"total":0, "safe":0})
for (g, _bn), v in per_batch.items():
    grp[g]["total"] += v["total"]
    grp[g]["safe"] += v["safe"]
rows = sorted(grp.items(), key=lambda x: -x[1]["total"])[:25]
print(f"  {'Group':<14} {'cases':>6} {'CPT-safe':>8} {'%':>6}")
for g, v in rows:
    print(f"  {g:<14} {v['total']:>6} {v['safe']:>8} {v['safe']/v['total']*100:>5.1f}%")

# Show the top NEW buckets the CPT-safe view adds vs the old strict clean view
new_buckets = set(k for k,*_ in cpt_pool) - set(k for k,*_ in clean_pool)
print(f"\n=== New buckets unlocked by CPT-safe view ({len(new_buckets)}) — top 25 by volume ===")
rows = [(k, b[k]["total"], b[k]["cpt_ok"], b[k]["clean"]) for k in new_buckets]
rows.sort(key=lambda r: -r[1])
print(f"  {'Group':<12} {'Location':<40} {'CPT':>6} {'n':>4} {'cptOK':>6} {'strict':>6}")
for k, tot, cok, cln in rows[:25]:
    print(f"  {k[0]:<12} {k[1][:40]:<40} {k[2]:>6} {tot:>4} {cok:>6} {cln:>6}")
