"""Find (group, location, CPT) buckets that are historically error-free.

A case is "clean" if the change log has zero entries for that AccountNumber.
For each bucket, report total cases and clean-rate. Focus on high-volume + 100% clean.
"""
import glob
import os
from collections import defaultdict

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_cache")

def read_pipe(path):
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8-sig", errors="replace").replace("\r", "")
    lines = [ln for ln in raw.split("\n") if ln]
    if not lines:
        return []
    header = lines[0].split("|")
    return [dict(zip(header, ln.split("|"))) for ln in lines[1:]]


# Which accounts had ANY change logged?
dirty_accounts = set()  # (group, batch, account)
for p in glob.glob(os.path.join(CACHE, "*__changes.txt")):
    for r in read_pipe(p):
        if r.get("Change", "").strip():
            dirty_accounts.add((r.get("WorktrackerGroupName", ""),
                                r.get("WorktrackerBatchNo", ""),
                                r.get("AccountNumber", "")))

# Pick primary charge line per account (highest ChargeAmount) from __all.txt
primary = {}
for p in glob.glob(os.path.join(CACHE, "*__all.txt")):
    for r in read_pipe(p):
        key = (r.get("WorktrackerGroupName", ""),
               r.get("WorktrackerBatchNo", ""),
               r.get("AccountNumber", ""))
        try:
            amt = float(r.get("ChargeAmount", "0") or 0)
        except ValueError:
            amt = 0.0
        cur = primary.get(key)
        if cur is None or amt > cur[0]:
            primary[key] = (amt, r.get("CPT", ""), r.get("Location", ""))

# Bucket by (group, location, cpt)
buckets = defaultdict(lambda: {"total": 0, "clean": 0})
for acct_key, (_amt, cpt, loc) in primary.items():
    grp = acct_key[0]
    bucket = (grp, loc, cpt)
    buckets[bucket]["total"] += 1
    if acct_key not in dirty_accounts:
        buckets[bucket]["clean"] += 1

# Also bucket by (group, cpt) — simpler view
buckets_gc = defaultdict(lambda: {"total": 0, "clean": 0})
for acct_key, (_amt, cpt, _loc) in primary.items():
    bucket = (acct_key[0], cpt)
    buckets_gc[bucket]["total"] += 1
    if acct_key not in dirty_accounts:
        buckets_gc[bucket]["clean"] += 1

def print_buckets(title, buckets, min_total, only_perfect=False):
    print(f"\n=== {title} (min volume {min_total}{'; 100% clean only' if only_perfect else ''}) ===")
    rows = []
    for key, v in buckets.items():
        if v["total"] < min_total: continue
        rate = v["clean"] / v["total"]
        if only_perfect and rate < 1.0: continue
        rows.append((key, v["total"], v["clean"], rate))
    rows.sort(key=lambda r: (-r[3], -r[1]))
    print(f"  {'Bucket':<60} {'n':>4} {'clean':>5} {'rate':>7}")
    for key, tot, cln, rate in rows[:50]:
        label = " | ".join(str(x) for x in key)
        print(f"  {label:<60} {tot:>4} {cln:>5} {rate*100:>6.1f}%")
    print(f"  ({len(rows)} buckets total)")
    auto = sum(tot for _, tot, _, _ in rows)
    print(f"  Volume auto-processable at this threshold: {auto} cases")

print(f"Total cases:           {len(primary)}")
print(f"Dirty (had any change): {len(dirty_accounts)}")
print(f"Clean:                  {len(primary) - sum(1 for k in primary if k in dirty_accounts)}")

print_buckets("Group + CPT — 100% clean, n>=10", buckets_gc, 10, True)
print_buckets("Group + CPT — 100% clean, n>=5",  buckets_gc, 5, True)
print_buckets("Group + Location + CPT — 100% clean, n>=5", buckets, 5, True)

# Also: top group+cpt buckets by absolute clean volume
print("\n=== Top (group, CPT) buckets by CLEAN volume (any rate) ===")
rows = [(k, v["total"], v["clean"], v["clean"]/v["total"]) for k, v in buckets_gc.items() if v["total"] >= 10]
rows.sort(key=lambda r: -r[2])
print(f"  {'Bucket':<40} {'n':>4} {'clean':>5} {'rate':>7}")
for key, tot, cln, rate in rows[:30]:
    label = " | ".join(key)
    print(f"  {label:<40} {tot:>4} {cln:>5} {rate*100:>6.1f}%")
