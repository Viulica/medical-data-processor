"""Safe-bucket analysis using a meaningful 'dirty' definition.

Only flag an account as dirty if a billing-critical field changed:
  CPT, Icd1, Icd2, Icd3, Icd4, Modifier1, Modifier2, Modifier3, Modifier4
Excludes: Modifier Units (policy zeroing), Amount, Scanned Date,
  Charge Status, Rendering Doctor (downstream issue), times/units (rarely
  cause claim denials).
"""
import glob, os, re
from collections import defaultdict

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_cache")
CRITICAL = re.compile(r"^(cpt|icd[1-4]|modifier[1-4])\b", re.IGNORECASE)

def read_pipe(path):
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8-sig", errors="replace").replace("\r", "")
    lines = [ln for ln in raw.split("\n") if ln]
    if not lines: return []
    header = lines[0].split("|")
    return [dict(zip(header, ln.split("|"))) for ln in lines[1:]]

dirty_accounts = set()
for p in glob.glob(os.path.join(CACHE, "*__changes.txt")):
    for r in read_pipe(p):
        ch = r.get("Change", "").strip()
        if CRITICAL.match(ch):
            dirty_accounts.add((r.get("WorktrackerGroupName",""),
                                r.get("WorktrackerBatchNo",""),
                                r.get("AccountNumber","")))

primary = {}
for p in glob.glob(os.path.join(CACHE, "*__all.txt")):
    for r in read_pipe(p):
        key = (r.get("WorktrackerGroupName",""),
               r.get("WorktrackerBatchNo",""),
               r.get("AccountNumber",""))
        try:
            amt = float(r.get("ChargeAmount","0") or 0)
        except ValueError:
            amt = 0.0
        cur = primary.get(key)
        if cur is None or amt > cur[0]:
            primary[key] = (amt, r.get("CPT",""), r.get("Location",""))

buckets = defaultdict(lambda: {"total":0, "clean":0})
buckets_gc = defaultdict(lambda: {"total":0, "clean":0})
for acct_key, (_, cpt, loc) in primary.items():
    grp = acct_key[0]
    clean = acct_key not in dirty_accounts
    buckets[(grp, loc, cpt)]["total"] += 1
    buckets_gc[(grp, cpt)]["total"] += 1
    if clean:
        buckets[(grp, loc, cpt)]["clean"] += 1
        buckets_gc[(grp, cpt)]["clean"] += 1

print(f"Total cases: {len(primary)}")
print(f"Dirty (critical-field change): {len(dirty_accounts)}")
print(f"Clean: {len(primary) - sum(1 for k in primary if k in dirty_accounts)} "
      f"({(1 - len(dirty_accounts)/len(primary))*100:.1f}%)")

def report(title, data, min_total, rate_floor):
    rows = [(k, v["total"], v["clean"], v["clean"]/v["total"])
            for k, v in data.items() if v["total"] >= min_total and v["clean"]/v["total"] >= rate_floor]
    rows.sort(key=lambda r: (-r[3], -r[1]))
    print(f"\n=== {title} ({len(rows)} buckets; {sum(r[1] for r in rows)} cases) ===")
    print(f"  {'Bucket':<55} {'n':>4} {'clean':>5} {'rate':>7}")
    for key, tot, cln, rate in rows[:40]:
        label = " | ".join(str(x) for x in key)
        print(f"  {label[:55]:<55} {tot:>4} {cln:>5} {rate*100:>6.1f}%")

report("Group+CPT — 100% clean, n>=10",  buckets_gc, 10, 1.0)
report("Group+CPT — >=95% clean, n>=10", buckets_gc, 10, 0.95)
report("Group+CPT — >=90% clean, n>=10", buckets_gc, 10, 0.90)
report("Group+Location+CPT — 100% clean, n>=5", buckets, 5, 1.0)
report("Group+Location+CPT — >=95% clean, n>=5", buckets, 5, 0.95)
