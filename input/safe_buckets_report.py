"""Final safe-bucket report: all (group, location, CPT) buckets at >=95% clean.

Also writes a CSV for downstream use.
"""
import csv, glob, os, re
from collections import defaultdict

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_cache")
CRITICAL = re.compile(r"^(cpt|icd[1-4]|modifier[1-4])\b", re.IGNORECASE)
THRESHOLD = 0.95

def read_pipe(path):
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8-sig", errors="replace").replace("\r", "")
    lines = [ln for ln in raw.split("\n") if ln]
    if not lines: return []
    header = lines[0].split("|")
    return [dict(zip(header, ln.split("|"))) for ln in lines[1:]]

dirty = set()
for p in glob.glob(os.path.join(CACHE, "*__changes.txt")):
    for r in read_pipe(p):
        if CRITICAL.match(r.get("Change", "").strip()):
            dirty.add((r.get("WorktrackerGroupName",""),
                       r.get("WorktrackerBatchNo",""),
                       r.get("AccountNumber","")))

primary = {}
for p in glob.glob(os.path.join(CACHE, "*__all.txt")):
    for r in read_pipe(p):
        key = (r.get("WorktrackerGroupName",""),
               r.get("WorktrackerBatchNo",""),
               r.get("AccountNumber",""))
        try: amt = float(r.get("ChargeAmount","0") or 0)
        except ValueError: amt = 0.0
        cur = primary.get(key)
        if cur is None or amt > cur[0]:
            primary[key] = (amt, r.get("CPT",""), r.get("Location",""), r.get("PlaceOfService",""))

def bucketize(keyfn):
    b = defaultdict(lambda: {"total":0, "clean":0})
    for acct_key, meta in primary.items():
        bk = keyfn(acct_key, meta)
        b[bk]["total"] += 1
        if acct_key not in dirty: b[bk]["clean"] += 1
    return b

bgc  = bucketize(lambda k, m: (k[0], m[1]))                    # group, CPT
bglc = bucketize(lambda k, m: (k[0], m[2], m[1]))              # group, location, CPT
bgpc = bucketize(lambda k, m: (k[0], m[3], m[1]))              # group, POS, CPT

def qualifying(b, min_n):
    return [(k, v["total"], v["clean"], v["clean"]/v["total"])
            for k, v in b.items()
            if v["total"] >= min_n and v["clean"]/v["total"] >= THRESHOLD]

def dump(title, rows, cols):
    rows = sorted(rows, key=lambda r: (-r[3], -r[1]))
    cases = sum(r[1] for r in rows)
    clean = sum(r[2] for r in rows)
    print(f"\n=== {title} — {len(rows)} buckets, {cases} cases, {clean} historically clean ({clean/cases*100:.1f}%) ===")
    hdr = "  " + " | ".join(f"{c:<{w}}" for c, w in cols) + f" {'n':>4} {'clean':>5} {'rate':>6}"
    print(hdr)
    for key, tot, cln, rate in rows:
        label = "  " + " | ".join(f"{str(key[i])[:w]:<{w}}" for i, (_, w) in enumerate(cols))
        print(f"{label} {tot:>4} {cln:>5} {rate*100:>5.1f}%")

dump("Group + Location + CPT, n>=5, >=95% clean",
     qualifying(bglc, 5),
     [("Group", 10), ("Location", 42), ("CPT", 5)])

dump("Group + Location + CPT, n>=3, >=95% clean",
     qualifying(bglc, 3),
     [("Group", 10), ("Location", 42), ("CPT", 5)])

dump("Group + POS + CPT, n>=5, >=95% clean",
     qualifying(bgpc, 5),
     [("Group", 10), ("POS", 4), ("CPT", 5)])

dump("Group + CPT only, n>=10, >=95% clean",
     qualifying(bgc, 10),
     [("Group", 12), ("CPT", 5)])

# Write CSV of group+location+CPT >=95% at n>=3
out = os.path.join(os.path.dirname(CACHE), "safe_buckets.csv")
rows = sorted(qualifying(bglc, 3), key=lambda r: (-r[1]))
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["group", "location", "cpt", "total", "clean", "clean_rate"])
    for (g, loc, cpt), tot, cln, rate in rows:
        w.writerow([g, loc, cpt, tot, cln, f"{rate:.4f}"])

total_auto = sum(r[1] for r in rows)
clean_auto = sum(r[2] for r in rows)
print(f"\nWrote {out}")
print(f"Auto-advance pool (group+loc+CPT, n>=3, >=95%): {total_auto} cases, {clean_auto} historically clean")
print(f"Share of total volume: {total_auto}/{len(primary)} = {total_auto/len(primary)*100:.1f}%")
