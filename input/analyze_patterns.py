"""Pattern analysis of CPT / ICD1 changes across all cached batches."""
import glob
import os
import re
from collections import Counter, defaultdict

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_cache")

def read_pipe(path):
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8-sig", errors="replace").replace("\r", "")
    lines = [ln for ln in raw.split("\n") if ln]
    if not lines:
        return []
    header = lines[0].split("|")
    return [dict(zip(header, ln.split("|"))) for ln in lines[1:]]


cpt_changes = Counter()       # (from_code, to_code)
icd_changes = Counter()       # (from_code, to_code)
cpt_per_group = defaultdict(Counter)
icd_per_group = defaultdict(Counter)
all_change_types = Counter()

from_to_re = re.compile(r"from\s+(\S+?)\s+to\s+(\S+)", re.IGNORECASE)

for path in sorted(glob.glob(os.path.join(CACHE, "*__changes.txt"))):
    for r in read_pipe(path):
        change = r.get("Change", "").strip()
        if not change:
            continue
        # Classify change type by first word-group
        head = re.split(r"\s+(changed|added|deleted)", change, 1)[0].strip()
        all_change_types[head] += 1

        group = r.get("WorktrackerGroupName", "")
        m = from_to_re.search(change)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        low = change.lower()
        if low.startswith("cpt"):
            cpt_changes[(src, dst)] += 1
            cpt_per_group[group][(src, dst)] += 1
        elif low.startswith("icd1"):
            icd_changes[(src, dst)] += 1
            icd_per_group[group][(src, dst)] += 1

def show(title, counter, n=25):
    print(f"\n{title} (top {n})")
    total = sum(counter.values())
    print(f"  total: {total}")
    for (src, dst), c in counter.most_common(n):
        print(f"  {c:>4}  {src:>8}  ->  {dst}")

show("Top CPT transitions", cpt_changes, 25)
show("Top ICD1 transitions", icd_changes, 30)

print("\nAll change-type frequencies (top 25)")
for head, c in all_change_types.most_common(25):
    print(f"  {c:>5}  {head}")

# Into-what buckets
print("\nCPT: what are codes being corrected TO? (top 15)")
dst_counter = Counter()
for (s, d), c in cpt_changes.items():
    dst_counter[d] += c
for d, c in dst_counter.most_common(15):
    print(f"  {c:>4}  {d}")

print("\nCPT: what FROM codes are most frequently wrong? (top 15)")
src_counter = Counter()
for (s, d), c in cpt_changes.items():
    src_counter[s] += c
for s, c in src_counter.most_common(15):
    print(f"  {c:>4}  {s}")

print("\nICD1: what are codes being corrected TO? (top 20)")
dst = Counter()
for (s, d), c in icd_changes.items():
    dst[d] += c
for d, c in dst.most_common(20):
    print(f"  {c:>4}  {d}")

print("\nICD1: what FROM codes are most frequently wrong? (top 20)")
src = Counter()
for (s, d), c in icd_changes.items():
    src[s] += c
for s, c in src.most_common(20):
    print(f"  {c:>4}  {s}")

# Colonoscopy family
print("\nColonoscopy swaps (00811/00812/00813)")
colon = {"00811", "00812", "00813"}
for (s, d), c in sorted(cpt_changes.items(), key=lambda x: -x[1]):
    if s in colon or d in colon:
        print(f"  {c:>4}  {s}  ->  {d}")
