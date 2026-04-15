"""Modifier change analysis."""
import glob, os, re
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

by_field = defaultdict(Counter)   # field_name -> Counter[(from, to)]
by_field_group = defaultdict(lambda: defaultdict(Counter))  # field -> group -> Counter
units_changes = Counter()  # numeric delta patterns

ft_re = re.compile(r"from\s+(\S+?)\s+to\s+(\S+)", re.IGNORECASE)
mod_field_re = re.compile(r"^(Modifier\d)", re.IGNORECASE)

for path in sorted(glob.glob(os.path.join(CACHE, "*__changes.txt"))):
    for r in read_pipe(path):
        ch = r.get("Change", "").strip()
        group = r.get("WorktrackerGroupName", "")
        m = mod_field_re.match(ch)
        if m:
            field = m.group(1).title()
            ft = ft_re.search(ch)
            if ft:
                by_field[field][(ft.group(1), ft.group(2))] += 1
                by_field_group[field][group][(ft.group(1), ft.group(2))] += 1
            else:
                # e.g., "Modifier1 added P2" or "Modifier1 deleted"
                head = re.sub(r"\s+", " ", ch[:40])
                by_field[field][(head,)] += 1
        elif ch.startswith("Modifier Units"):
            ft = ft_re.search(ch)
            if ft:
                units_changes[(ft.group(1), ft.group(2))] += 1

for fld in ("Modifier1", "Modifier2", "Modifier3", "Modifier4"):
    c = by_field[fld]
    print(f"\n=== {fld} — {sum(c.values())} total changes — top 20 ===")
    for k, v in c.most_common(20):
        arrow = f"{k[0]:>8} -> {k[1]}" if len(k) == 2 else k[0]
        print(f"  {v:>4}  {arrow}")

print(f"\n=== Modifier Units — {sum(units_changes.values())} total — top 20 ===")
for k, v in units_changes.most_common(20):
    print(f"  {v:>4}  {k[0]:>6} -> {k[1]}")

# Per-group top modifier pain
print("\n=== Groups with highest Modifier1 change volume ===")
grp_totals = [(g, sum(c.values())) for g, c in by_field_group["Modifier1"].items()]
for g, t in sorted(grp_totals, key=lambda x: -x[1])[:15]:
    print(f"  {t:>4}  {g}")

print("\n=== Groups with highest Modifier2 change volume ===")
grp_totals = [(g, sum(c.values())) for g, c in by_field_group["Modifier2"].items()]
for g, t in sorted(grp_totals, key=lambda x: -x[1])[:15]:
    print(f"  {t:>4}  {g}")
