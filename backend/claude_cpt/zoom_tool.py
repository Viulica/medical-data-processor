"""zoom.py PAGE X0 Y0 X1 Y1 [ROT]: crop a region of chart.pdf (fractions 0-1 of the page, origin top-left) at high
resolution, optionally rotated clockwise by ROT degrees (90, 180 or 270) for sideways scans, and save it as zoom_N.png. Copied into each zoomed extraction call's working directory by server.py."""
import glob
import sys

import fitz

try:
    page, x0, y0, x1, y1 = int(sys.argv[1]), *map(float, sys.argv[2:6])
    rot = int(sys.argv[6]) if len(sys.argv) > 6 else 0
except (IndexError, ValueError):
    sys.exit("usage: python3 zoom.py PAGE X0 Y0 X1 Y1 [ROT]   (e.g. python3 zoom.py 1 0.5 0 1 0.25 90)")
if rot not in (0, 90, 180, 270):
    sys.exit("ROT must be 0, 90, 180 or 270")
doc = fitz.open("chart.pdf")
if not 1 <= page <= len(doc):
    sys.exit(f"page must be 1-{len(doc)}")
x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
if x1 - x0 < 0.01 or y1 - y0 < 0.01:
    sys.exit("region too small: use a box at least 0.01 wide and tall")
r = doc[page - 1].rect
clip = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
dpi = max(150, min(600, int(1400 / (clip.width / 72))))  # about 1400 px wide, capped at 600 dpi
out = f"zoom_{len(glob.glob('zoom_*.png')) + 1}.png"
doc[page - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72).prerotate(rot), clip=clip).save(out)
print(f"saved {out} ({dpi} dpi{f', rotated {rot}' if rot else ''}) - Read it")
