#!/usr/bin/env python3
"""Print every page's text layer of a chart PDF; flags pages that need to be viewed as images."""
import sys

import fitz

d = fitz.open(sys.argv[1])
for i, p in enumerate(d, 1):
    t = p.get_text().strip()
    print(f"===== PAGE {i}/{d.page_count} =====")
    print(t if len(t) > 80 else "[NO TEXT LAYER - view this page as an image with the Read tool]")
