"""Straighten sideways / upside-down scanned pages in a unified job's input ZIP before any model sees them.

Tesseract's orientation detection (--psm 0) runs on every page without a text layer; a page it confidently
reads as rotated gets its PDF /Rotate set so it renders upright for extraction, CPT and ICD alike.
Enabled per worktracker group via AUTO_ORIENT_GROUPS (comma-separated substrings, "*" = every group).
"""
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import fitz

MIN_CONFIDENCE = float(os.environ.get("AUTO_ORIENT_MIN_CONFIDENCE", "2.0"))


def enabled_for(group):
    """True when the worktracker group matches AUTO_ORIENT_GROUPS (default: PAG-AHS only)."""
    wanted = [g.strip().upper() for g in os.environ.get("AUTO_ORIENT_GROUPS", "PAG-AHS").split(",") if g.strip()]
    g = (group or "").upper()
    return "*" in wanted or any(w in g for w in wanted)


def _detect(png_path):
    """Clockwise degrees needed to make the page upright (0 when unsure)."""
    try:
        out = subprocess.run(["tesseract", png_path, "-", "--psm", "0"], capture_output=True, text=True, timeout=30).stdout
    except (subprocess.TimeoutExpired, OSError):
        return 0
    rot = re.search(r"Rotate:\s*(\d+)", out)
    conf = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    if not rot or not conf or float(conf.group(1)) < MIN_CONFIDENCE:
        return 0
    return int(rot.group(1)) % 360


def _orient_pdf(path, work):
    """Rotate the pages of one PDF in place; returns the number of pages rotated."""
    doc = fitz.open(path)
    jobs = []
    for i, page in enumerate(doc):
        if len(page.get_text().strip()) > 50:  # typed/EMR page: already upright
            continue
        png = os.path.join(work, f"{os.path.basename(path)}_{i}.png")
        page.get_pixmap(dpi=200, colorspace=fitz.csGRAY).save(png)  # 100 dpi is too coarse for handwritten forms
        jobs.append((i, png))
    with ThreadPoolExecutor(8) as ex:
        rots = list(ex.map(lambda j: (j[0], _detect(j[1])), jobs))
    n = 0
    for i, rot in rots:
        if rot:
            doc[i].set_rotation((doc[i].rotation + rot) % 360)
            n += 1
    if n:
        doc.save(path + ".tmp")
        doc.close()
        os.replace(path + ".tmp", path)
    else:
        doc.close()
    return n


def orient_zip(zip_path):
    """Rewrite the input ZIP with upright pages. Returns (pdfs checked, pages rotated)."""
    work = tempfile.mkdtemp(prefix="orient_")
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            z.extractall(work)
        pdfs = [n for n in names if n.lower().endswith(".pdf")]
        rotated = sum(_orient_pdf(os.path.join(work, n), work) for n in pdfs)
        if rotated:
            tmp = zip_path + ".tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                for n in names:
                    p = os.path.join(work, n)
                    if os.path.isfile(p):
                        z.write(p, n)
            os.replace(tmp, zip_path)
        return len(pdfs), rotated
    finally:
        shutil.rmtree(work, ignore_errors=True)
