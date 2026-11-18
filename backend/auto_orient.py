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
    """True when the worktracker group matches AUTO_ORIENT_GROUPS (default: PAG-AHS and BAP)."""
    wanted = [g.strip().upper() for g in os.environ.get("AUTO_ORIENT_GROUPS", "PAG-AHS,BAP").split(",") if g.strip()]
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


def _is_scan(page):
    """True when one image covers most of the page (a scanned sheet)."""
    area = abs(page.rect)
    try:
        return any(abs(r) > 0.6 * area for img in page.get_images(full=True) for r in page.get_image_rects(img[0]))
    except Exception:
        return False


def _text_layer_rotation(page):
    """Clockwise degrees to make a scanned page upright, judged by its OCR text layer: 0 when most text runs
    left-to-right, 180 when it runs right-to-left; None when the layer is too thin or runs sideways."""
    votes = {}
    for b in page.get_text("dict").get("blocks", []):
        for line in b.get("lines", []):
            d = tuple(round(x) for x in line.get("dir", (1, 0)))
            votes[d] = votes.get(d, 0) + sum(len(s.get("text", "").strip()) for s in line.get("spans", []))
    total = sum(votes.values())
    if total < 200:
        return None
    if votes.get((1, 0), 0) >= 0.8 * total:
        return 0
    if votes.get((-1, 0), 0) >= 0.8 * total:
        return 180
    return None


def _orient_pdf(path, work):
    """Rotate the pages of one PDF in place; returns the number of pages rotated."""
    doc = fitz.open(path)
    jobs = []
    known = []
    for i, page in enumerate(doc):
        if len(page.get_text().strip()) > 50:
            # A page with a text layer: typed/EMR pages are upright; a scan whose scanner OCR'd it (BAP) tells its
            # orientation by the direction its text runs (right-to-left = upside down), which beats Tesseract.
            if not _is_scan(page):
                continue
            rot = _text_layer_rotation(page)
            if rot is not None:
                known.append((i, rot))
                continue
        png = os.path.join(work, f"{os.path.basename(path)}_{i}.png")
        page.get_pixmap(dpi=200, colorspace=fitz.csGRAY).save(png)  # 100 dpi is too coarse for handwritten forms
        jobs.append((i, png))
    with ThreadPoolExecutor(8) as ex:
        rots = known + list(ex.map(lambda j: (j[0], _detect(j[1])), jobs))
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
