"""Fetch Account Log Export API for every CHARGE batch; cache to disk.

- Default mode (change log) and allCharges=true (final charges).
- 1s delay between HTTP requests.
- Skips already-cached files so the script is resumable.
"""
import csv
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request

API_KEY = "250e0955028c50480c4117acb4345d738795f81030cf3b9c597a7d9d17127df2"
BASE_URL = "https://billing.anesthesiapartners.com/services/AccountLogExport.ashx"
START_DATE = "2026-01-01"
END_DATE = "2026-04-30"

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "charge_batches.csv")
CACHE_DIR = os.path.join(HERE, "api_cache")
LOG_PATH = os.path.join(HERE, "api_cache.log")

os.makedirs(CACHE_DIR, exist_ok=True)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def parse_batch(name: str):
    """Return (group, batch_number) or None."""
    hash_match = re.search(r"#\s*(\d+)", name)
    if not hash_match:
        return None
    batch_number = hash_match.group(1)

    head = name[: hash_match.start()]
    # Strip any date patterns: e.g. "03-29_03-30-26", "04-01-2026", "03-25-03-27-2026"
    head = re.sub(r"\d{1,2}-\d{1,2}(?:[-_]\d{1,2}-\d{1,2})?-\d{2,4}", "", head)
    # Remove extra separators/spaces
    group = re.sub(r"[\s\-_]+$", "", head.strip())
    # Collapse " - " style (e.g. "CHA - HDH - ")
    group = re.sub(r"\s*-\s*", "-", group)
    group = re.sub(r"\s+", " ", group).strip()
    # Trim trailing hyphens/spaces again
    group = group.strip(" -_")
    if not group:
        return None
    return group, batch_number


def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def fetch(group: str, batch: str, all_charges: bool) -> bytes:
    params = {
        "startDate": START_DATE,
        "endDate": END_DATE,
        "worktrackerGroup": group,
        "worktrackerBatchNumber": batch,
    }
    if all_charges:
        params["allCharges"] = "true"
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"X-Api-Key": API_KEY})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as resp:
        return resp.read()


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


def main():
    with open(CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))

    parsed = []
    unparsed = []
    for r in rows:
        p = parse_batch(r["batch_name"])
        if p:
            parsed.append((r["batch_name"], p[0], p[1]))
        else:
            unparsed.append(r["batch_name"])

    log(f"Parsed {len(parsed)} / {len(rows)} batches. Unparsed: {len(unparsed)}")
    for u in unparsed:
        log(f"  UNPARSED: {u}")

    total = len(parsed) * 2
    done = 0
    fetched = 0
    errors = 0

    for original_name, group, batch in parsed:
        for all_charges in (False, True):
            done += 1
            suffix = "all" if all_charges else "changes"
            fname = f"{safe_slug(group)}__{batch}__{suffix}.txt"
            path = os.path.join(CACHE_DIR, fname)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            try:
                data = fetch(group, batch, all_charges)
                with open(path, "wb") as out:
                    out.write(data)
                fetched += 1
                log(f"[{done}/{total}] OK  {group} #{batch} {suffix} ({len(data)} bytes)")
            except Exception as e:
                errors += 1
                err_path = path + ".error"
                with open(err_path, "w") as out:
                    out.write(f"{type(e).__name__}: {e}\n")
                log(f"[{done}/{total}] ERR {group} #{batch} {suffix}: {e}")
            time.sleep(1)

    log(f"Done. fetched={fetched} errors={errors} cached_skipped={total - fetched - errors}")


if __name__ == "__main__":
    main()
