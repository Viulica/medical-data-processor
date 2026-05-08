"""Build PAC-MHI dataset: fetch batches 752+ via allCharges API, download PDFs from SharePoint,
write a master CSV linking local PDF paths to CPT/ICD codes."""
import csv
import os
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "250e0955028c50480c4117acb4345d738795f81030cf3b9c597a7d9d17127df2"
API_URL = "https://billing.anesthesiapartners.com/services/AccountLogExport.ashx"

TENANT_ID = "0138a897-ff88-4dc2-933b-fad359609873"
CLIENT_ID = "a54eaddf-654d-4f0e-9071-2b8c8ad26942"
CLIENT_SECRET = "MlO8Q~6ARQrwWgbvb6v9qWOuHCWIU3m6MaKsTczK"
DRIVE_ID = "b!ngvjU5cJZkK_4oF06KsHXBNacfJKjWNPmBwGVZ1Z4EPXuNIeAqOqSoPA_SSOtvFY"

GROUP = "PAC-MHI"
START_BATCH = 752
END_BATCH = 793
DATE_START = "2025-01-01"
DATE_END = "2026-12-31"

ROOT = Path(__file__).parent
PDF_ROOT = ROOT / "pdfs"
CSV_PATH = ROOT / "pac_mhi_dataset.csv"

COLUMNS = [
    "WorktrackerGroupName", "WorktrackerBatchNo", "DataIntegrationBatchNo", "Client",
    "AccountNumber", "PatientLastName", "PatientFirstName", "EhrPath", "DOS", "CPT",
    "Modifier1", "Modifier2", "Modifier3", "Modifier4", "ICD1", "ICD2", "ICD3", "ICD4",
    "Location", "PlaceOfService", "MD", "CRNA", "ReferringProvider", "Rendering",
    "ChargeAmount", "ChargeStatus", "TimeUnits", "BaseUnits", "Minutes",
]


def get_sharepoint_token() -> str:
    resp = requests.post(
        f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_batch(batch_no: int) -> list[dict]:
    params = {
        "startDate": DATE_START,
        "endDate": DATE_END,
        "worktrackerGroup": GROUP,
        "worktrackerBatchNumber": batch_no,
        "allCharges": "true",
    }
    r = requests.get(API_URL, params=params, headers={"X-Api-Key": API_KEY}, verify=False, timeout=60)
    r.raise_for_status()
    text = r.text.lstrip("﻿").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return []
    header = lines[0].split("|")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("|")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return rows


def ehr_path_to_drive_path(ehr: str) -> str | None:
    marker = "SHARED%20DOCUMENTS/"
    idx = ehr.upper().find(marker)
    if idx == -1:
        return None
    encoded = ehr[idx + len(marker):]
    return urllib.parse.unquote(encoded)


def download_pdf(token: str, drive_path: str, out_path: Path) -> tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = urllib.parse.quote(drive_path)
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{encoded}:/content"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, allow_redirects=True, timeout=120)
        if r.status_code != 200:
            return False, f"http {r.status_code}"
        out_path.write_bytes(r.content)
        return True, "ok"
    except Exception as e:
        return False, f"err {e}"


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def main():
    print(f"Fetching batches {START_BATCH}-{END_BATCH} for {GROUP}...")
    all_rows = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(fetch_batch, b): b for b in range(START_BATCH, END_BATCH + 1)}
        for f in as_completed(futs):
            b = futs[f]
            rows = f.result()
            print(f"  batch {b}: {len(rows)} rows")
            all_rows.extend(rows)
    print(f"Total rows: {len(all_rows)}")

    print("Getting SharePoint token...")
    token = get_sharepoint_token()

    # Build download tasks: one PDF per unique EhrPath
    pdf_tasks = {}  # drive_path -> local_path
    for row in all_rows:
        ehr = row.get("EhrPath", "")
        if not ehr:
            continue
        drive_path = ehr_path_to_drive_path(ehr)
        if not drive_path:
            continue
        batch = row["WorktrackerBatchNo"]
        fname = safe_filename(drive_path.rsplit("/", 1)[-1])
        local = PDF_ROOT / f"batch_{batch}" / fname
        pdf_tasks[drive_path] = local
        row["LocalPdfPath"] = str(local.relative_to(ROOT))

    print(f"Unique PDFs to download: {len(pdf_tasks)}")
    ok = 0
    fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=50) as ex:
        futs = {ex.submit(download_pdf, token, dp, lp): (dp, lp) for dp, lp in pdf_tasks.items()}
        for i, f in enumerate(as_completed(futs), 1):
            dp, lp = futs[f]
            success, msg = f.result()
            if success:
                ok += 1
            else:
                fail += 1
                failures.append((dp, msg))
            if i % 50 == 0:
                print(f"  downloaded {i}/{len(pdf_tasks)} (ok={ok} fail={fail})")
    print(f"Downloads complete: ok={ok} fail={fail}")
    if failures[:10]:
        print("Sample failures:")
        for dp, msg in failures[:10]:
            print(f"  {msg}: {dp}")

    print(f"Writing CSV: {CSV_PATH}")
    out_cols = COLUMNS + ["LocalPdfPath"]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"Done. {len(all_rows)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
