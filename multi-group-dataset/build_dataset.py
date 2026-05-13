"""Build a multi-group dataset (KAP-ASC, KAP-CYP, APO-UTP).

Walk backwards from the highest populated batch per group, accumulating accounts
across all groups until total accounts >= TARGET_ACCOUNTS, then download every
referenced PDF from SharePoint and write a master CSV linking local PDF paths
to CPT/ICD codes.
"""
import csv
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

# (group, starting batch number — highest known populated)
GROUPS = [
    ("KAP-ASC", 254),
    ("KAP-CYP", 262),
    ("APO-UTP", 885),
    ("GOS-GOH", 412),
    ("GOS-GOSC", 239),
]
TARGET_ACCOUNTS = 10000  # now: target number of anesthesia (CPT starts with "0") rows
DATE_START = "2024-01-01"
DATE_END = "2026-12-31"
# stop walking back if we hit this many consecutive empty batches per group
EMPTY_STREAK_STOP = 15

ROOT = Path(__file__).parent
PDF_ROOT = ROOT / "pdfs"
CSV_PATH = ROOT / "anesthesia_main_dataset.csv"

COLUMNS = [
    "WorktrackerGroupName", "WorktrackerBatchNo", "DataIntegrationBatchNo", "Client",
    "AccountNumber", "PatientLastName", "PatientFirstName", "EhrPath", "DOS", "CPT",
    "Modifier1", "Modifier2", "Modifier3", "Modifier4", "ICD1", "ICD2", "ICD3", "ICD4",
    "Location", "PlaceOfService", "MD", "CRNA", "ReferringProvider", "Rendering",
    "ChargeAmount", "ChargeStatus", "TimeUnits", "BaseUnits", "Minutes",
]


def get_sharepoint_token() -> str:
    r = requests.post(
        f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def fetch_batch(group: str, batch_no: int) -> list[dict]:
    params = {
        "startDate": DATE_START,
        "endDate": DATE_END,
        "worktrackerGroup": group,
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


def is_anesthesia_main_line(row: dict) -> bool:
    """Main anesthesia charge: CPT starts with '0' (e.g. 00100-01999, 0xxxx)."""
    cpt = (row.get("CPT") or "").strip()
    return cpt.startswith("0")


def collect_groups_until_target() -> list[dict]:
    """Walk batches backwards across groups in round-robin. Keep only the main
    anesthesia charge line per account (CPT starts with '0'). Stop when total
    filtered rows >= TARGET_ACCOUNTS."""
    state = {g: {"next_batch": start, "empty_streak": 0, "done": False} for g, start in GROUPS}
    kept_rows = []
    seen_keys = set()  # (group, account) — keep only ONE main line per case

    while not all(s["done"] for s in state.values()):
        if len(kept_rows) >= TARGET_ACCOUNTS:
            break
        for group, _ in GROUPS:
            if state[group]["done"]:
                continue
            b = state[group]["next_batch"]
            if b < 1:
                state[group]["done"] = True
                continue
            try:
                rows = fetch_batch(group, b)
            except Exception as e:
                print(f"  {group} #{b}: ERROR {e}; retrying once")
                try:
                    rows = fetch_batch(group, b)
                except Exception as e2:
                    print(f"  {group} #{b}: FAILED {e2}; skipping")
                    rows = []
            n = len(rows)
            new_kept = 0
            if n == 0:
                state[group]["empty_streak"] += 1
                if state[group]["empty_streak"] >= EMPTY_STREAK_STOP:
                    state[group]["done"] = True
                    print(f"  {group}: stopping after {EMPTY_STREAK_STOP} empty batches in a row")
            else:
                state[group]["empty_streak"] = 0
                # group rows by account, then pick the highest-amount anesthesia line
                by_acct: dict[str, list[dict]] = {}
                for r in rows:
                    acct = r.get("AccountNumber", "")
                    by_acct.setdefault(acct, []).append(r)
                for acct, lines in by_acct.items():
                    key = (group, acct)
                    if key in seen_keys:
                        continue
                    anes = [l for l in lines if is_anesthesia_main_line(l)]
                    if not anes:
                        continue
                    # pick the line with the largest charge amount (the "main" one)
                    def amt(l):
                        try:
                            return float(l.get("ChargeAmount") or 0)
                        except Exception:
                            return 0.0
                    main = max(anes, key=amt)
                    main["_group"] = group
                    kept_rows.append(main)
                    seen_keys.add(key)
                    new_kept += 1
            print(f"  {group} #{b}: {n} api rows, +{new_kept} kept (cum kept={len(kept_rows)})")
            state[group]["next_batch"] = b - 1
            if len(kept_rows) >= TARGET_ACCOUNTS:
                break
    print(f"Collected {len(kept_rows)} main anesthesia rows")
    return kept_rows


def ehr_path_to_drive_path(ehr: str) -> str | None:
    marker = "SHARED%20DOCUMENTS/"
    idx = ehr.upper().find(marker)
    if idx == -1:
        return None
    return urllib.parse.unquote(ehr[idx + len(marker):])


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


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


def main():
    print(f"Walking backwards across {[g for g,_ in GROUPS]} until {TARGET_ACCOUNTS} accounts...")
    all_rows = collect_groups_until_target()

    print("Getting SharePoint token...")
    token = get_sharepoint_token()

    pdf_tasks = {}
    for row in all_rows:
        ehr = row.get("EhrPath", "")
        if not ehr:
            continue
        dp = ehr_path_to_drive_path(ehr)
        if not dp:
            continue
        group = row.get("_group") or row.get("WorktrackerGroupName") or "UNK"
        batch = row.get("WorktrackerBatchNo") or "0"
        fname = safe_filename(dp.rsplit("/", 1)[-1])
        local = PDF_ROOT / group / f"batch_{batch}" / fname
        pdf_tasks[dp] = local
        row["LocalPdfPath"] = str(local.relative_to(ROOT))

    print(f"Unique PDFs: {len(pdf_tasks)}")
    ok = fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=50) as ex:
        futs = {ex.submit(download_pdf, token, dp, lp): (dp, lp) for dp, lp in pdf_tasks.items()}
        for i, f in enumerate(as_completed(futs), 1):
            dp, _ = futs[f]
            success, msg = f.result()
            if success:
                ok += 1
            else:
                fail += 1
                failures.append((dp, msg))
            if i % 100 == 0:
                print(f"  {i}/{len(pdf_tasks)} ok={ok} fail={fail}")
    print(f"Downloads: ok={ok} fail={fail}")
    for dp, msg in failures[:10]:
        print(f"  fail {msg}: {dp}")

    out_cols = ["Group"] + COLUMNS + ["LocalPdfPath"]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            r["Group"] = r.get("_group") or r.get("WorktrackerGroupName") or ""
            w.writerow(r)
    print(f"Wrote {len(all_rows)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
