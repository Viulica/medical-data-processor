#!/usr/bin/env python3
"""
CPT Accuracy Bulk Analysis
==========================
For every batch in the unified_results DB that has worktracker info:
  1. Downloads the predicted XLSX from Supabase
  2. Calls the Account Log Export API to get the change log
  3. Matches rows by (PatientLastName, PatientFirstName) — drops ambiguous cases
  4. An account with ZERO CPT changes in the log = our prediction was accurate
  5. An account WITH a "CPT changed from X to Y" = our prediction was wrong

Output: cpt_accuracy_analysis.xlsx (3 sheets)
  - Summary by Batch
  - All Cases
  - CPT Code Breakdown
"""

import os
import re
import sys
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from db_utils import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_BUCKET,
    get_all_unified_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://billing.anesthesiapartners.com/services/AccountLogExport.ashx"
API_KEY = "250e0955028c50480c4117acb4345d738795f81030cf3b9c597a7d9d17127df2"
DATE_START = "2024-01-01"
DATE_END = "2026-12-31"
MAX_WORKERS = 100


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def download_xlsx_from_supabase(supabase_path: str) -> Optional[pd.DataFrame]:
    """Download an XLSX from Supabase Storage and return as DataFrame."""
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{supabase_path}"
        r = httpx.get(
            url,
            headers={"Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=60,
            follow_redirects=True,
        )
        r.raise_for_status()
        return pd.read_excel(io.BytesIO(r.content), dtype=str)
    except Exception as e:
        logger.warning(f"Failed to download {supabase_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_api_log(worktracker_group: str, batch_number: str) -> list[dict]:
    """Call the Account Log Export API. Returns list of row dicts."""
    try:
        r = httpx.get(
            API_URL,
            params={
                "startDate": DATE_START,
                "endDate": DATE_END,
                "worktrackerGroup": worktracker_group,
                "worktrackerBatchNumber": batch_number,
            },
            headers={"X-Api-Key": API_KEY},
            verify=False,
            timeout=30,
        )
        r.raise_for_status()

        text = r.text.replace("\uFEFF", "").replace("\r", "")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) < 2:
            return []

        headers = lines[0].split("|")
        rows = []
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) == len(headers):
                rows.append(dict(zip(headers, parts)))
        return rows
    except Exception as e:
        logger.warning(f"API failed for {worktracker_group} #{batch_number}: {e}")
        return []


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).upper().strip())


# ---------------------------------------------------------------------------
# Core batch analysis
# ---------------------------------------------------------------------------

def analyze_batch(unified_result: dict, api_rows: list[dict]) -> Optional[dict]:
    """
    Download XLSX for one batch, match against API rows by patient name,
    and compute CPT accuracy.
    """
    wt_group = unified_result.get("worktracker_group", "")
    wt_batch = unified_result.get("worktracker_batch", "")
    supabase_path = unified_result.get("supabase_path", "")

    if not supabase_path:
        logger.warning(f"No supabase_path for {wt_group} #{wt_batch}, skipping")
        return None

    df = download_xlsx_from_supabase(supabase_path)
    if df is None or df.empty:
        return None

    # Find the CPT column (whichever comes first)
    cpt_col = next((c for c in ["ASA Code", "Procedure Code"] if c in df.columns), None)
    if not cpt_col:
        logger.warning(
            f"{wt_group} #{wt_batch}: no CPT column found. Columns: {list(df.columns)}"
        )
        return None

    # Find patient name columns
    first_col = next(
        (c for c in df.columns if "first" in c.lower() and "name" in c.lower()), None
    )
    last_col = next(
        (c for c in df.columns if "last" in c.lower() and "name" in c.lower()), None
    )
    if not first_col or not last_col:
        logger.warning(f"{wt_group} #{wt_batch}: no name columns found")
        return None

    # Count name occurrences in XLSX so we can drop ambiguous (duplicate) names
    xlsx_name_counts: dict[tuple, int] = {}
    for _, row in df.iterrows():
        key = (norm(row.get(last_col, "")), norm(row.get(first_col, "")))
        xlsx_name_counts[key] = xlsx_name_counts.get(key, 0) + 1

    # Index API rows by (last, first) — track count for ambiguity detection
    api_name_counts: dict[tuple, int] = {}
    api_by_name: dict[tuple, list] = {}
    for api_row in api_rows:
        key = (norm(api_row.get("PatientLastName", "")), norm(api_row.get("PatientFirstName", "")))
        api_name_counts[key] = api_name_counts.get(key, 0) + 1
        api_by_name.setdefault(key, []).append(api_row)

    cases = []
    total = accurate = changed = skipped_ambiguous = 0

    for _, row in df.iterrows():
        first = str(row.get(first_col, "")).strip()
        last = str(row.get(last_col, "")).strip()
        predicted_cpt = str(row.get(cpt_col, "")).strip()

        if not first or not last or first.lower() == "nan" or last.lower() == "nan":
            continue

        key = (norm(last), norm(first))

        # Drop if name appears more than once in our XLSX
        if xlsx_name_counts.get(key, 0) > 1:
            skipped_ambiguous += 1
            continue

        total += 1

        # All API log rows for this patient in this batch
        patient_api_rows = api_by_name.get(key, [])

        # Find CPT change rows
        cpt_change_rows = [
            r for r in patient_api_rows
            if re.search(r"CPT\s*:?\s*changed\s+from", r.get("Change", ""), re.IGNORECASE)
        ]

        if not cpt_change_rows:
            # No CPT changes → prediction was accepted
            accurate += 1
            final_cpt = predicted_cpt
            status = "ACCURATE"
        else:
            changed += 1
            # Final CPT = last "to XXXXX" in the change description
            last_change_text = cpt_change_rows[-1].get("Change", "")
            to_match = re.search(r"\bto\s+(\w+)", last_change_text, re.IGNORECASE)
            final_cpt = to_match.group(1) if to_match else ""
            status = "CHANGED"

        cases.append(
            {
                "worktracker_group": wt_group,
                "worktracker_batch": wt_batch,
                "patient_last": last,
                "patient_first": first,
                "predicted_cpt": predicted_cpt,
                "final_cpt": final_cpt,
                "status": status,
                "cpt_changes_count": len(cpt_change_rows),
            }
        )

    accuracy_pct = (accurate / total * 100) if total > 0 else 0.0
    logger.info(
        f"  {wt_group} #{wt_batch}: {accurate}/{total} accurate ({accuracy_pct:.1f}%), "
        f"{skipped_ambiguous} skipped"
    )

    return {
        "worktracker_group": wt_group,
        "worktracker_batch": wt_batch,
        "job_id": unified_result.get("job_id", ""),
        "total": total,
        "accurate": accurate,
        "changed": changed,
        "skipped_ambiguous": skipped_ambiguous,
        "accuracy_pct": accuracy_pct,
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Fetch all unified results that have worktracker info
    logger.info("Fetching all unified results from database...")
    all_results = []
    page = 1
    while True:
        page_data = get_all_unified_results(page=page, page_size=200)
        all_results.extend(page_data["results"])
        if len(page_data["results"]) < 200:
            break
        page += 1

    results_with_wt = [
        r for r in all_results
        if r.get("worktracker_group") and r.get("worktracker_batch") and r.get("supabase_path")
    ]
    logger.info(
        f"Found {len(results_with_wt)} batches with worktracker info + Supabase path "
        f"(out of {len(all_results)} total)"
    )

    if not results_with_wt:
        logger.error("No batches to analyze.")
        return

    # Process all batches in parallel
    batch_summaries = []
    all_cases = []

    def process_one(unified_result: dict):
        wt_group = unified_result["worktracker_group"]
        wt_batch = unified_result["worktracker_batch"]
        logger.info(f"Processing {wt_group} #{wt_batch}...")

        api_rows = fetch_api_log(wt_group, wt_batch)
        logger.info(f"  {wt_group} #{wt_batch}: {len(api_rows)} API log rows")

        return analyze_batch(unified_result, api_rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, r): r for r in results_with_wt}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    batch_summaries.append(
                        {
                            "Group": result["worktracker_group"],
                            "Batch": result["worktracker_batch"],
                            "Total Cases": result["total"],
                            "Accurate": result["accurate"],
                            "Changed": result["changed"],
                            "Skipped (Ambiguous Names)": result["skipped_ambiguous"],
                            "Accuracy %": round(result["accuracy_pct"], 1),
                        }
                    )
                    all_cases.extend(result["cases"])
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    if not batch_summaries:
        logger.warning("No analysis results produced.")
        return

    # Build DataFrames
    summary_df = pd.DataFrame(batch_summaries).sort_values(["Group", "Batch"])
    cases_df = pd.DataFrame(all_cases) if all_cases else pd.DataFrame()

    # CPT code breakdown (across all batches)
    if not cases_df.empty and "predicted_cpt" in cases_df.columns:
        cpt_breakdown = (
            cases_df.groupby("predicted_cpt")
            .agg(
                total_cases=("status", "count"),
                accurate_cases=("status", lambda x: (x == "ACCURATE").sum()),
            )
            .reset_index()
        )
        cpt_breakdown["accuracy_pct"] = (
            cpt_breakdown["accurate_cases"] / cpt_breakdown["total_cases"] * 100
        ).round(1)
        cpt_breakdown = cpt_breakdown.sort_values("total_cases", ascending=False)
    else:
        cpt_breakdown = pd.DataFrame()

    # Save to XLSX
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "cpt_accuracy_analysis.xlsx"
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary by Batch", index=False)
        if not cases_df.empty:
            cases_df.to_excel(writer, sheet_name="All Cases", index=False)
        if not cpt_breakdown.empty:
            cpt_breakdown.to_excel(writer, sheet_name="CPT Code Breakdown", index=False)

    # Print summary
    total_cases = summary_df["Total Cases"].sum()
    total_accurate = summary_df["Accurate"].sum()
    overall_pct = (total_accurate / total_cases * 100) if total_cases > 0 else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis complete  →  {output_path}")
    logger.info(
        f"Overall CPT accuracy: {total_accurate}/{total_cases} = {overall_pct:.1f}%"
    )
    logger.info(f"{'='*60}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
