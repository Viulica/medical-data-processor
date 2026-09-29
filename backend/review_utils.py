"""
Sorter review stage for unified (Extract + CPT + ICD) results.

After a unified job finishes, sorters get one screen per batch where they can fix the
handful of fields they correct by hand today in Excel — Responsible Provider, MD, CRNA,
An Start, An Stop, Points — and only then download the CSV for the billing app.

Two provider modes exist and the review payload tells the UI which one it is in:

  * 'annotation'  – the template has "Extract providers from PDF annotations" ON.
                    The roster is the template's provider_mapping column, which carries
                    MedNet codes, so the UI can offer a "type the MedNet code" shortcut
                    (same "7/1" / "1/SRNA" grammar the red-number pipeline understands).
                    Every row also carries a 'Provider Source' cell that says whether
                    the value came from a matched annotation or is an AI fallback.
  * 'ai'          – no annotation extraction; providers are AI-read names, corrected
                    against the roster harvested from the Responsible Provider field.
                    There are no MedNet codes, so the UI offers names only.

Rows are addressed by their 0-based position in the result file. Edits are applied to
the DataFrame, written back to the local CSV/XLSX (when the job is still in memory) and
re-uploaded to Supabase so the Results tab always serves the reviewed file.
"""

import io
import json
import logging
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Fields the sorter may change on the review screen (order = column order in the UI).
EDITABLE_FIELDS = ['Responsible Provider', 'MD', 'CRNA', 'An Start', 'An Stop', 'Points']

# Read-only context shown next to the editable cells when present.
CONTEXT_FIELDS = [
    'Patient Last Name', 'Patient First Name', 'Patient Middle Name', 'Patient DOB',
    'Case #', 'Surgeon', 'ASA Code', 'Anesthesia Type', 'Concurrent Providers',
    'Provider Source', 'StaffVerify', 'CoderVerify', 'SRNA', 'source_file',
]

REVIEW_CACHE_DIR = Path('/tmp/review_cache')


def _local_result_paths(job_id: str) -> Tuple[str, str]:
    base = f"/tmp/results/{job_id}_unified_result"
    return f"{base}.csv", f"{base}.xlsx"


# ---------------------------------------------------------------------------
# Loading / saving the result table
# ---------------------------------------------------------------------------

def load_result_df(job_id: str, job=None, db_record: Optional[dict] = None) -> Tuple[pd.DataFrame, str]:
    """Return (df, source) where source is 'local' or 'supabase'.

    Everything is read as text so times like '9/30/2025 9:22:00 AM' and IDs with
    leading zeros survive the round-trip untouched.
    """
    csv_path = getattr(job, 'result_file', None) if job is not None else None
    if not csv_path or not os.path.exists(csv_path):
        csv_path = _local_result_paths(job_id)[0]
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        return df, 'local'

    if db_record and db_record.get('supabase_path'):
        from db_utils import download_from_supabase
        data = download_from_supabase(db_record['supabase_path'])
        df = pd.read_excel(io.BytesIO(data), dtype=str, engine='openpyxl')
        df = df.fillna('')
        return df, 'supabase'

    raise FileNotFoundError(f"No result file available for job {job_id}")


def save_result_df(job_id: str, df: pd.DataFrame, job=None, db_record: Optional[dict] = None) -> Tuple[Optional[str], int]:
    """Write the reviewed table to the local CSV/XLSX and re-upload the XLSX to Supabase.

    Returns (supabase_path, file_size_bytes). Local files are always (re)written so the
    export endpoint can serve CSV without another Supabase round-trip.
    """
    csv_path, xlsx_path = _local_result_paths(job_id)
    if job is not None:
        if getattr(job, 'result_file', None):
            csv_path = job.result_file
        if getattr(job, 'result_file_xlsx', None):
            xlsx_path = job.result_file_xlsx
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
    except Exception as e:
        logger.warning(f"[Review {job_id}] Failed to write XLSX: {e}")
    if job is not None:
        job.result_file = csv_path
        job.result_file_xlsx = xlsx_path

    supabase_path = (db_record or {}).get('supabase_path') or f"{job_id}_unified_result.xlsx"
    try:
        from db_utils import upload_to_supabase
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        upload_to_supabase(
            supabase_path, buf.getvalue(),
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
    except Exception as e:
        logger.warning(f"[Review {job_id}] Failed to re-upload XLSX to Supabase: {e}")
        supabase_path = None

    size = os.path.getsize(xlsx_path) if os.path.exists(xlsx_path) else os.path.getsize(csv_path)
    return supabase_path, size


# ---------------------------------------------------------------------------
# Provider roster
# ---------------------------------------------------------------------------

def _title_from_name(name: str) -> str:
    m = re.search(r',\s*([A-Z]{2,5})\s*$', name.strip().upper())
    return m.group(1) if m else ''


def build_roster(template: Optional[dict]) -> Tuple[List[dict], bool, str]:
    """Return (roster, has_mednet, provider_mode) for a template.

    roster entries: {'code': '7' | None, 'name': 'SUM, DAVID, MD', 'title': 'MD'|'CRNA'|...}
    """
    if not template:
        return [], False, 'ai'

    annotations_on = bool(template.get('extract_providers_from_annotations'))
    fields = (template.get('template_data') or {}).get('fields') or []

    current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'current')
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from field_definitions import derive_provider_mapping_for_template

    text, has_mednet = derive_provider_mapping_for_template(
        template.get('provider_mapping'), fields, annotations_on
    )
    roster: List[dict] = []
    if text and has_mednet:
        from provider_annotation_utils import parse_provider_mapping
        parsed = parse_provider_mapping(text)  # code -> {'name','title'}
        for code, info in parsed.items():
            roster.append({'code': code, 'name': info['name'], 'title': info.get('title') or _title_from_name(info['name'])})
        roster.sort(key=lambda r: (r['title'], r['name']))
    elif text:
        for line in text.splitlines():
            name = line.strip()
            if name:
                roster.append({'code': None, 'name': name, 'title': _title_from_name(name)})

    return roster, bool(has_mednet), ('annotation' if annotations_on else 'ai')


# ---------------------------------------------------------------------------
# Review payload
# ---------------------------------------------------------------------------

def _row_needs_review(row: dict, provider_mode: str) -> List[str]:
    reasons = []
    src = str(row.get('Provider Source', '')).strip().lower()
    if provider_mode == 'annotation' and src and not src.startswith('annotation'):
        reasons.append('provider-ai-fallback')
    for col in ('StaffVerify', 'CoderVerify'):
        if str(row.get(col, '')).strip():
            reasons.append(col.lower())
    if not str(row.get('Responsible Provider', '')).strip():
        reasons.append('no-provider')
    if not str(row.get('An Start', '')).strip() or not str(row.get('An Stop', '')).strip():
        reasons.append('missing-time')
    return reasons


def build_review_payload(job_id: str, df: pd.DataFrame, template: Optional[dict],
                         db_record: Optional[dict], file_source: str) -> dict:
    roster, has_mednet, provider_mode = build_roster(template)
    if db_record and db_record.get('provider_mode'):
        provider_mode = db_record['provider_mode']
    roster_source = 'template' if roster else 'none'
    if not roster:
        # Template carries no provider list (e.g. RIV-CHARGE's reference list is empty):
        # offer the names the extraction already produced so the sorter still gets a
        # pick-list instead of retyping names.
        seen = set()
        for col in ('Responsible Provider', 'MD', 'CRNA'):
            if col not in df.columns:
                continue
            for v in df[col].astype(str):
                v = v.strip()
                if v and v.upper() not in seen:
                    seen.add(v.upper())
                    roster.append({'code': None, 'name': v, 'title': _title_from_name(v)})
        roster.sort(key=lambda r: (r['title'], r['name']))
        roster_source = 'results' if roster else 'none'

    columns = list(df.columns)
    editable = [f for f in EDITABLE_FIELDS if f in columns]
    context = [f for f in CONTEXT_FIELDS if f in columns]

    roster_names = {r['name'].strip().upper() for r in roster}
    rows = []
    for i, rec in enumerate(df.to_dict(orient='records')):
        row = {'_row': i}
        row.update(rec)
        row['_needs_review'] = _row_needs_review(rec, provider_mode)
        row['_off_roster'] = [
            f for f in ('Responsible Provider', 'MD', 'CRNA')
            if f in rec and str(rec[f]).strip() and roster_names and str(rec[f]).strip().upper() not in roster_names
        ]
        rows.append(row)

    source_file_map = (db_record or {}).get('source_file_map') or {}
    if isinstance(source_file_map, str):
        try:
            source_file_map = json.loads(source_file_map)
        except Exception:
            source_file_map = {}

    return {
        'job_id': job_id,
        'group': (db_record or {}).get('worktracker_group'),
        'batch': (db_record or {}).get('worktracker_batch'),
        'template_id': (template or {}).get('id'),
        'template_name': (template or {}).get('name'),
        'provider_mode': provider_mode,
        'roster': roster,
        'roster_has_mednet': has_mednet,
        'roster_source': roster_source,
        'editable_fields': editable,
        'context_fields': context,
        'columns': columns,
        'rows': rows,
        'row_count': len(rows),
        'file_source': file_source,
        'has_input_zip': bool((db_record or {}).get('input_zip_supabase_path')),
        'source_file_map': source_file_map,
        'reviewed_at': (db_record or {}).get('reviewed_at').isoformat() if (db_record or {}).get('reviewed_at') else None,
        'review_edit_count': len((db_record or {}).get('review_edits') or []),
    }


# ---------------------------------------------------------------------------
# Applying edits
# ---------------------------------------------------------------------------

def apply_edits(df: pd.DataFrame, edits: List[dict]) -> Tuple[pd.DataFrame, List[dict]]:
    """Apply [{row, field, value}] to df in place. Returns (df, change_log).

    Only EDITABLE_FIELDS (plus SRNA, which the MedNet shortcut may set) are accepted;
    anything else is ignored rather than silently written into billing data.
    """
    allowed = set(EDITABLE_FIELDS) | {'SRNA'}
    log = []
    now = datetime.now(timezone.utc).isoformat()
    for e in edits:
        field = e.get('field')
        if field not in allowed:
            continue
        try:
            idx = int(e.get('row'))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(df):
            continue
        if field not in df.columns:
            df[field] = ''
        new = '' if e.get('value') is None else str(e.get('value')).strip()
        old = str(df.at[idx, field]) if pd.notna(df.at[idx, field]) else ''
        if new == old:
            continue
        df.at[idx, field] = new
        log.append({'row': idx, 'field': field, 'old': old, 'new': new, 'at': now,
                    'source_file': str(df.at[idx, 'source_file']) if 'source_file' in df.columns else None})
    return df, log


# ---------------------------------------------------------------------------
# Input PDFs (for the side-by-side viewer)
# ---------------------------------------------------------------------------

def get_input_pdf_bytes(job_id: str, source_file: str, job=None, db_record: Optional[dict] = None) -> Optional[bytes]:
    """Pull one patient's PDF out of the job's input ZIP.

    source_file may be the renamed name (what the result table shows) — the stored
    source_file_map turns it back into the name inside the ZIP.
    """
    source_file_map = (db_record or {}).get('source_file_map') or {}
    if isinstance(source_file_map, str):
        try:
            source_file_map = json.loads(source_file_map)
        except Exception:
            source_file_map = {}
    if job is not None and isinstance(getattr(job, 'metadata', None), dict):
        source_file_map = job.metadata.get('source_file_map') or source_file_map

    candidates = [source_file]
    if source_file in source_file_map:
        candidates.insert(0, source_file_map[source_file])

    zip_path = None
    if job is not None and isinstance(getattr(job, 'metadata', None), dict):
        zp = job.metadata.get('input_zip_path')
        if zp and os.path.exists(zp):
            zip_path = zp
    if zip_path is None and db_record and db_record.get('input_zip_supabase_path'):
        REVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = REVIEW_CACHE_DIR / f"{job_id}_input.zip"
        if not cached.exists():
            from db_utils import download_from_supabase
            cached.write_bytes(download_from_supabase(db_record['input_zip_supabase_path']))
        zip_path = str(cached)
    if zip_path is None:
        return None

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        by_base = {}
        for n in names:
            by_base.setdefault(os.path.basename(n), n)
        for c in candidates:
            hit = by_base.get(c) or by_base.get(os.path.basename(c))
            if hit:
                return zf.read(hit)
    return None
