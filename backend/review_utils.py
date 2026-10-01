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
EDITABLE_FIELDS = [
    'Responsible Provider', 'MD', 'CRNA', 'Surgeon', 'An Start', 'An Stop', 'Points', 'Notes',
]
INSURANCE_CODE_FIELDS = ['Primary Mednet Code', 'Secondary Mednet Code', 'Tertiary Mednet Code']

# The Sorting tab's Insurance Code Prediction step gets its own review: the sorter checks
# the predicted MedNet code against the payer name/address on the record and fixes it.
INSURANCE_EDITABLE_FIELDS = [
    f'{tier} {col}'
    for tier in ('Primary', 'Secondary', 'Tertiary')
    for col in ('Mednet Code', 'Company Name', 'Company Address 1', 'Sub ID')
]

# Which fields a review screen may change. 'unified' = the Extract+CPT+ICD output,
# 'insurance' = the output of the Insurance Code Prediction job.
PROFILES = {
    'unified': EDITABLE_FIELDS,
    'insurance': INSURANCE_EDITABLE_FIELDS,
}
PROVIDER_FIELDS = ['Responsible Provider', 'MD', 'CRNA']

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
            xlsx_path = re.sub(r'\.csv$', '.xlsx', csv_path)  # keep the job's own naming
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

    if not db_record:
        size = os.path.getsize(xlsx_path) if os.path.exists(xlsx_path) else os.path.getsize(csv_path)
        return None, size
    supabase_path = db_record.get('supabase_path') or f"{job_id}_unified_result.xlsx"
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


# Lines the roster regex picks up that are not providers: placeholder entries and the
# "Always be X" single-provider instruction some templates use as an example.
_ROSTER_JUNK = re.compile(r'do not use|provider,\s*unknown|^\s*always\b', re.IGNORECASE)


def _clean_name(name: str) -> str:
    """Collapse whitespace, drop spaces before commas and a trailing period — the
    variations the rosters actually contain (e.g. 'NUNNEWAR, SACHIN , MD')."""
    return re.sub(r'\s+,', ',', re.sub(r'\s+', ' ', str(name or '').strip().rstrip('.')))


def _norm_name(name: str) -> str:
    return _clean_name(name).upper()


def _roles_for(title: str) -> List[str]:
    """Which review cells a provider may be picked in, from the credential suffix."""
    t = (title or '').upper()
    if t in ('MD', 'DO', 'MDA'):
        return ['Responsible Provider', 'MD']
    if t in ('CRNA', 'SRNA', 'AA'):
        return ['Responsible Provider', 'CRNA']
    return ['Responsible Provider', 'MD', 'CRNA']  # untitled: don't hide it anywhere


def _harvest_field(fields: list, field_name: str) -> List[str]:
    current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'current')
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from field_definitions import extract_provider_list_from_text
    f = next((x for x in fields if x.get('name') == field_name), None)
    if not f:
        return []
    text = max([f.get('description') or '', f.get('location') or '', f.get('output_format') or ''], key=len)
    return [n for n in extract_provider_list_from_text(text) if not _ROSTER_JUNK.search(n)]


def build_roster(template: Optional[dict]) -> dict:
    """Unified provider roster for a template.

    Providers live in two places, and only one is read by the pipeline at a time:
      * provider_mapping column — "NAME, CRED (MedNet Code: N)"; the pipeline uses it
        only when extract_providers_from_annotations is ON.
      * Responsible Provider / MD / CRNA field prose — harvested by regex; used when OFF.
    Some templates carry both and they drift apart. For the review pick-list we union
    the two, tag every name with its source(s), and mark whether it is in the list the
    pipeline actually uses ('in_use'), so the sorter can still pick a name that only
    exists in the other list but sees that it is off the active roster.

    Returns {providers, surgeons, has_mednet, provider_mode, sources, conflict}.
    """
    out = {'providers': [], 'surgeons': [], 'has_mednet': False, 'provider_mode': 'ai',
           'sources': {'mapping': 0, 'field': 0}, 'conflict': None}
    if not template:
        return out

    annotations_on = bool(template.get('extract_providers_from_annotations'))
    fields = (template.get('template_data') or {}).get('fields') or []
    active = 'mapping' if annotations_on else 'field'
    out['provider_mode'] = 'annotation' if annotations_on else 'ai'

    by_key: Dict[str, dict] = {}

    def add(name: str, source: str, title: str = '', code: Optional[str] = None, roles: Optional[List[str]] = None):
        name = _clean_name(name)
        if not name or _ROSTER_JUNK.search(name):
            return
        k = _norm_name(name)
        e = by_key.get(k)
        title = (title or _title_from_name(name)).upper()
        if e is None:
            e = {'name': name, 'code': None, 'title': title, 'sources': [], 'roles': [], 'in_use': False}
            by_key[k] = e
        if code and not e['code']:
            e['code'] = code
        if title and not e['title']:
            e['title'] = title
        if source not in e['sources']:
            e['sources'].append(source)
        for r in (roles or _roles_for(e['title'])):
            if r not in e['roles']:
                e['roles'].append(r)
        if source == active:
            e['in_use'] = True

    # 1. provider_mapping (MedNet-coded)
    pm_text = (template.get('provider_mapping') or '').strip()
    if pm_text:
        from provider_annotation_utils import parse_provider_mapping
        parsed = parse_provider_mapping(pm_text)
        for code, info in parsed.items():
            add(info['name'], 'mapping', info.get('title') or '', code)
        out['sources']['mapping'] = len(parsed)
        out['has_mednet'] = annotations_on and bool(parsed)

    # 2. field prose — the MD / CRNA fields fix the title even when the name lacks one
    field_names = set()
    for fname, forced_title in (('Responsible Provider', ''), ('MD', 'MD'), ('CRNA', 'CRNA')):
        for n in _harvest_field(fields, fname):
            add(n, 'field', forced_title)
            field_names.add(_norm_name(n))
    out['sources']['field'] = len(field_names)

    # 3. surgeons — separate list, own field
    seen = set()
    for n in _harvest_field(fields, 'Surgeon'):
        k = _norm_name(n)
        if k not in seen:
            seen.add(k)
            out['surgeons'].append({'name': _clean_name(n), 'sources': ['field'], 'in_use': True})

    mapping_keys = {k for k, e in by_key.items() if 'mapping' in e['sources']}
    if mapping_keys and field_names:
        fo, mo = len(field_names - mapping_keys), len(mapping_keys - field_names)
        if fo or mo:
            out['conflict'] = {'active': active, 'field_only': fo, 'mapping_only': mo}

    out['providers'] = sorted(by_key.values(), key=lambda e: (not e['in_use'], e['title'], e['name']))
    return out


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


def _insurance_needs_review(row: dict) -> List[str]:
    reasons = []
    for tier in ('Primary', 'Secondary'):
        name = str(row.get(f'{tier} Company Name', '')).strip()
        code = str(row.get(f'{tier} Mednet Code', '')).strip()
        if name and not code:
            reasons.append(f'{tier.lower()}-code-missing')
    if not str(row.get('Primary Company Name', '')).strip():
        reasons.append('no-primary-insurance')
    return reasons


def build_review_payload(job_id: str, df: pd.DataFrame, template: Optional[dict],
                         db_record: Optional[dict], file_source: str, profile: str = 'unified') -> dict:
    if profile == 'insurance':
        return _build_insurance_payload(job_id, df, db_record, file_source)
    r = build_roster(template)
    roster, has_mednet, provider_mode = r['providers'], r['has_mednet'], r['provider_mode']
    surgeons = r['surgeons']
    if db_record and db_record.get('provider_mode'):
        provider_mode = db_record['provider_mode']
    roster_source = 'template' if roster else 'none'

    # Names the extraction already produced that are in no list: still offer them (tagged)
    # so the sorter can keep an AI value without retyping it — but sees it is off-roster.
    def _extend(target: list, cols: tuple):
        known = {_norm_name(e['name']) for e in target}
        for col in cols:
            if col not in df.columns:
                continue
            for v in df[col].astype(str):
                v = _clean_name(v)
                if v and _norm_name(v) not in known:
                    known.add(_norm_name(v))
                    target.append({'name': v, 'code': None, 'title': _title_from_name(v), 'sources': ['results'],
                                   'roles': _roles_for(_title_from_name(v)), 'in_use': False})
    _extend(roster, tuple(PROVIDER_FIELDS))
    _extend(surgeons, ('Surgeon',))
    if roster_source == 'none' and roster:
        roster_source = 'results'

    columns = list(df.columns)
    editable = [f for f in EDITABLE_FIELDS if f in columns]
    context = [f for f in CONTEXT_FIELDS if f in columns]

    roster_names = {_norm_name(e['name']) for e in roster if e.get('in_use')}
    rows = []
    for i, rec in enumerate(df.to_dict(orient='records')):
        row = {'_row': i}
        row.update(rec)
        row['_needs_review'] = _row_needs_review(rec, provider_mode)
        row['_off_roster'] = [
            f for f in ('Responsible Provider', 'MD', 'CRNA')
            if f in rec and str(rec[f]).strip() and roster_names and _norm_name(rec[f]) not in roster_names
        ]
        rows.append(row)

    source_file_map = (db_record or {}).get('source_file_map') or {}
    if isinstance(source_file_map, str):
        try:
            source_file_map = json.loads(source_file_map)
        except Exception:
            source_file_map = {}

    insurance_names = {}

    return {
        'job_id': job_id,
        'profile': 'unified',
        'insurance_code_fields': [],
        'insurance_names': insurance_names,
        'group': (db_record or {}).get('worktracker_group'),
        'batch': (db_record or {}).get('worktracker_batch'),
        'template_id': (template or {}).get('id'),
        'template_name': (template or {}).get('name'),
        'provider_mode': provider_mode,
        'roster': roster,
        'surgeon_roster': surgeons,
        'roster_has_mednet': has_mednet,
        'roster_source': roster_source,
        'roster_sources': r['sources'],
        'roster_conflict': r['conflict'],
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
# Insurance (MedNet) lookup — sorting/mednet.csv is the master list
# ---------------------------------------------------------------------------

_INSURANCE_INDEX: Optional[List[dict]] = None


def _insurance_index() -> List[dict]:
    """Load sorting/mednet.csv once: [{code, name, plan, _k}] deduped by code."""
    global _INSURANCE_INDEX
    if _INSURANCE_INDEX is not None:
        return _INSURANCE_INDEX
    path = Path(__file__).resolve().parent / 'sorting' / 'mednet.csv'
    out: List[dict] = []
    seen = set()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors='replace')
        for rec in df.to_dict(orient='records'):
            code = str(rec.get('MedNet Code', '')).strip()
            name = str(rec.get('Name', '')).strip()
            if not code or not name or code.upper() in seen:
                continue
            seen.add(code.upper())
            out.append({'code': code, 'name': name, 'plan': str(rec.get('Insurance Plan', '')).strip(),
                        '_k': f"{name} {code}".upper()})
    except Exception as e:
        logger.error(f"Failed to load insurance index from {path}: {e}")
    _INSURANCE_INDEX = out
    return out


def search_insurance(q: str = '', codes: Optional[List[str]] = None, limit: int = 25) -> dict:
    """Name/code search (every token must appear) and exact code resolution."""
    idx = _insurance_index()
    result = {'matches': [], 'resolved': {}}
    if codes:
        want = {c.strip().upper() for c in codes if c and c.strip()}
        for e in idx:
            if e['code'].upper() in want:
                result['resolved'][e['code']] = e['name']
    q = (q or '').strip().upper()
    if q:
        tokens = q.split()
        exact = [e for e in idx if e['code'].upper() == q]
        rest = [e for e in idx if e not in exact and all(t in e['_k'] for t in tokens)]
        rest.sort(key=lambda e: (not e['code'].upper().startswith(q), not e['name'].upper().startswith(q), e['name']))
        result['matches'] = [{'code': e['code'], 'name': e['name'], 'plan': e['plan']} for e in (exact + rest)[:limit]]
    return result


def _build_insurance_payload(job_id: str, df: pd.DataFrame, db_record: Optional[dict], file_source: str) -> dict:
    columns = list(df.columns)
    editable = [f for f in INSURANCE_EDITABLE_FIELDS if f in columns]
    # Tertiary columns only earn their width when the batch actually has tertiary data
    tert_used = any(
        col in df.columns and df[col].astype(str).str.strip().ne('').any()
        for col in ('Tertiary Company Name', 'Tertiary Mednet Code')
    )
    if not tert_used:
        editable = [f for f in editable if not f.startswith('Tertiary')]
    context = [f for f in ('Patient Last Name', 'Patient First Name', 'Patient DOB', 'source_file') if f in columns]

    present_codes = set()
    for col in INSURANCE_CODE_FIELDS:
        if col in df.columns:
            present_codes.update(v.strip() for v in df[col].astype(str) if v.strip())
    insurance_names = search_insurance(codes=list(present_codes))['resolved'] if present_codes else {}

    rows = []
    for i, rec in enumerate(df.to_dict(orient='records')):
        row = {'_row': i}
        row.update(rec)
        row['_needs_review'] = _insurance_needs_review(rec)
        row['_off_roster'] = []
        rows.append(row)

    return {
        'job_id': job_id,
        'profile': 'insurance',
        'insurance_code_fields': [f for f in INSURANCE_CODE_FIELDS if f in editable],
        'insurance_names': insurance_names,
        'group': None, 'batch': None, 'template_id': None, 'template_name': None,
        'provider_mode': 'ai', 'roster': [], 'surgeon_roster': [], 'roster_has_mednet': False,
        'roster_source': 'none', 'roster_sources': {}, 'roster_conflict': None,
        'editable_fields': editable,
        'context_fields': context,
        'columns': columns,
        'rows': rows,
        'row_count': len(rows),
        'file_source': file_source,
        'has_input_zip': False,
        'source_file_map': {},
        'reviewed_at': None,
        'review_edit_count': 0,
    }


# ---------------------------------------------------------------------------
# Applying edits
# ---------------------------------------------------------------------------

def apply_edits(df: pd.DataFrame, edits: List[dict], profile: str = 'unified') -> Tuple[pd.DataFrame, List[dict]]:
    """Apply [{row, field, value}] to df in place. Returns (df, change_log).

    Only the profile's editable fields (plus SRNA, which the MedNet shortcut may set)
    are accepted; anything else is ignored rather than silently written into billing data.
    """
    allowed = set(PROFILES.get(profile, EDITABLE_FIELDS)) | ({'SRNA'} if profile == 'unified' else set())
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
