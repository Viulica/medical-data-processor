import os
import json
import csv
import glob
import tempfile
import sys
import time
import random

print("🚀 [STARTUP] Extraction script started", flush=True)

# Force unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)

print(f"🚀 [STARTUP] Python version: {sys.version}", flush=True)

# Try importing pandas
try:
    print("🚀 [STARTUP] Importing pandas...", flush=True)
    import pandas as pd
    print("✅ [STARTUP] pandas imported successfully", flush=True)
except Exception as e:
    print(f"❌ [STARTUP] Failed to import pandas: {e}", flush=True)
    sys.exit(1)

# Try importing google.genai with timeout handling
try:
    print("🚀 [STARTUP] Importing google.genai...", flush=True)
    import google.genai as genai
    from google.genai import types
    print("✅ [STARTUP] google.genai imported successfully", flush=True)
except Exception as e:
    print(f"❌ [STARTUP] Failed to import google.genai: {e}", flush=True)
    sys.exit(1)

# Try importing PyPDF2
try:
    print("🚀 [STARTUP] Importing PyPDF2...", flush=True)
    from PyPDF2 import PdfReader, PdfWriter
    print("✅ [STARTUP] PyPDF2 imported successfully", flush=True)
except Exception as e:
    print(f"❌ [STARTUP] Failed to import PyPDF2: {e}", flush=True)
    sys.exit(1)

# Try importing field_definitions
try:
    print("🚀 [STARTUP] Importing field_definitions...", flush=True)
    from field_definitions import get_fieldnames, generate_extraction_prompt, get_priority_fields, get_very_high_priority_fields, get_low_priority_fields, get_normal_fields, generate_priority_field_prompt, generate_priority_fields_group_prompt
    print("✅ [STARTUP] field_definitions imported successfully", flush=True)
except Exception as e:
    print(f"❌ [STARTUP] Failed to import field_definitions: {e}", flush=True)
    sys.exit(1)

# Hardcoded priority-field groups: fields listed together in the same tuple
# are extracted in a single API call when all of them are present as priority fields.
PRIORITY_FIELD_GROUPS = [
    ("An Start", "An Stop"),
    ("AnStart", "AnStop"),
    ("Responsible Provider", "MD", "CRNA"),
]


def _group_priority_fields(priority_fields):
    """Cluster priority fields by PRIORITY_FIELD_GROUPS. Returns a list of lists;
    each inner list is a group to extract in one API call. Fields not part of any
    group come back as singletons."""
    by_name = {f['name']: f for f in priority_fields}
    groups = []
    consumed = set()
    for group_names in PRIORITY_FIELD_GROUPS:
        members = [by_name[n] for n in group_names if n in by_name and n not in consumed]
        if len(members) >= 2:
            groups.append(members)
            consumed.update(f['name'] for f in members)
    for f in priority_fields:
        if f['name'] not in consumed:
            groups.append([f])
    return groups
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import base64
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path

# Thread-local storage for temporary files cleanup
thread_local = threading.local()


def extract_csn_from_filename(filename):
    """Extract CSN number from PDF filename.
    
    Looks for pattern like: CSN-100255177928 or _CSN-100255177928_
    Example: "24_CSN-100255177928_D.pdf" -> "100255177928"
    """
    if not filename:
        return None
    
    # Pattern to match CSN- followed by digits
    # Matches: CSN-123456789 or _CSN-123456789_ or similar variations
    pattern = r'CSN-(\d+)'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        return match.group(1)  # Return just the digits
    
    return None


def format_phone_number(phone_str):
    """Format phone number to add space after area code: (712)301-6622 -> (712) 301-6622"""
    if not phone_str or not isinstance(phone_str, str):
        return phone_str
    
    # Remove any existing spaces first to standardize
    phone_str = phone_str.strip().replace(' ', '')
    
    # Pattern to match phone numbers like (712)301-6622 or (712)3016622
    pattern = r'^\((\d{3})\)(\d{3})[-]?(\d{4})$'
    match = re.match(pattern, phone_str)
    
    if match:
        area_code, prefix, line = match.groups()
        return f"({area_code}) {prefix}-{line}"
    
    # If it doesn't match the expected pattern, return as is
    return phone_str


def clean_field_value(value):
    """Clean field values by removing unwanted characters like ? at the beginning"""
    if not value or not isinstance(value, str):
        return value
    
    # Remove common problematic characters at the beginning
    cleaned = value.strip()
    
    # Remove question marks at the beginning (common encoding issue)
    while cleaned.startswith('?'):
        cleaned = cleaned[1:].strip()
    
    # Remove other common invisible/problematic characters
    # Remove zero-width space, non-breaking space, BOM, etc.
    cleaned = cleaned.lstrip('\ufeff\u200b\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000')
    
    # Remove any remaining leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Remove any question marks that might appear in the middle due to encoding issues
    # This is more aggressive cleaning for problematic characters
    cleaned = cleaned.replace('?', '')
    
    # CRITICAL FIX: Remove newlines that break CSV structure in Excel
    # Replace newlines with semicolons for addresses to maintain readability
    cleaned = cleaned.replace('\n', '; ').replace('\r', '; ')
    
    # Remove multiple consecutive semicolons and spaces
    while '; ; ' in cleaned:
        cleaned = cleaned.replace('; ; ', '; ')
    
    # Remove trailing semicolons
    cleaned = cleaned.rstrip('; ')
    
    return cleaned


def extract_first_n_pages_as_pdf(input_pdf_path, n_pages=2):
    """Extract the first n pages from PDF and return as temporary PDF file."""
    try:
        reader = PdfReader(input_pdf_path)
        writer = PdfWriter()
        
        total_pages = len(reader.pages)
        pages_to_extract = min(n_pages, total_pages)
        
        print(f"    📄 Extracting first {pages_to_extract} pages from {total_pages} total pages")
        
        # Add the first n pages
        for page_idx in range(pages_to_extract):
            writer.add_page(reader.pages[page_idx])
        
        # Create temporary file for the combined pages
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.PDF')
        writer.write(temp_file)
        temp_file.close()
        
        return temp_file.name
            
    except Exception as e:
        print(f"    ❌ Error extracting first {n_pages} pages: {str(e)}")
        return None


# ---------------------------------------------------------------------------
# EXPERIMENT: self-hosted vLLM routing (revert by deleting this block + the
# is_vllm_model() call at the top of extract_info_from_patient_pdf).
# Enabled only when VLLM_BASE_URL is set, so production paths are unaffected.
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
VLLM_THINKING = os.getenv("VLLM_THINKING", "0") == "1"


def is_vllm_model(model_name):
    """Route self-hosted checkpoints (nvidia/..., Qwen/...) to the vLLM server."""
    if not VLLM_BASE_URL or not model_name:
        return False
    return model_name.startswith("nvidia/") or model_name.startswith("Qwen/")


def extract_with_vllm(patient_pdf_path, pdf_filename, extraction_prompt, model,
                      max_retries=3, field_name_for_log=None):
    """Extract via an OpenAI-compatible vLLM endpoint using the image route.

    Mirrors extract_with_openrouter's image path; the self-hosted server has no
    30 MB cap and no direct-PDF (file_data) support, so images are always used.
    """
    log_suffix = f" - {field_name_for_log}" if field_name_for_log else ""

    image_data_list = pdf_to_images_base64(patient_pdf_path)
    if not image_data_list:
        print(f"    ❌ Failed to convert PDF to images for {pdf_filename}{log_suffix}")
        return None

    messages = _build_image_messages(extraction_prompt, image_data_list)
    headers = {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": VLLM_THINKING},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(f"{VLLM_BASE_URL}/chat/completions",
                                     headers=headers, json=payload,
                                     timeout=900, verify=False)
            response.raise_for_status()
            msg = response.json()['choices'][0]['message']
            # With thinking enabled the answer may land in 'reasoning'.
            response_text = (msg.get('content') or msg.get('reasoning') or "").strip()
            if not response_text or len(response_text) < 10:
                raise ValueError(f"Response too short or empty: {response_text!r}")

            # Same fence-stripping + JSON validation as the OpenRouter path.
            cleaned_response = response_text
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            json.loads(cleaned_response.strip())  # Validate JSON

            print(f"    ✅ Successfully processed {pdf_filename}{log_suffix} with vLLM on attempt {attempt + 1}")
            return response_text, "vllm"
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 8)
                print(f"    ⚠️  vLLM error for {pdf_filename}{log_suffix}: {e}; "
                      f"retry {attempt+1}/{max_retries} in {wait_time}s")
                time.sleep(wait_time)
            else:
                print(f"    ❌ vLLM failed for {pdf_filename}{log_suffix}: {e}")
    return None, None


def is_openrouter_model(model_name):
    """Check if model name indicates OpenRouter (contains '/' or starts with 'google/')"""
    return '/' in model_name or model_name.startswith('google/')

def is_gemini_model(model_name):
    """Check if model name indicates a Gemini model (not OpenRouter format)"""
    # Remove OpenRouter prefixes/suffixes if present
    clean_model = model_name.replace('google/', '').replace(':online', '')
    return clean_model.startswith('gemini') or 'gemini' in clean_model.lower()

def normalize_gemini_model(model_name):
    """Normalize Gemini model name by removing OpenRouter prefixes/suffixes and normalizing spaces"""
    # Remove OpenRouter format prefixes/suffixes and models/ prefix
    clean_model = model_name.replace('google/', '').replace(':online', '').replace('models/', '')
    # Replace spaces with hyphens for consistency (e.g., "gemini flash lite latest" -> "gemini-flash-lite-latest")
    clean_model = clean_model.replace(' ', '-').lower()
    return clean_model

def pdf_to_images_base64(pdf_path, max_pages=100):
    """Convert PDF pages to base64 encoded images for OpenRouter"""
    try:
        # Use PyMuPDF (fitz) which is already in requirements
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        image_data_list = []
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))  # 200 DPI
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            image_data_list.append(img_base64)
        doc.close()
        return image_data_list
    except ImportError:
        print(f"    ⚠️  PyMuPDF (fitz) not available for PDF to image conversion")
        return []
    except Exception as e:
        print(f"    ⚠️  Failed to convert PDF to images: {str(e)}")
        return []

def _build_image_messages(extraction_prompt, image_data_list):
    """Build OpenRouter messages payload with PDF-rendered images."""
    content = [{"type": "text", "text": extraction_prompt}]
    for img_data in image_data_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_data}"}
        })
    return [{"role": "user", "content": content}]


def _build_pdf_file_messages(extraction_prompt, patient_pdf_path, pdf_filename):
    """Build OpenRouter messages payload using direct PDF upload (file_data).

    Used as a fallback when image-route fails due to OpenRouter's 30 MB image
    payload limit. Gemini handles PDFs natively, so this bypasses the cap.
    """
    import base64 as _b64
    try:
        with open(patient_pdf_path, "rb") as _fp:
            _pdf_bytes = _fp.read()
        _pdf_b64 = _b64.b64encode(_pdf_bytes).decode("ascii")
    except Exception as _e:
        print(f"    ⚠️  Failed to read PDF bytes for direct-PDF fallback: {_e}")
        return None
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": extraction_prompt},
            {"type": "file", "file": {
                "filename": pdf_filename or os.path.basename(patient_pdf_path),
                "file_data": f"data:application/pdf;base64,{_pdf_b64}",
            }},
        ],
    }]


# OpenRouter caps the *image* payload at 30 MB. If our base64-encoded images
# approach this, switch to direct PDF mode preemptively. 25 MB headroom.
_OPENROUTER_IMAGE_PAYLOAD_BUDGET = 25 * 1024 * 1024  # 25 MB of base64 image data


def extract_with_openrouter(patient_pdf_path, pdf_filename, extraction_prompt, model, max_retries=5, field_name_for_log=None):
    """Extract patient information using OpenRouter API.

    Primary path: render PDF pages to PNG images and send via image_url parts.
    Fallback path: when total image payload would exceed OpenRouter's 30 MB
    cap (or a 413 is returned), send the PDF directly via the file_data
    content type, which Gemini handles natively.
    """
    log_suffix = f" - {field_name_for_log}" if field_name_for_log else ""

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"    ❌ OpenRouter API key not found for {pdf_filename}{log_suffix}")
        return None

    # Convert PDF to images
    image_data_list = pdf_to_images_base64(patient_pdf_path)
    if not image_data_list:
        print(f"    ❌ Failed to convert PDF to images for {pdf_filename}{log_suffix}")
        return None

    # Decide between image route and direct-PDF route based on total image size.
    image_payload_size = sum(len(d) for d in image_data_list)
    using_direct_pdf = False
    if image_payload_size > _OPENROUTER_IMAGE_PAYLOAD_BUDGET:
        print(
            f"    📦 {pdf_filename}{log_suffix}: image payload ~{image_payload_size/1024/1024:.1f} MB "
            f"exceeds OpenRouter image cap; using direct-PDF upload (file_data)."
        )
        messages = _build_pdf_file_messages(extraction_prompt, patient_pdf_path, pdf_filename)
        if not messages:
            # fallback to images even if oversized, let the API reject it
            messages = _build_image_messages(extraction_prompt, image_data_list)
        else:
            using_direct_pdf = True
    else:
        messages = _build_image_messages(extraction_prompt, image_data_list)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/medical-data-processor",
        "X-Title": "Medical Data Processor"
    }
    
    # Ensure DeepSeek model uses correct OpenRouter format
    openrouter_model = model
    if "deepseek" in model.lower():
        model_lower = model.lower()
        if "reasoner" in model_lower:
            openrouter_model = "deepseek/deepseek-reasoner"
        elif "chat" in model_lower or "v3" in model_lower or "v3.2" in model_lower:
            openrouter_model = "deepseek/deepseek-chat"
        elif model.startswith("deepseek/"):
            openrouter_model = model
        else:
            openrouter_model = "deepseek/deepseek-chat"  # Default to most common model
    
    flex_disabled = False
    flex_503_retries = 0
    for attempt in range(max_retries):
        try:
            payload = {
                "model": openrouter_model,
                "messages": messages,
                "usage": {"include": True},
            }

            # Enable reasoning + flex tier for Gemini 3 models via OpenRouter (half-price)
            if "gemini-3" in openrouter_model:
                payload["reasoning"] = {"effort": "high"}
                if not flex_disabled:
                    payload["service_tier"] = "flex"
                    payload["provider"] = {"sort": "throughput"}

            response = requests.post(url, headers=headers, json=payload, timeout=300)
            # On flex tier 503 (capacity exhausted), retry twice on flex, then fall back to standard tier
            if response.status_code == 503 and payload.get("service_tier") == "flex":
                if flex_503_retries < 2:
                    flex_503_retries += 1
                    wait_time = 2 ** flex_503_retries
                    print(f"    ⚠️  OpenRouter 503 on flex tier for {pdf_filename}{log_suffix}; flex retry {flex_503_retries}/2 in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"    ⚠️  OpenRouter 503 on flex tier for {pdf_filename}{log_suffix} after 2 retries; falling back to standard tier")
                    flex_disabled = True
                    continue
            response.raise_for_status()

            result = response.json()

            # OpenRouter sometimes returns HTTP 200 with {"error": {...}} body when the
            # upstream provider rate-limits or fails. Route transient codes through the
            # same flex->standard fallback path used for HTTP 503-on-flex above.
            err_body = result.get("error") if isinstance(result, dict) else None
            if err_body and "choices" not in result:
                err_code = err_body.get("code") if isinstance(err_body, dict) else None
                msg = (err_body.get("message") if isinstance(err_body, dict) else str(err_body)) or ""
                transient = err_code in (429, 502, 503, 504) or any(
                    x in str(msg).lower() for x in ("rate limit", "overloaded", "temporarily", "provider returned error")
                )
                if transient:
                    if payload.get("service_tier") == "flex":
                        if flex_503_retries < 2:
                            flex_503_retries += 1
                            wait_time = 2 ** flex_503_retries
                            print(f"    ⚠️  OpenRouter 200+error-body on flex tier for {pdf_filename}{log_suffix} (code={err_code}); flex retry {flex_503_retries}/2 in {wait_time}s")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"    ⚠️  OpenRouter 200+error-body on flex tier for {pdf_filename}{log_suffix} after 2 retries; falling back to standard tier")
                            flex_disabled = True
                            continue
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 8)
                        print(f"    ⚠️  OpenRouter 200+error-body for {pdf_filename}{log_suffix} (code={err_code}); retry {attempt+1}/{max_retries} in {wait_time}s")
                        time.sleep(wait_time)
                        continue

            response_text = result['choices'][0]['message']['content'].strip()
            
            # Validate response
            if not response_text or len(response_text) < 10:
                raise ValueError(f"Response too short or empty: {response_text}")
            
            # Try to parse JSON to validate format
            cleaned_response = response_text
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            json.loads(cleaned_response)  # Validate JSON
            
            print(f"    ✅ Successfully processed {pdf_filename}{log_suffix} with OpenRouter on attempt {attempt + 1}")
            return response_text, "openrouter"

        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parsing failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"    ❌ Final JSON parsing failure for {pdf_filename}{log_suffix}")
                return None, None
        except Exception as e:
            print(f"    ⚠️  OpenRouter API call failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            # DEBUG: dump response body on 400 so we can see the actual error reason
            body_preview_lc = ""
            try:
                if hasattr(e, 'response') and e.response is not None:
                    body_preview = e.response.text[:2000]
                    body_preview_lc = body_preview.lower()
                    print(f"    🔎 Response body: {body_preview}")
            except Exception:
                pass

            # If image payload was too big, swap to direct-PDF upload and retry.
            # Catches both HTTP 413 and HTTP 400 variants (OpenRouter sometimes
            # returns 400 with a size-related message when the upstream provider
            # rejects an oversized inline image before the gateway 413 fires).
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            size_keywords = ("payload too large", "cannot exceed 30mb", "request entity", "too large", "exceeds", "size limit")
            is_size_error = (
                status_code == 413
                or "413" in str(e)
                or any(kw in body_preview_lc for kw in size_keywords)
                or (status_code == 400 and any(kw in body_preview_lc for kw in size_keywords))
            )
            if is_size_error and not using_direct_pdf:
                pdf_messages = _build_pdf_file_messages(extraction_prompt, patient_pdf_path, pdf_filename)
                if pdf_messages:
                    print(f"    🔁 Switching {pdf_filename}{log_suffix} to direct-PDF upload (file_data) after {status_code} size error; retrying.")
                    messages = pdf_messages
                    using_direct_pdf = True
                    # Don't burn a retry attempt — proceed to backoff/retry block.
            if attempt == max_retries - 1:
                print(f"    ❌ Final OpenRouter API failure for {pdf_filename}{log_suffix}")
                return None, None

        # Exponential backoff
        if attempt < max_retries - 1:
            base_delay = 2 ** attempt
            jitter = random.uniform(0.5, 1.5)
            delay = base_delay * jitter
            print(f"    ⏳ Retrying {pdf_filename}{log_suffix} in {delay:.1f} seconds...")
            time.sleep(delay)

    return None, None

def extract_info_from_patient_pdf(client, patient_pdf_path, pdf_filename, extraction_prompt, model="gemini-flash-latest", max_retries=5, field_name_for_log=None):
    """Extract patient information from a multi-page patient PDF file.
    
    Args:
        client: Google GenAI client (ignored if using OpenRouter)
        patient_pdf_path: Path to PDF file
        pdf_filename: Name of PDF file for logging
        extraction_prompt: Prompt for extraction
        model: Model name (if contains '/', uses OpenRouter)
        max_retries: Maximum retry attempts
        field_name_for_log: Optional field name to include in log messages (for priority field extraction)
    """
    
    # EXPERIMENT: self-hosted vLLM. Must precede the OpenRouter check because
    # "nvidia/..." also contains "/". No-op unless VLLM_BASE_URL is set.
    if is_vllm_model(model):
        return extract_with_vllm(patient_pdf_path, pdf_filename, extraction_prompt, model, 3, field_name_for_log)

    # Check if using OpenRouter (explicitly, or Gemini model without GOOGLE_API_KEY)
    if is_openrouter_model(model) and not is_gemini_model(model):
        return extract_with_openrouter(patient_pdf_path, pdf_filename, extraction_prompt, model, max_retries, field_name_for_log)

    # Explicit OpenRouter routing: if the caller passed a "google/..." prefixed model,
    # always route through OpenRouter even though it's also a Gemini model.
    if model.startswith('google/'):
        return extract_with_openrouter(patient_pdf_path, pdf_filename, extraction_prompt, model, max_retries, field_name_for_log)

    original_model_for_fallback = model
    if is_gemini_model(model):
        # If no GOOGLE_API_KEY and no client, fall back to OpenRouter
        if client is None and not os.environ.get("GOOGLE_API_KEY"):
            # Use gemini-3-pro-preview for standard tab, otherwise preserve selected model
            if normalize_gemini_model(model) == "gemini-3-pro-preview":
                openrouter_model = "google/gemini-3-pro-preview"
            else:
                openrouter_model = f"google/{normalize_gemini_model(model)}"
            return extract_with_openrouter(patient_pdf_path, pdf_filename, extraction_prompt, openrouter_model, max_retries, field_name_for_log)
        model = normalize_gemini_model(model)

    def _try_openrouter_fallback(reason):
        """Fall through to OpenRouter when the Google path is unrecoverable."""
        if not is_gemini_model(original_model_for_fallback):
            return None, None
        if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            print(f"    ⚠️  Cannot fall back to OpenRouter for {pdf_filename}: no OPENROUTER_API_KEY/OPENAI_API_KEY set")
            return None, None
        normalized = normalize_gemini_model(original_model_for_fallback)
        or_model = f"google/{normalized}"
        print(f"    🔁 Google path exhausted for {pdf_filename} ({reason}); falling back to OpenRouter as {or_model}")
        result = extract_with_openrouter(patient_pdf_path, pdf_filename, extraction_prompt, or_model, max_retries, field_name_for_log)
        if isinstance(result, tuple):
            response_text, _ = result
            if response_text:
                return response_text, "openrouter (google fallback)"
            return None, None
        return result, "openrouter (google fallback)" if result else (None, None)

    def _is_unrecoverable_google_error(exc):
        """Detect errors where retrying Google won't help (rate limit, payment, auth)."""
        msg = str(exc).lower()
        markers = ("429", "too many requests", "rate limit", "quota",
                   "402", "payment required",
                   "401", "403", "permission denied", "unauthenticated",
                   "resource_exhausted")
        return any(m in msg for m in markers)
    
    log_suffix = f" - {field_name_for_log}" if field_name_for_log else ""
    
    use_flex = False  # Flex disabled for now due to 503 retry storms
    FLEX_TIMEOUT = 600  # 10 minutes

    for attempt in range(max_retries):
        try:
            with open(patient_pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type="application/pdf",
                            data=pdf_data,
                        ),
                        types.Part.from_text(text=extraction_prompt)],
                    )
                ]

            # Use thinking_budget=-1 for gemini-3-pro-preview, "MEDIUM" for gemini-3-flash-preview, thinking_budget=-1 for others
            if model == "gemini-3-pro-preview":
                thinking_config = types.ThinkingConfig(
                    thinking_budget=-1,
                )
            elif model == "gemini-3-flash-preview":
                thinking_config = types.ThinkingConfig(
                    thinking_budget=-1,
                )
            else:
                thinking_config = types.ThinkingConfig(
                    thinking_budget=-1,
                )

            tools = [
                types.Tool(code_execution=types.ToolCodeExecution),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config=thinking_config,
                media_resolution="MEDIA_RESOLUTION_HIGH",
                tools=tools,
                http_options=types.HttpOptions(
                    timeout=120 * 1000,
                ),
            )

            # Non-streaming call to get response headers (service tier info)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )

                response_text = response.text.strip() if response.text else ""

                # Read actual service tier from response headers
                actual_tier = "unknown"
                try:
                    resp_dict = response.to_json_dict()
                    actual_tier = resp_dict.get('sdk_http_response', {}).get('headers', {}).get('x-gemini-service-tier', 'standard')
                except:
                    actual_tier = "flex" if use_flex else "standard"

                # Validate that we got a meaningful response
                if not response_text or len(response_text) < 10:
                    raise ValueError(f"Response too short or empty: {response_text}")

                # Try to parse JSON to validate response format
                cleaned_response = response_text
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                json.loads(cleaned_response)

                # If we fell back from flex, mark it
                if not use_flex and actual_tier == "standard":
                    actual_tier = "standard (fallback)"
                print(f"    ✅ Successfully processed {pdf_filename}{log_suffix} on attempt {attempt + 1} (tier: {actual_tier})")
                return response_text, actual_tier

            except json.JSONDecodeError as e:
                print(f"    ⚠️  JSON parsing failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    print(f"    ❌ Final JSON parsing failure for {pdf_filename}{log_suffix}")
                    print(f"    Raw response: {response_text[:200]}...")
                    return _try_openrouter_fallback("Google returned unparseable JSON")

            except (TimeoutError, Exception) as api_error:
                is_timeout = isinstance(api_error, TimeoutError)
                if use_flex:
                    reason = "timeout" if is_timeout else str(api_error)[:80]
                    print(f"    ⚠️  Flex failed for {pdf_filename}{log_suffix} ({reason}), switching to standard tier")
                    use_flex = False
                    continue
                print(f"    ⚠️  API call failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(api_error)}")
                # Skip retries on unrecoverable errors (rate limit, payment required, auth) — fall back immediately
                if _is_unrecoverable_google_error(api_error):
                    print(f"    ⛔ Unrecoverable Google error for {pdf_filename}{log_suffix}; skipping further retries")
                    return _try_openrouter_fallback(f"unrecoverable: {str(api_error)[:120]}")
                if attempt == max_retries - 1:
                    print(f"    ❌ Final API failure for {pdf_filename}{log_suffix}")
                    return _try_openrouter_fallback(f"Google API failure: {str(api_error)[:120]}")

        except Exception as e:
            if use_flex:
                print(f"    ⚠️  Flex failed for {pdf_filename}{log_suffix} ({str(e)[:80]}), switching to standard tier")
                use_flex = False
                continue
            print(f"    ⚠️  Unexpected error for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if _is_unrecoverable_google_error(e):
                print(f"    ⛔ Unrecoverable Google error for {pdf_filename}{log_suffix}; skipping further retries")
                return _try_openrouter_fallback(f"unrecoverable: {str(e)[:120]}")
            if attempt == max_retries - 1:
                print(f"    ❌ Final failure for {pdf_filename}{log_suffix}")
                return _try_openrouter_fallback(f"Google unexpected error: {str(e)[:120]}")

        # Exponential backoff with jitter for retries
        if attempt < max_retries - 1:
            base_delay = 2 ** attempt
            jitter = random.uniform(0.5, 1.5)
            delay = base_delay * jitter
            print(f"    ⏳ Retrying {pdf_filename}{log_suffix} in {delay:.1f} seconds...")
            time.sleep(delay)

    return _try_openrouter_fallback("Google retries exhausted")


def process_single_patient_pdf_task(args):
    """Task function for processing a single patient PDF in a thread."""
    import time
    task_start = time.time()

    client, pdf_file_path, extraction_prompt, priority_fields, low_priority_fields, excel_file_path, n_pages, model, priority_model, low_priority_model, order_index, provider_mapping, extract_providers_from_annotations, very_high_priority_fields, very_high_priority_model, pb_provider_mapping, pb_has_mednet = args

    pdf_filename = os.path.basename(pdf_file_path)
    is_problem_pdf = pdf_filename in ['Record_04.pdf', 'Record_07.pdf']

    if is_problem_pdf:
        print(f"  🔴 [DEBUG] STARTING {pdf_filename} (problematic PDF)")

    # Extract first n pages as temporary PDF
    extract_start = time.time()
    temp_patient_pdf = extract_first_n_pages_as_pdf(pdf_file_path, n_pages)
    extract_time = time.time() - extract_start

    if is_problem_pdf:
        print(f"  🔴 [DEBUG] Extracted pages for {pdf_filename} in {extract_time:.1f}s")
    if not temp_patient_pdf:
        return pdf_filename, None, temp_patient_pdf, order_index
    
    # Extract normal (non-priority) fields with one API call
    # Pass client (may be None for OpenRouter)
    if is_problem_pdf:
        print(f"  🔴 [DEBUG] Calling extract_info_from_patient_pdf for {pdf_filename}...")
        api_start = time.time()

    normal_result = extract_info_from_patient_pdf(client, temp_patient_pdf, pdf_filename, extraction_prompt, model)

    if is_problem_pdf:
        api_time = time.time() - api_start
        print(f"  🔴 [DEBUG] API call completed for {pdf_filename} in {api_time:.1f}s")

    normal_response, service_tier = normal_result if isinstance(normal_result, tuple) else (normal_result, None)

    if not normal_response:
        if is_problem_pdf:
            print(f"  🔴 [DEBUG] No response for {pdf_filename}!")
        return pdf_filename, None, temp_patient_pdf, order_index
    
    # Parse the normal response
    try:
        cleaned_response = normal_response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        merged_data = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"    ❌ Failed to parse normal fields response for {pdf_filename}: {str(e)}")
        return pdf_filename, None, temp_patient_pdf, order_index
    
    # Extract priority fields (some grouped into a single call) and merge into the result
    if priority_fields:
        priority_groups = _group_priority_fields(priority_fields)
        bundled = [g for g in priority_groups if len(g) > 1]
        print(f"    🎯 Processing {len(priority_fields)} priority field(s) in {len(priority_groups)} call(s) for {pdf_filename}")
        if bundled:
            for g in bundled:
                print(f"    🔗 Bundling priority group [{' + '.join(f['name'] for f in g)}] into 1 call for {pdf_filename}")

        for group in priority_groups:
            group_names = [f['name'] for f in group]
            log_label = " + ".join(group_names)
            priority_prompt = generate_priority_fields_group_prompt(
                group,
                provider_mapping=pb_provider_mapping,
                provider_mapping_has_mednet=pb_has_mednet,
            )

            priority_result = extract_info_from_patient_pdf(
                client, temp_patient_pdf, pdf_filename, priority_prompt, priority_model,
                field_name_for_log=log_label
            )
            priority_response = priority_result[0] if isinstance(priority_result, tuple) else priority_result

            if priority_response:
                try:
                    cleaned_priority = priority_response.strip()
                    if cleaned_priority.startswith('```json'):
                        cleaned_priority = cleaned_priority[7:]
                    if cleaned_priority.startswith('```'):
                        cleaned_priority = cleaned_priority[3:]
                    if cleaned_priority.endswith('```'):
                        cleaned_priority = cleaned_priority[:-3]
                    cleaned_priority = cleaned_priority.strip()

                    priority_data = json.loads(cleaned_priority)

                    for field_name in group_names:
                        if field_name in priority_data:
                            merged_data[field_name] = priority_data[field_name]
                            print(f"    ✅ Merged priority field '{field_name}' for {pdf_filename}")
                        else:
                            print(f"    ⚠️  Priority field '{field_name}' not found in response for {pdf_filename}")

                except json.JSONDecodeError as e:
                    print(f"    ❌ Failed to parse priority group '{log_label}' for {pdf_filename}: {str(e)}")
            else:
                print(f"    ❌ Failed to extract priority group '{log_label}' for {pdf_filename}")

    # Extract each low-priority field separately using cheaper model
    if low_priority_fields:
        print(f"    ⚡ Processing {len(low_priority_fields)} low-priority field(s) for {pdf_filename} (fast model)")

        for lp_field in low_priority_fields:
            field_name = lp_field['name']
            lp_prompt = generate_priority_field_prompt(
                lp_field,
                provider_mapping=pb_provider_mapping,
                provider_mapping_has_mednet=pb_has_mednet,
            )

            lp_result = extract_info_from_patient_pdf(
                client, temp_patient_pdf, pdf_filename, lp_prompt, low_priority_model,
                field_name_for_log=field_name
            )
            lp_response = lp_result[0] if isinstance(lp_result, tuple) else lp_result

            if lp_response:
                try:
                    cleaned_lp = lp_response.strip()
                    if cleaned_lp.startswith('```json'):
                        cleaned_lp = cleaned_lp[7:]
                    if cleaned_lp.startswith('```'):
                        cleaned_lp = cleaned_lp[3:]
                    if cleaned_lp.endswith('```'):
                        cleaned_lp = cleaned_lp[:-3]
                    cleaned_lp = cleaned_lp.strip()

                    lp_data = json.loads(cleaned_lp)

                    if field_name in lp_data:
                        merged_data[field_name] = lp_data[field_name]
                        print(f"    ✅ Merged low-priority field '{field_name}' for {pdf_filename}")
                    else:
                        print(f"    ⚠️  Low-priority field '{field_name}' not found in response for {pdf_filename}")

                except json.JSONDecodeError as e:
                    print(f"    ❌ Failed to parse low-priority field '{field_name}' for {pdf_filename}: {str(e)}")
            else:
                print(f"    ❌ Failed to extract low-priority field '{field_name}' for {pdf_filename}")

    # Extract each very-high-priority field separately using best model
    if very_high_priority_fields:
        print(f"    🔥 Processing {len(very_high_priority_fields)} very-high-priority field(s) for {pdf_filename} (pro model)")

        for vhp_field in very_high_priority_fields:
            field_name = vhp_field['name']
            vhp_prompt = generate_priority_field_prompt(
                vhp_field,
                provider_mapping=pb_provider_mapping,
                provider_mapping_has_mednet=pb_has_mednet,
            )

            vhp_result = extract_info_from_patient_pdf(
                client, temp_patient_pdf, pdf_filename, vhp_prompt, very_high_priority_model,
                field_name_for_log=field_name
            )
            vhp_response = vhp_result[0] if isinstance(vhp_result, tuple) else vhp_result

            if vhp_response:
                try:
                    cleaned_vhp = vhp_response.strip()
                    if cleaned_vhp.startswith('```json'):
                        cleaned_vhp = cleaned_vhp[7:]
                    if cleaned_vhp.startswith('```'):
                        cleaned_vhp = cleaned_vhp[3:]
                    if cleaned_vhp.endswith('```'):
                        cleaned_vhp = cleaned_vhp[:-3]
                    cleaned_vhp = cleaned_vhp.strip()

                    vhp_data = json.loads(cleaned_vhp)

                    if field_name in vhp_data:
                        merged_data[field_name] = vhp_data[field_name]
                        print(f"    ✅ Merged very-high-priority field '{field_name}' for {pdf_filename}")
                    else:
                        print(f"    ⚠️  Very-high-priority field '{field_name}' not found in response for {pdf_filename}")

                except json.JSONDecodeError as e:
                    print(f"    ❌ Failed to parse very-high-priority field '{field_name}' for {pdf_filename}: {str(e)}")
            else:
                print(f"    ❌ Failed to extract very-high-priority field '{field_name}' for {pdf_filename}")

    # Extract providers and/or insurance from PDF annotations if enabled
    if extract_providers_from_annotations:
        try:
            # Import the provider annotation utility
            import sys
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from provider_annotation_utils import extract_annotations_data

            ann = extract_annotations_data(pdf_file_path, provider_mapping)

            # Providers
            if ann['responsible']:
                merged_data['Responsible Provider'] = ann['responsible']
                print(f"    ✅ Set Responsible Provider from annotation: {ann['responsible']}")
            if ann['md']:
                merged_data['MD'] = ann['md']
                print(f"    ✅ Set MD from annotation: {ann['md']}")
            if ann['crna']:
                merged_data['CRNA'] = ann['crna']
                print(f"    ✅ Set CRNA from annotation: {ann['crna']}")
            if ann['has_srna']:
                merged_data['SRNA'] = 'SRNA, SRNA, SRNA'
                print(f"    ✅ Set SRNA from annotation: SRNA, SRNA, SRNA")

            # Insurance
            if ann['primary_mednet']:
                merged_data['Primary Mednet Code'] = ann['primary_mednet']
                print(f"    ✅ Set Primary Mednet Code from annotation: {ann['primary_mednet']}")
            if ann['secondary_mednet']:
                merged_data['Secondary Mednet Code'] = ann['secondary_mednet']
                print(f"    ✅ Set Secondary Mednet Code from annotation: {ann['secondary_mednet']}")
            if ann['tertiary_mednet']:
                merged_data['Tertiary Mednet Code'] = ann['tertiary_mednet']
                print(f"    ✅ Set Tertiary Mednet Code from annotation: {ann['tertiary_mednet']}")

        except Exception as e:
            print(f"    ⚠️  Failed to extract annotations for {pdf_filename}: {e}")
    
    # Copy Surgeon to Referring (they are the same field)
    if 'Surgeon' in merged_data and merged_data['Surgeon']:
        merged_data['Referring'] = merged_data['Surgeon']

    # Add service tier info to output
    if service_tier:
        merged_data['service_tier'] = service_tier

    # Convert merged data back to JSON string for compatibility with existing code
    merged_response = json.dumps(merged_data)

    return pdf_filename, merged_response, temp_patient_pdf, order_index


def process_all_patient_pdfs(input_folder="input", excel_file_path="WPA for testing FINAL.xlsx", n_pages=2, max_workers=50, model="gemini-flash-latest", priority_model="gemini-flash-latest", low_priority_model="google/gemini-3.1-flash-lite-preview", very_high_priority_model="gemini-3.1-pro-preview", worktracker_group=None, worktracker_batch=None, extract_csn=False, progress_file=None, provider_mapping=None, extract_providers_from_annotations=False, scanned_date=None):
    """Process all patient PDFs in the input folder, combining first n pages per patient into one CSV."""
    
    print(f"🚀 process_all_patient_pdfs called with progress_file={progress_file}, extract_providers_from_annotations={extract_providers_from_annotations}")
    
    # Check if Excel file exists
    if not os.path.exists(excel_file_path):
        print(f"❌ Error: Excel file '{excel_file_path}' not found!")
        return
    
    print(f"📋 Using field definitions from: {excel_file_path}")
    print(f"📄 Processing first {n_pages} pages per patient PDF")
    print(f"🧵 Max concurrent threads: {max_workers}")
    
    # Get priority and normal fields
    very_high_priority_fields = get_very_high_priority_fields(excel_file_path)
    priority_fields = get_priority_fields(excel_file_path)
    low_priority_fields_list = get_low_priority_fields(excel_file_path)
    normal_fields = get_normal_fields(excel_file_path)

    if very_high_priority_fields:
        very_high_field_names = [f['name'] for f in very_high_priority_fields]
        print(f"🔥 Very-high-priority fields (separate API calls): {', '.join(very_high_field_names)}")
        print(f"🤖 Using model '{very_high_priority_model}' for very-high-priority fields (pro model)")
    if priority_fields:
        priority_field_names = [f['name'] for f in priority_fields]
        print(f"🎯 High-priority fields (separate API calls): {', '.join(priority_field_names)}")
        print(f"🤖 Using model '{priority_model}' for high-priority fields (better accuracy)")
    if low_priority_fields_list:
        low_priority_field_names = [f['name'] for f in low_priority_fields_list]
        print(f"⚡ Low-priority fields (separate API calls): {', '.join(low_priority_field_names)}")
        print(f"⚡ Using model '{low_priority_model}' for low-priority fields (fast model)")
    if not very_high_priority_fields and not priority_fields and not low_priority_fields_list:
        print(f"ℹ️  No priority fields defined")
    
    if normal_fields:
        print(f"📊 Normal fields (single API call): {len(normal_fields)} fields")
    
    # Resolve the effective provider-mapping text for this template ONCE.
    # The source is gated by extract_providers_from_annotations (the "Extract
    # providers from PDF annotated (pasted) providers" checkbox):
    #   * checkbox ON  → use the provider_mapping DB column (has MedNet codes for
    #                    red-number annotation matching), which users keep updated.
    #   * checkbox OFF → harvest the roster from the Responsible Provider field,
    #                    which is what users maintain in that mode. A stale/leftover
    #                    provider_mapping column is NOT used when the checkbox is off.
    from field_definitions import derive_provider_mapping_for_template, get_field_definitions
    all_template_fields = get_field_definitions(excel_file_path)
    _pb_provider_mapping, _pb_has_mednet = derive_provider_mapping_for_template(
        provider_mapping, all_template_fields, extract_providers_from_annotations
    )
    if _pb_provider_mapping:
        src = 'provider_mapping column (with MedNet codes)' if _pb_has_mednet else 'Responsible Provider field (regex, no MedNet codes)'
        print(f"🧷 peripheral_blocks provider list resolved from: {src} ({len(_pb_provider_mapping)} chars)")
    else:
        print(f"ℹ️  No provider list found (no provider_mapping column and no Responsible Provider field text) — peripheral_blocks footer will be skipped")

    extraction_prompt = generate_extraction_prompt(
        excel_file_path,
        provider_mapping=_pb_provider_mapping,
        provider_mapping_has_mednet=_pb_has_mednet,
    )
    fieldnames = get_fieldnames(excel_file_path)
    
    # Save the prompt to a text file for easy copying
    with open("extraction_prompt.txt", "w", encoding="utf-8") as prompt_file:
        prompt_file.write(extraction_prompt)
    print(f"✅ Saved extraction prompt to 'extraction_prompt.txt'")
    
    # Remove page_number but keep source_file for tracking
    fieldnames = [field for field in fieldnames if field not in ['page_number']]

    # Auto-add Referring column next to Surgeon — value is copied from Surgeon
    # downstream (see merged_data['Referring'] = merged_data['Surgeon']).
    # Without this, the Referring value gets stripped by the CSV column filter.
    if 'Surgeon' in fieldnames and 'Referring' not in fieldnames:
        fieldnames.insert(fieldnames.index('Surgeon') + 1, 'Referring')
    
    # Add CSN as the first column if extract_csn is enabled
    if extract_csn:
        # Remove CSN if it exists elsewhere in the list
        if 'CSN' in fieldnames:
            fieldnames.remove('CSN')
        # Insert CSN at the first position
        fieldnames.insert(0, 'CSN')
        print(f"🔍 CSN extraction enabled - will extract CSN from PDF filenames")
    
    # Add source_file as the last column in the output
    if 'source_file' not in fieldnames:
        fieldnames.append('source_file')

    # Add service tier column
    if 'service_tier' not in fieldnames:
        fieldnames.append('service_tier')
    
    # Add provider fields to fieldnames if extract_providers_from_annotations is enabled
    # This ensures they appear in the CSV output even if not in the template
    if extract_providers_from_annotations:
        provider_fields = [
            'Responsible Provider', 'MD', 'CRNA', 'SRNA',
            'Primary Mednet Code', 'Secondary Mednet Code', 'Tertiary Mednet Code',
        ]
        for field in provider_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        print(f"📋 Provider annotation extraction enabled - added provider + insurance fields to output")
    
    # EXPERIMENT: when the main model is a self-hosted vLLM checkpoint, force every
    # priority tier to the same model so the A/B measures one model, not a mix.
    # Also prevents the Gemini branch below from rewriting `model`.
    # (Revert: delete this block.)
    if is_vllm_model(model):
        print(f"🧪 vLLM mode: forcing all priority tiers to '{model}'")
        priority_model = model
        low_priority_model = model
        very_high_priority_model = model

    # Initialize Google AI client (only needed if not using OpenRouter)
    # Check if we're using Gemini models (use Gemini API) or OpenRouter models
    using_gemini = is_gemini_model(model) or (priority_model and is_gemini_model(priority_model)) or (low_priority_model and is_gemini_model(low_priority_model))
    using_openrouter = (not using_gemini) and (is_openrouter_model(model) or (priority_model and is_openrouter_model(priority_model)) or (low_priority_model and is_openrouter_model(low_priority_model)))

    client = None
    if using_gemini:
        # Check if GOOGLE_API_KEY is available; if not, fall back to OpenRouter
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if google_api_key:
            # Normalize Gemini model names
            model = normalize_gemini_model(model)
            if priority_model:
                priority_model = normalize_gemini_model(priority_model)
            if low_priority_model:
                low_priority_model = normalize_gemini_model(low_priority_model)
            if very_high_priority_model:
                very_high_priority_model = normalize_gemini_model(very_high_priority_model)
            client = genai.Client(api_key=google_api_key)
            print(f"🔑 Using Google GenAI SDK directly for Gemini model '{model}'")
        else:
            # Fall back to OpenRouter with google/ prefix
            # Use gemini-3-pro-preview for standard tab, otherwise preserve selected model
            if normalize_gemini_model(model) == "gemini-3-pro-preview":
                print(f"⚠️ No GOOGLE_API_KEY found, routing '{model}' through OpenRouter as google/gemini-3-pro-preview")
                model = "google/gemini-3-pro-preview"
            else:
                print(f"⚠️ No GOOGLE_API_KEY found, routing '{model}' through OpenRouter")
                model = f"google/{normalize_gemini_model(model)}"
            if priority_model:
                priority_model = f"google/{normalize_gemini_model(priority_model)}"
            if low_priority_model:
                low_priority_model = f"google/{normalize_gemini_model(low_priority_model)}"
            if very_high_priority_model:
                very_high_priority_model = f"google/{normalize_gemini_model(very_high_priority_model)}"
            using_gemini = False
            using_openrouter = True

    # EXPERIMENT: vLLM needs no OpenRouter/Google key. (Revert: delete this branch.)
    if is_vllm_model(model):
        using_openrouter = False
        using_gemini = False
        print(f"🤖 Using self-hosted vLLM at {VLLM_BASE_URL} for extraction")
    elif using_openrouter:
        # Verify OpenRouter API key is available
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not openrouter_key:
            print("❌ OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set!")
            sys.exit(1)
        print("🤖 Using OpenRouter API for extraction")
    elif not client:
        api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w")
        if not api_key:
            print("❌ GOOGLE_API_KEY environment variable not set!")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    
    # Find all PDF files in the input folder (both uppercase and lowercase extensions)
    # Search recursively to handle ZIP files with folder structures
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf")) + glob.glob(os.path.join(input_folder, "*.PDF"))
    pdf_files += glob.glob(os.path.join(input_folder, "**", "*.pdf"), recursive=True)
    pdf_files += glob.glob(os.path.join(input_folder, "**", "*.PDF"), recursive=True)
    
    # Remove duplicates (in case a file is found both ways) and filter out __MACOSX metadata
    pdf_files = [f for f in set(pdf_files) if '__MACOSX' not in f]
    
    if not pdf_files:
        print(f"❌ ERROR: No PDF files found in the '{input_folder}' folder.")
        print(f"❌ Searched directory: {os.path.abspath(input_folder)}")
        print(f"❌ Directory exists: {os.path.exists(input_folder)}")
        if os.path.exists(input_folder):
            all_files = []
            for root, dirs, files in os.walk(input_folder):
                all_files.extend([os.path.join(root, f) for f in files])
            print(f"❌ Files found in directory: {all_files[:20]}")  # Show first 20 files
        sys.exit(1)  # Exit with error code so the main script can catch it
    
    # Sort PDF files to ensure consistent ordering
    pdf_files.sort()
    print(f"📁 Found {len(pdf_files)} patient PDF files to process.")
    
    # Process all PDFs concurrently
    all_extracted_data = []
    temp_files = []  # Keep track of temporary files for cleanup
    failed_pdfs = []  # Track PDFs that failed completely
    
    try:
        # Prepare tasks for all PDFs with order tracking
        tasks = []
        for order_index, pdf_file in enumerate(pdf_files):
            tasks.append((client, pdf_file, extraction_prompt, priority_fields, low_priority_fields_list, excel_file_path, n_pages, model, priority_model, low_priority_model, order_index, provider_mapping, extract_providers_from_annotations, very_high_priority_fields, very_high_priority_model, _pb_provider_mapping, _pb_has_mednet))

        import time
        start_time = time.time()
        print(f"\n🚀 Starting concurrent processing of {len(tasks)} patient PDFs...")
        print(f"  [DEBUG] Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        # Initialize progress tracking
        completed_count = 0
        total_count = len(tasks)
        
        def write_progress():
            """Write progress to file if provided"""
            if progress_file:
                try:
                    # Ensure parent directory exists
                    progress_path = Path(progress_file)
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(progress_file, 'w') as f:
                        f.write(f"{completed_count}\n{total_count}\n")
                        f.flush()  # Ensure it's written immediately
                        import os
                        os.fsync(f.fileno())  # Force write to disk
                    if completed_count % 10 == 0 or completed_count == total_count:  # Log every 10 or on completion
                        print(f"  📊 Progress: {completed_count}/{total_count} (written to {progress_file})")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not write progress file {progress_file}: {e}")
                    import traceback
                    print(f"  ⚠️  Traceback: {traceback.format_exc()}")
        
        # Write initial progress
        print(f"  📝 Initializing progress tracking (file: {progress_file})")
        write_progress()
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pdf_files))) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_single_patient_pdf_task, task): task for task in tasks}
            
            # Collect results as they complete, but store with order index for later sorting
            results_with_order = []
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                pdf_file_path = task[1]  # PDF file path from task
                order_index = task[8]    # Order index from task (still at index 8)
                pdf_filename = os.path.basename(pdf_file_path)

                elapsed = time.time() - start_time
                print(f"  [DEBUG] PDF COMPLETED: {pdf_filename} after {elapsed:.1f}s total")

                try:
                    filename, response, temp_patient_pdf, order_idx = future.result()
                    
                    if temp_patient_pdf:
                        temp_files.append(temp_patient_pdf)
                    
                    if response:
                        try:
                            # Clean the response by removing markdown code block formatting
                            cleaned_response = response.strip()
                            if cleaned_response.startswith('```json'):
                                cleaned_response = cleaned_response[7:]  # Remove ```json
                            if cleaned_response.startswith('```'):
                                cleaned_response = cleaned_response[3:]   # Remove ```
                            if cleaned_response.endswith('```'):
                                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
                            cleaned_response = cleaned_response.strip()
                            
                            # Parse the JSON response
                            extracted_record = json.loads(cleaned_response)
                            
                            # Clean and format all field values
                            for field_name, value in extracted_record.items():
                                if value:
                                    # First clean the value (removes ?, invisible chars, etc.)
                                    cleaned_value = clean_field_value(value)
                                    
                                    # Then apply specific formatting for phone numbers
                                    if 'phone' in field_name.lower():
                                        cleaned_value = format_phone_number(cleaned_value)
                                    
                                    extracted_record[field_name] = cleaned_value
                            
                            # Extract CSN from filename if enabled
                            if extract_csn:
                                csn = extract_csn_from_filename(pdf_filename)
                                if csn:
                                    extracted_record['CSN'] = csn
                                    print(f"  🔍 Extracted CSN: {csn} from {pdf_filename}")
                                else:
                                    extracted_record['CSN'] = None
                                    print(f"  ⚠️  Could not extract CSN from {pdf_filename}")
                            
                            # Add source file info for reference
                            extracted_record['source_file'] = pdf_filename
                            
                            # Store result with order index for later sorting
                            results_with_order.append((order_idx, extracted_record))
                            print(f"  ✅ Successfully added data for {pdf_filename}")
                            
                        except json.JSONDecodeError as e:
                            print(f"  ❌ JSON parsing error for {pdf_filename}: {str(e)}")
                            failed_pdfs.append(pdf_filename)
                    else:
                        print(f"  ❌ All retries failed for {pdf_filename}")
                        failed_pdfs.append(pdf_filename)
                    
                    # Update progress after each PDF is processed (success or failure)
                    completed_count += 1
                    write_progress()
                        
                except Exception as e:
                    print(f"  ❌ Exception processing {pdf_filename}: {str(e)}")
                    failed_pdfs.append(pdf_filename)
                    # Update progress even on exception
                    completed_count += 1
                    write_progress()
            
            # Sort results by original order to preserve PDF order
            results_with_order.sort(key=lambda x: x[0])
            all_extracted_data = [result[1] for result in results_with_order]
        
        # Summary of processing
        success_count = len(all_extracted_data)
        fail_count = len(failed_pdfs)
        
        if fail_count > 0:
            print(f"\n⚠️  Successfully processed {success_count} PDFs, {fail_count} PDFs failed after retries")
            print(f"   Failed PDFs: {sorted(failed_pdfs)}")
        else:
            print(f"\n🎉 Successfully processed all {success_count} patient PDFs")
        
        # Create the combined CSV file
        if all_extracted_data:
            # Filter extracted data to only include expected fields (exclude source_file from final output)
            filtered_data = []
            for record in all_extracted_data:
                filtered_record = {}
                for field in fieldnames:
                    value = record.get(field, None)
                    # Ensure ID fields and numeric-looking strings stay as strings
                    if value is not None and isinstance(value, (str, int, float)):
                        value = str(value)
                        # Clean the value one more time (removes ?, invisible chars, etc.)
                        value = clean_field_value(value)
                        
                    filtered_record[field] = value
                filtered_data.append(filtered_record)
            
            # Add worktracker columns if provided
            if worktracker_group:
                for record in filtered_data:
                    record['Worktracker Group'] = worktracker_group
                if 'Worktracker Group' not in fieldnames:
                    fieldnames.append('Worktracker Group')
            
            if worktracker_batch:
                for record in filtered_data:
                    record['Worktracker Batch #'] = worktracker_batch
                if 'Worktracker Batch #' not in fieldnames:
                    fieldnames.append('Worktracker Batch #')

            if scanned_date:
                for record in filtered_data:
                    record['Scanned Date'] = scanned_date
                if 'Scanned Date' not in fieldnames:
                    fieldnames.append('Scanned Date')

            # Save to both CSV and Excel formats
            extracted_folder = "extracted"
            os.makedirs(extracted_folder, exist_ok=True)
            
            # Create combined filenames with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_csv_filename = f"combined_patient_data_{timestamp}.csv"
            combined_excel_filename = f"combined_patient_data_{timestamp}.xlsx"
            extracted_csv_path = os.path.join(extracted_folder, combined_csv_filename)
            extracted_excel_path = os.path.join(extracted_folder, combined_excel_filename)
            
            # CSV output (clean data for medical billing apps)
            with open(extracted_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_data)
            
            # Excel output (preserves data types, no scientific notation)
            df = pd.DataFrame(filtered_data)
            
            # Replace None values with empty strings for cleaner Excel display
            df = df.fillna('')
            
            # Replace 'None' strings with empty strings (in case any slipped through)
            df = df.replace('None', '')
            
            # Explicitly set ID columns as text to prevent scientific notation
            id_columns = ['Primary Subsc ID', 'Secondary Subsc ID', 'MRN', 'CSN']
            for col in id_columns:
                if col in df.columns:
                    # Only convert non-empty values to string to avoid 'nan' text
                    df[col] = df[col].apply(lambda x: str(x) if x != '' else '')
            
            df.to_excel(extracted_excel_path, index=False, engine='openpyxl')
            
            print(f"📊 Created {combined_csv_filename} with {len(filtered_data)} patient records (clean CSV for imports)")
            print(f"   CSV saved to: {extracted_csv_path}")
            print(f"📊 Created {combined_excel_filename} with {len(filtered_data)} patient records (Excel format, no scientific notation)")
            print(f"   Excel saved to: {extracted_excel_path}")
        else:
            print(f"❌ ERROR: No data extracted from any PDF files")
            print(f"❌ Total PDFs processed: {len(pdf_files)}")
            print(f"❌ Successful extractions: {success_count}")
            print(f"❌ Failed extractions: {fail_count}")
            sys.exit(1)  # Exit with error code
                
    except Exception as e:
        print(f"❌ FATAL ERROR during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Exit with error code
    
    finally:
        # Clean up temporary files
        print(f"🧹 Cleaning up {len(temp_files)} temporary files...")
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    print(f"\n✅ Processing complete!")


if __name__ == "__main__":
    # Allow specifying input folder, Excel file, number of pages, max workers, model, and worktracker fields as command line arguments
    input_folder = "input"  # Default input folder
    excel_file = "WPA for testing FINAL.xlsx"  # Default Excel file
    n_pages = 2  # Default number of pages to extract per patient
    max_workers = 50  # Default thread pool size
    model = "gemini-3-flash-preview"  # Default model for normal fields
    priority_model = "gemini-3-flash-preview"  # Default model for high-priority fields
    low_priority_model = "google/gemini-3.1-flash-lite-preview"  # Default model for low-priority fields
    very_high_priority_model = "gemini-3.1-pro-preview"  # Default model for very-high-priority fields
    worktracker_group = None  # Optional worktracker group
    worktracker_batch = None  # Optional worktracker batch
    extract_csn = False  # Extract CSN from PDF filenames
    provider_mapping = None  # Optional provider mapping text
    extract_providers_from_annotations = False  # Extract providers from PDF annotations
    scanned_date = None  # Optional scanned date (RIV only)
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        excel_file = sys.argv[2]
    if len(sys.argv) > 3:
        try:
            n_pages = int(sys.argv[3])
        except ValueError:
            print("⚠️  Warning: Invalid n_pages value, using default of 2")
    if len(sys.argv) > 4:
        try:
            max_workers = int(sys.argv[4])
        except ValueError:
            print("⚠️  Warning: Invalid max_workers value, using default of 50")
    if len(sys.argv) > 5:
        model = sys.argv[5]
    if len(sys.argv) > 6:
        worktracker_group = sys.argv[6] if sys.argv[6].strip() else None
    if len(sys.argv) > 7:
        worktracker_batch = sys.argv[7] if sys.argv[7].strip() else None
    if len(sys.argv) > 8:
        extract_csn = sys.argv[8].lower() == "true" if sys.argv[8].strip() else False
    progress_file = None  # Optional progress file path
    if len(sys.argv) > 9:
        progress_file = sys.argv[9] if sys.argv[9].strip() else None
    if len(sys.argv) > 10:
        provider_mapping = sys.argv[10] if sys.argv[10].strip() else None
    if len(sys.argv) > 11:
        extract_providers_from_annotations = sys.argv[11].lower() == "true" if sys.argv[11].strip() else False
    if len(sys.argv) > 12:
        scanned_date = sys.argv[12] if sys.argv[12].strip() else None
    
    print(f"🔧 Configuration:")
    print(f"   Input folder: {input_folder}")
    print(f"   Excel file: {excel_file}")
    print(f"   Pages per patient: {n_pages}")
    print(f"   Max workers: {max_workers}")
    print(f"   Model (normal fields): {model}")
    print(f"   Model (high-priority fields): {priority_model}")
    print(f"   Model (low-priority fields): {low_priority_model}")
    print(f"   Model (very-high-priority fields): {very_high_priority_model}")
    if worktracker_group:
        print(f"   Worktracker Group: {worktracker_group}")
    if worktracker_batch:
        print(f"   Worktracker Batch #: {worktracker_batch}")
    if extract_csn:
        print(f"   Extract CSN: Enabled")
    if progress_file:
        print(f"   Progress file: {progress_file}")
    if extract_providers_from_annotations:
        print(f"   Extract Providers from Annotations: Enabled")
        if provider_mapping:
            print(f"   Provider Mapping: Loaded ({len(provider_mapping)} characters)")
    if scanned_date:
        print(f"   Scanned Date: {scanned_date}")
    print()
    
    process_all_patient_pdfs(input_folder, excel_file, n_pages, max_workers, model, priority_model, low_priority_model, very_high_priority_model, worktracker_group, worktracker_batch, extract_csn, progress_file, provider_mapping, extract_providers_from_annotations, scanned_date)