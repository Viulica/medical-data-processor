import os
import json
import csv
import glob
import tempfile
import sys
import time
import random
import pandas as pd
import google.genai as genai
from google.genai import types
from PyPDF2 import PdfReader, PdfWriter
from field_definitions import get_fieldnames, generate_extraction_prompt, get_priority_fields, get_normal_fields, generate_priority_field_prompt
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

def pdf_to_images_base64(pdf_path, max_pages=10):
    """Convert PDF pages to base64 encoded images for OpenRouter"""
    try:
        # Use PyMuPDF (fitz) which is already in requirements
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        image_data_list = []
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
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

def extract_with_openrouter(patient_pdf_path, pdf_filename, extraction_prompt, model, max_retries=5, field_name_for_log=None):
    """Extract patient information using OpenRouter API"""
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
    
    # Build content for OpenRouter
    content = [{"type": "text", "text": extraction_prompt}]
    for img_data in image_data_list:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_data}"
            }
        })
    
    messages = [{"role": "user", "content": content}]
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/medical-data-processor",
        "X-Title": "Medical Data Processor"
    }
    
    # Ensure DeepSeek model uses exact format
    openrouter_model = model
    if "deepseek" in model.lower():
        openrouter_model = "deepseek/deepseek-v3.2"
    
    for attempt in range(max_retries):
        try:
            payload = {
                "model": openrouter_model,
                "messages": messages
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
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
            return response_text
            
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parsing failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"    ❌ Final JSON parsing failure for {pdf_filename}{log_suffix}")
                return None
        except Exception as e:
            print(f"    ⚠️  OpenRouter API call failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"    ❌ Final OpenRouter API failure for {pdf_filename}{log_suffix}")
                return None
        
        # Exponential backoff
        if attempt < max_retries - 1:
            base_delay = 2 ** attempt
            jitter = random.uniform(0.5, 1.5)
            delay = base_delay * jitter
            print(f"    ⏳ Retrying {pdf_filename}{log_suffix} in {delay:.1f} seconds...")
            time.sleep(delay)
    
    return None

def extract_info_from_patient_pdf(client, patient_pdf_path, pdf_filename, extraction_prompt, model="gemini-3-pro-preview", max_retries=5, field_name_for_log=None):
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
    
    # Check if using OpenRouter (only if not a Gemini model)
    if not is_gemini_model(model) and is_openrouter_model(model):
        return extract_with_openrouter(patient_pdf_path, pdf_filename, extraction_prompt, model, max_retries, field_name_for_log)
    
    # Normalize Gemini model name if needed
    if is_gemini_model(model):
        model = normalize_gemini_model(model)
    
    log_suffix = f" - {field_name_for_log}" if field_name_for_log else ""
    
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
            
            tools = [
                types.Tool(googleSearch=types.GoogleSearch(
                )),
            ]
            
            # Use thinking_level="HIGH" for Gemini 3 models, thinking_budget=-1 for others
            if model in ["gemini-3-pro-preview", "gemini-3-flash-preview"]:
                thinking_config = types.ThinkingConfig(
                    thinking_level="HIGH",
                )
            else:
                thinking_config = types.ThinkingConfig(
                    thinking_budget=-1,
                )
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                thinking_config=thinking_config,
                tools=tools
            )

            # Collect the full response with retry on API failures
            full_response = ""
            try:
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text is not None:
                        full_response += chunk.text
                
                response_text = full_response.strip()
                
                # Validate that we got a meaningful response
                if not response_text or len(response_text) < 10:
                    raise ValueError(f"Response too short or empty: {response_text}")
                
                # Try to parse JSON to validate response format
                cleaned_response = response_text
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]   # Remove ```
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove trailing ```
                cleaned_response = cleaned_response.strip()
                
                # Parse JSON to validate format (this will raise JSONDecodeError if invalid)
                json.loads(cleaned_response)
                
                # If we get here, everything worked
                print(f"    ✅ Successfully processed {pdf_filename}{log_suffix} on attempt {attempt + 1}")
                return response_text
                
            except json.JSONDecodeError as e:
                print(f"    ⚠️  JSON parsing failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    print(f"    ❌ Final JSON parsing failure for {pdf_filename}{log_suffix}")
                    print(f"    Raw response: {response_text[:200]}...")
                    return None
                # Continue to retry logic below
                
            except Exception as api_error:
                print(f"    ⚠️  API call failed for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(api_error)}")
                if attempt == max_retries - 1:
                    print(f"    ❌ Final API failure for {pdf_filename}{log_suffix}")
                    return None
                # Continue to retry logic below
        
        except Exception as e:
            print(f"    ⚠️  Unexpected error for {pdf_filename}{log_suffix} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"    ❌ Final failure for {pdf_filename}{log_suffix}")
                return None
        
        # Exponential backoff with jitter for retries
        if attempt < max_retries - 1:
            base_delay = 2 ** attempt  # 1, 2, 4, 8 seconds
            jitter = random.uniform(0.5, 1.5)  # Add randomness to prevent thundering herd
            delay = base_delay * jitter
            print(f"    ⏳ Retrying {pdf_filename}{log_suffix} in {delay:.1f} seconds...")
            time.sleep(delay)
    
    return None


def process_single_patient_pdf_task(args):
    """Task function for processing a single patient PDF in a thread."""
    client, pdf_file_path, extraction_prompt, priority_fields, excel_file_path, n_pages, model, priority_model, order_index = args
    
    pdf_filename = os.path.basename(pdf_file_path)
    
    # Extract first n pages as temporary PDF
    temp_patient_pdf = extract_first_n_pages_as_pdf(pdf_file_path, n_pages)
    if not temp_patient_pdf:
        return pdf_filename, None, temp_patient_pdf, order_index
    
    # Extract normal (non-priority) fields with one API call
    # Pass client (may be None for OpenRouter)
    normal_response = extract_info_from_patient_pdf(client, temp_patient_pdf, pdf_filename, extraction_prompt, model)
    
    if not normal_response:
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
    
    # Extract each priority field separately and merge into the result
    if priority_fields:
        print(f"    🎯 Processing {len(priority_fields)} priority field(s) for {pdf_filename}")
        
        for priority_field in priority_fields:
            field_name = priority_field['name']
            priority_prompt = generate_priority_field_prompt(priority_field)
            
            # Extract this priority field using the better model
            # Pass client (may be None for OpenRouter)
            priority_response = extract_info_from_patient_pdf(
                client, temp_patient_pdf, pdf_filename, priority_prompt, priority_model, 
                field_name_for_log=field_name
            )
            
            if priority_response:
                try:
                    # Parse the priority field response
                    cleaned_priority = priority_response.strip()
                    if cleaned_priority.startswith('```json'):
                        cleaned_priority = cleaned_priority[7:]
                    if cleaned_priority.startswith('```'):
                        cleaned_priority = cleaned_priority[3:]
                    if cleaned_priority.endswith('```'):
                        cleaned_priority = cleaned_priority[:-3]
                    cleaned_priority = cleaned_priority.strip()
                    
                    priority_data = json.loads(cleaned_priority)
                    
                    # Merge this priority field into the main result
                    if field_name in priority_data:
                        merged_data[field_name] = priority_data[field_name]
                        print(f"    ✅ Merged priority field '{field_name}' for {pdf_filename}")
                    else:
                        print(f"    ⚠️  Priority field '{field_name}' not found in response for {pdf_filename}")
                        
                except json.JSONDecodeError as e:
                    print(f"    ❌ Failed to parse priority field '{field_name}' for {pdf_filename}: {str(e)}")
            else:
                print(f"    ❌ Failed to extract priority field '{field_name}' for {pdf_filename}")
    
    # Convert merged data back to JSON string for compatibility with existing code
    merged_response = json.dumps(merged_data)
    
    return pdf_filename, merged_response, temp_patient_pdf, order_index


def process_all_patient_pdfs(input_folder="input", excel_file_path="WPA for testing FINAL.xlsx", n_pages=2, max_workers=50, model="gemini-3-pro-preview", priority_model="gemini-3-pro-preview", worktracker_group=None, worktracker_batch=None, extract_csn=False, progress_file=None):
    """Process all patient PDFs in the input folder, combining first n pages per patient into one CSV."""
    
    print(f"🚀 process_all_patient_pdfs called with progress_file={progress_file}")
    
    # Check if Excel file exists
    if not os.path.exists(excel_file_path):
        print(f"❌ Error: Excel file '{excel_file_path}' not found!")
        return
    
    print(f"📋 Using field definitions from: {excel_file_path}")
    print(f"📄 Processing first {n_pages} pages per patient PDF")
    print(f"🧵 Max concurrent threads: {max_workers}")
    
    # Get priority and normal fields
    priority_fields = get_priority_fields(excel_file_path)
    normal_fields = get_normal_fields(excel_file_path)
    
    if priority_fields:
        priority_field_names = [f['name'] for f in priority_fields]
        print(f"🎯 Priority fields (separate API calls): {', '.join(priority_field_names)}")
        print(f"🤖 Using model '{priority_model}' for priority fields (better accuracy)")
    else:
        print(f"ℹ️  No priority fields defined")
    
    if normal_fields:
        print(f"📊 Normal fields (single API call): {len(normal_fields)} fields")
    
    # Generate extraction prompt from Excel file (only for normal fields)
    extraction_prompt = generate_extraction_prompt(excel_file_path)
    fieldnames = get_fieldnames(excel_file_path)
    
    # Save the prompt to a text file for easy copying
    with open("extraction_prompt.txt", "w", encoding="utf-8") as prompt_file:
        prompt_file.write(extraction_prompt)
    print(f"✅ Saved extraction prompt to 'extraction_prompt.txt'")
    
    # Remove page_number but keep source_file for tracking
    fieldnames = [field for field in fieldnames if field not in ['page_number']]
    
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
    
    # Initialize Google AI client (only needed if not using OpenRouter)
    # Check if we're using Gemini models (use Gemini API) or OpenRouter models
    using_gemini = is_gemini_model(model) or is_gemini_model(priority_model)
    using_openrouter = (not using_gemini) and (is_openrouter_model(model) or is_openrouter_model(priority_model))
    
    client = None
    if using_gemini:
        # Normalize Gemini model names
        model = normalize_gemini_model(model)
        if priority_model:
            priority_model = normalize_gemini_model(priority_model)
    
    if not using_openrouter:
        api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w")
        if not api_key:
            print("❌ GOOGLE_API_KEY environment variable not set!")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    else:
        # Verify OpenRouter API key is available
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not openrouter_key:
            print("❌ OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set!")
            sys.exit(1)
        print("🤖 Using OpenRouter API for extraction")
    
    # Find all PDF files in the input folder (both uppercase and lowercase extensions)
    # Search recursively to handle ZIP files with folder structures
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf")) + glob.glob(os.path.join(input_folder, "*.PDF"))
    pdf_files += glob.glob(os.path.join(input_folder, "**", "*.pdf"), recursive=True)
    pdf_files += glob.glob(os.path.join(input_folder, "**", "*.PDF"), recursive=True)
    
    # Remove duplicates (in case a file is found both ways)
    pdf_files = list(set(pdf_files))
    
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
            tasks.append((client, pdf_file, extraction_prompt, priority_fields, excel_file_path, n_pages, model, priority_model, order_index))
        
        print(f"\n🚀 Starting concurrent processing of {len(tasks)} patient PDFs...")
        
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
                order_index = task[8]    # Order index from task
                pdf_filename = os.path.basename(pdf_file_path)
                
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
    model = "gemini-3-pro-preview"  # Default model for normal fields
    priority_model = "gemini-3-pro-preview"  # Default model for priority fields (always use best model)
    worktracker_group = None  # Optional worktracker group
    worktracker_batch = None  # Optional worktracker batch
    extract_csn = False  # Extract CSN from PDF filenames
    
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
    
    print(f"🔧 Configuration:")
    print(f"   Input folder: {input_folder}")
    print(f"   Excel file: {excel_file}")
    print(f"   Pages per patient: {n_pages}")
    print(f"   Max workers: {max_workers}")
    print(f"   Model (normal fields): {model}")
    print(f"   Model (priority fields): {priority_model}")
    if worktracker_group:
        print(f"   Worktracker Group: {worktracker_group}")
    if worktracker_batch:
        print(f"   Worktracker Batch #: {worktracker_batch}")
    if extract_csn:
        print(f"   Extract CSN: Enabled")
    if progress_file:
        print(f"   Progress file: {progress_file}")
    print()
    
    process_all_patient_pdfs(input_folder, excel_file, n_pages, max_workers, model, priority_model, worktracker_group, worktracker_batch, extract_csn, progress_file) 