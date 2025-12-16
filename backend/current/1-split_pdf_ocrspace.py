#!/usr/bin/env python3
"""
PDF Splitting using OCR.space API
==================================
Super simple OCR API - just like 'resend' for emails but for OCR.
Get your free API key at: https://ocr.space/ocrapi

Free tier: 25,000 requests/month
No complex setup required!
"""

import os
import sys
import json
import time
import requests
import fitz  # PyMuPDF for high-quality PDF to image conversion
from PIL import Image, ImageEnhance
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore


class OCRSpaceAPI:
    """Simple wrapper for OCR.space API with connection pooling for speed"""
    
    # Semaphore to limit concurrent API requests (API allows max 10 concurrent)
    # Using 7 for optimal speed while staying under API limit
    _concurrency_semaphore = Semaphore(7)
    
    def __init__(self, api_key=None):
        # Get API key from environment variable (same approach as Google Gemini)
        self.api_key = api_key or os.environ.get("OCRSPACE_API_KEY")
        if not self.api_key:
            raise ValueError("OCRSPACE_API_KEY environment variable not set!")
        
        # Debug: Show API key info (masked for security)
        key_preview = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        print(f"🔑 Using API key: {key_preview} (length: {len(self.api_key)})")
        
        # Use PRO tier endpoint (free tier is api.ocr.space)
        self.base_url = "https://apipro1.ocr.space/parse/image"
        print(f"🌐 OCR endpoint: {self.base_url}")
        print(f"🔒 Max concurrent requests: 2 (API limit: 10)")
        
        # Use session for connection pooling (much faster!)
        self.session = requests.Session()
        # Reuse connections - keep pool size very low to respect API limits
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,   # Single connection pool
            pool_maxsize=2,       # Max 2 connections (well under API limit of 10)
            max_retries=3
        )
        self.session.mount('https://', adapter)
    
    def pdf_to_enhanced_image(self, pdf_bytes, dpi=200):
        """
        Convert PDF page to enhanced image for better OCR accuracy.
        Matches local OCR preprocessing pipeline.
        
        Args:
            pdf_bytes: PDF file bytes
            dpi: Resolution (default: 200 - higher than local's 150 for better quality)
            
        Returns:
            PNG image bytes, or None on error
        """
        try:
            # Open PDF with PyMuPDF (same as local OCR)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc.load_page(0)  # Single page PDF
            
            # Convert to image at high DPI (same as local OCR method)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image for preprocessing
            img_data = pix.tobytes("png")
            image = Image.open(BytesIO(img_data))
            doc.close()
            
            # Image enhancement pipeline (improves OCR accuracy)
            # Convert to grayscale for better text detection
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast (makes text sharper)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # 50% more contrast
            
            # Enhance sharpness (better character recognition)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)  # 2x sharpness
            
            # Convert back to PNG bytes
            output = BytesIO()
            image.save(output, format='PNG', optimize=True)
            return output.getvalue()
            
        except Exception as e:
            print(f"  ⚠️  Image preprocessing error: {str(e)}")
            return None
    
    def ocr_pdf_page(self, pdf_bytes, language='eng', timeout=30, fast_mode=True):
        """
        Extract text from a PDF page with enhanced preprocessing.
        
        Args:
            pdf_bytes: PDF file bytes
            language: OCR language (default: 'eng')
            timeout: Request timeout in seconds (reduced for speed)
            fast_mode: Use faster OCR engine (True) or more accurate (False)
            
        Returns:
            Extracted text as string, or None on error
        """
        # Acquire semaphore to limit concurrent requests (respects API limit of 10)
        # This ensures we never exceed 2 concurrent requests
        with self._concurrency_semaphore:
            # Small delay to ensure connections are properly managed
            time.sleep(0.1)
            try:
                # ENHANCEMENT: Convert PDF to preprocessed image for better accuracy
                # This matches the local OCR workflow (PDF -> Image -> OCR)
                print(f"  🖼️  Preprocessing PDF page (converting to enhanced 200 DPI image)...")
                image_bytes = self.pdf_to_enhanced_image(pdf_bytes, dpi=200)
                
                if not image_bytes:
                    print(f"  ⚠️  Image preprocessing failed, falling back to raw PDF")
                    image_bytes = pdf_bytes
                    file_type = 'application/pdf'
                    file_name = 'page.pdf'
                else:
                    file_type = 'image/png'
                    file_name = 'page.png'
                    print(f"  ✅ Preprocessed image ready ({len(image_bytes)} bytes)")
                
                # Prepare the request
                payload = {
                    'apikey': self.api_key,
                    'language': language,
                    'isOverlayRequired': False,
                    'detectOrientation': True,  # Enable for accuracy (matches local OCR)
                    'scale': True,  # Enable for better quality
                    'OCREngine': 2 if not fast_mode else 1,  # Engine 2 = more accurate (matches local Tesseract OEM 3)
                }
                
                # Debug: Show payload (mask API key)
                debug_payload = payload.copy()
                debug_payload['apikey'] = f"{payload['apikey'][:8]}...{payload['apikey'][-4:]}" if len(payload['apikey']) > 12 else "***"
                print(f"  📤 Request payload: {debug_payload}")
                print(f"  📎 File size: {len(image_bytes)} bytes ({file_name})")
                
                files = {
                    'file': (file_name, image_bytes, file_type)
                }
                
                # Make API request with session (reuses connections)
                response = self.session.post(
                    self.base_url,
                    data=payload,
                    files=files,
                    timeout=timeout
                )
                
                # Debug logging
                print(f"  📡 Response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"  ❌ HTTP Error {response.status_code}: {response.reason}")
                    print(f"  📄 Response body: {response.text[:500]}")
                    response.raise_for_status()
                
                result = response.json()
                
                # Check for errors in the API response
                if result.get('IsErroredOnProcessing'):
                    error_msg = result.get('ErrorMessage', ['Unknown error'])[0]
                    print(f"  ⚠️  OCR error: {error_msg}")
                    return None
                
                # Check for OCR exit status
                if 'OCRExitCode' in result and result['OCRExitCode'] != 1:
                    print(f"  ⚠️  OCR Exit Code: {result['OCRExitCode']}")
                    print(f"  📄 Full response: {json.dumps(result, indent=2)[:500]}")
                    return None
                
                # Extract text
                if result.get('ParsedResults'):
                    text = result['ParsedResults'][0].get('ParsedText', '')
                    text = text.strip()
                    
                    # Log first few lines of extracted text for debugging
                    if text:
                        lines = text.split('\n')[:5]  # First 5 lines
                        preview = '\n'.join(lines)
                        print(f"  📝 Extracted text preview ({len(text)} chars, {len(text.split())} words):")
                        print(f"     {preview[:200]}..." if len(preview) > 200 else f"     {preview}")
                    else:
                        print(f"  ⚠️  No text extracted from page")
                    
                    return text
                
                return None
                
            except requests.exceptions.Timeout:
                print(f"  ⚠️  OCR request timeout")
                return None
            except requests.exceptions.HTTPError as e:
                print(f"  ❌ HTTP Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"  📄 Response body: {e.response.text[:1000]}")
                return None
            except Exception as e:
                print(f"  ⚠️  OCR error: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"  🔍 Traceback: {traceback.format_exc()[:500]}")
                return None
    
    def __del__(self):
        """Close session when done"""
        if hasattr(self, 'session'):
            self.session.close()


def extract_single_page_pdf(reader, page_num):
    """Extract a single page from PDF as bytes (optimized - reuses reader)"""
    writer = PdfWriter()
    writer.add_page(reader.pages[page_num])
    
    from io import BytesIO
    pdf_bytes = BytesIO()
    writer.write(pdf_bytes)
    return pdf_bytes.getvalue()


def check_page_matches(reader, page_num, filter_strings, ocr_api, case_sensitive=False):
    """
    Check if a page contains ALL the filter strings.
    
    Returns:
        tuple: (page_num, matches: bool)
    """
    try:
        # Extract single page as PDF bytes (reuses reader - faster!)
        page_pdf_bytes = extract_single_page_pdf(reader, page_num)
        
        # OCR the page (using accurate mode to match local OCR quality)
        print(f"  🔍 Processing page {page_num + 1}...")
        text = ocr_api.ocr_pdf_page(page_pdf_bytes, fast_mode=False, timeout=30)
        
        if not text:
            print(f"  ⚠️  Page {page_num + 1}: No text extracted")
            return page_num, False
        
        # Log page text summary
        text_lines = text.split('\n')
        print(f"  📄 Page {page_num + 1}: Extracted {len(text)} chars, {len(text_lines)} lines")
        if text_lines:
            first_lines = '\n'.join(text_lines[:3])  # First 3 lines
            print(f"  📝 First lines:\n     {first_lines[:150]}..." if len(first_lines) > 150 else f"  📝 First lines:\n     {first_lines}")
        
        # Check if ALL filter strings are present
        if not case_sensitive:
            text_lower = text.lower()
            matches = all(filter_str.lower() in text_lower for filter_str in filter_strings)
        else:
            matches = all(filter_str in text for filter_str in filter_strings)
        
        if matches:
            print(f"  ✅ Page {page_num + 1} matches!")
        
        return page_num, matches
        
    except Exception as e:
        print(f"  ❌ Error processing page {page_num + 1}: {str(e)}")
        return page_num, False


def find_matching_pages(pdf_path, filter_strings, max_workers=7, case_sensitive=False, progress_callback=None):
    """
    Find all pages that contain ALL the filter strings.
    
    Args:
        pdf_path: Path to PDF file
        filter_strings: List of strings that ALL must be present
        max_workers: Number of parallel workers (default: 7 - matches API semaphore)
        case_sensitive: Whether search is case-sensitive
        progress_callback: Optional callback function(completed, total, message) for progress updates
        
    Returns:
        List of matching page numbers (0-based)
    """
    # Initialize OCR API (with connection pooling)
    ocr_api = OCRSpaceAPI()
    
    # Get total pages (read once, reuse for all pages)
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    filter_display = " AND ".join([f'"{s}"' for s in filter_strings])
    print(f"📄 Scanning {total_pages} pages for: {filter_display}")
    print(f"🔍 Using OCR.space API (Enhanced Mode)")
    print(f"🧵 Processing with {max_workers} parallel workers")
    print()
    
    # Process pages in parallel
    matching_pages = []
    completed_count = 0
    
    # Thread-safe progress tracking
    import threading
    progress_lock = threading.Lock()
    
    def update_progress():
        nonlocal completed_count
        with progress_lock:
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total_pages, f"Processing page {completed_count}/{total_pages}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks (reuse reader for speed)
        futures = {
            executor.submit(check_page_matches, reader, page_num, filter_strings, ocr_api, case_sensitive): page_num
            for page_num in range(total_pages)
        }
        
        # Collect results
        for future in as_completed(futures):
            page_num, matches = future.result()
            update_progress()  # Update progress after each page completes
            if matches:
                matching_pages.append(page_num)
    
    # Sort pages
    matching_pages.sort()
    
    print(f"\n✅ Scan complete! Found {len(matching_pages)} matching pages")
    if matching_pages:
        display_pages = [p + 1 for p in matching_pages]
        print(f"   Matching pages: {display_pages}")
    
    return matching_pages


def create_pdf_sections(input_pdf_path, output_folder, detection_pages, total_pages):
    """Create separate PDF files for each detected section."""
    try:
        reader = PdfReader(input_pdf_path)
        base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
        
        if not detection_pages:
            print("  ℹ️  No detections found - no PDFs created")
            return 0
        
        created_pdfs = 0
        
        # Create PDF sections
        for i, start_page in enumerate(detection_pages):
            # Determine end page (exclusive)
            if i + 1 < len(detection_pages):
                end_page = detection_pages[i + 1]
            else:
                end_page = total_pages
            
            # Create PDF for this section
            writer = PdfWriter()
            
            print(f"  📝 Creating section {i + 1}: pages {start_page + 1}-{end_page}")
            
            # Add pages to this section
            for page_idx in range(start_page, end_page):
                writer.add_page(reader.pages[page_idx])
            
            # Save section PDF
            section_filename = f"{base_name}_section_{i + 1:02d}_pages_{start_page + 1}-{end_page}.pdf"
            section_path = os.path.join(output_folder, section_filename)
            
            with open(section_path, 'wb') as output_file:
                writer.write(output_file)
            
            created_pdfs += 1
        
        return created_pdfs
        
    except Exception as e:
        print(f"  ❌ Error creating PDF sections: {str(e)}")
        return 0


def split_pdf_with_ocrspace(input_pdf_path, output_folder, filter_strings, max_workers=7, case_sensitive=False, progress_callback=None):
    """
    Main function to split a PDF using OCR.space API.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_folder: Path to the output folder for split PDFs
        filter_strings: List of strings that ALL must be present on a page
        max_workers: Number of parallel workers (default: 7 - matches API semaphore)
        case_sensitive: Whether search is case-sensitive
        progress_callback: Optional callback function(completed, total, message) for progress updates
        
    Returns:
        Number of sections created, or None if there was an error
    """
    
    print("=" * 70)
    print("PDF Splitting with OCR.space API")
    print("=" * 70)
    print(f"Input PDF: {input_pdf_path}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find matching pages
    detection_pages = find_matching_pages(input_pdf_path, filter_strings, max_workers, case_sensitive, progress_callback)
    
    if not detection_pages:
        print("\n⚠️  No matching pages found - no sections to create")
        return 0
    
    # Get total pages for section creation
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    
    # Create sections
    print(f"\n📑 Creating {len(detection_pages)} sections...")
    created_count = create_pdf_sections(input_pdf_path, output_folder, detection_pages, total_pages)
    
    if created_count > 0:
        print(f"\n✅ Successfully created {created_count} section PDFs")
    else:
        print("\n❌ Failed to create PDF sections")
    
    return created_count


def main():
    """Command-line interface"""
    
    # Default configuration
    INPUT_PDF = "input.pdf"
    OUTPUT_FOLDER = "output"
    FILTER_STRINGS = ["Patient Address"]
    MAX_WORKERS = 3  # Reduced to 3 to respect API's 10 concurrent connection limit
    CASE_SENSITIVE = False
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        INPUT_PDF = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FOLDER = sys.argv[2]
    if len(sys.argv) > 3:
        filter_arg = sys.argv[3]
        if ',' in filter_arg:
            FILTER_STRINGS = [s.strip() for s in filter_arg.split(',')]
        else:
            FILTER_STRINGS = [filter_arg.strip()]
    if len(sys.argv) > 4:
        try:
            MAX_WORKERS = int(sys.argv[4])
        except ValueError:
            print("⚠️  Warning: Invalid max_workers, using default of 5")
    
    # Validate input
    if not os.path.exists(INPUT_PDF):
        print(f"❌ Error: Input PDF '{INPUT_PDF}' not found!")
        sys.exit(1)
    
    if not FILTER_STRINGS or all(not s.strip() for s in FILTER_STRINGS):
        print("❌ Error: Filter strings cannot be empty!")
        sys.exit(1)
    
    # Run the splitting
    result = split_pdf_with_ocrspace(INPUT_PDF, OUTPUT_FOLDER, FILTER_STRINGS, MAX_WORKERS, CASE_SENSITIVE)
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
