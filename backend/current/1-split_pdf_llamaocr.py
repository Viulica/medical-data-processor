#!/usr/bin/env python3
"""
PDF Splitting using LlamaOCR (LlamaParse)
==========================================
Even simpler than OCR.space - specifically designed for documents!
Get your free API key at: https://cloud.llamaindex.ai/api-key

Free tier: 1000 pages/day
Perfect for medical documents!
"""

import os
import sys
import time
from PyPDF2 import PdfReader, PdfWriter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from llama_parse import LlamaParse
except ImportError:
    print("❌ llama_parse not installed!")
    print("Install with: pip install llama-parse")
    sys.exit(1)


def extract_single_page_pdf(pdf_path, page_num):
    """Extract a single page from PDF and save as temporary file"""
    import tempfile
    
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_num])
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    writer.write(temp_file)
    temp_file.close()
    
    return temp_file.name


def check_page_matches(pdf_path, page_num, filter_strings, parser, case_sensitive=False):
    """
    Check if a page contains ALL the filter strings.
    
    Returns:
        tuple: (page_num, matches: bool)
    """
    temp_file = None
    try:
        # Extract single page
        temp_file = extract_single_page_pdf(pdf_path, page_num)
        
        # Parse with LlamaParse
        documents = parser.load_data(temp_file)
        
        if not documents:
            return page_num, False
        
        # Get text from first document
        text = documents[0].text if documents else ""
        
        if not text:
            return page_num, False
        
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
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


def find_matching_pages(pdf_path, filter_strings, max_workers=3, case_sensitive=False):
    """
    Find all pages that contain ALL the filter strings.
    
    Args:
        pdf_path: Path to PDF file
        filter_strings: List of strings that ALL must be present
        max_workers: Number of parallel workers (default: 3, be gentle with API)
        case_sensitive: Whether search is case-sensitive
        
    Returns:
        List of matching page numbers (0-based)
    """
    # Initialize LlamaParse
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not set! Get one free at https://cloud.llamaindex.ai/api-key")
    
    parser = LlamaParse(
        api_key=api_key,
        result_type="text",  # Just get text, not markdown
        verbose=False,
        language="en",
    )
    
    # Get total pages
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    filter_display = " AND ".join([f'"{s}"' for s in filter_strings])
    print(f"📄 Scanning {total_pages} pages for: {filter_display}")
    print(f"🔍 Using LlamaOCR (LlamaParse)")
    print(f"🧵 Processing with {max_workers} parallel workers")
    print()
    
    # Process pages in parallel
    matching_pages = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(check_page_matches, pdf_path, page_num, filter_strings, parser, case_sensitive): page_num
            for page_num in range(total_pages)
        }
        
        # Collect results
        for future in as_completed(futures):
            page_num, matches = future.result()
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


def split_pdf_with_llamaocr(input_pdf_path, output_folder, filter_strings, max_workers=3, case_sensitive=False):
    """
    Main function to split a PDF using LlamaOCR.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_folder: Path to the output folder for split PDFs
        filter_strings: List of strings that ALL must be present on a page
        max_workers: Number of parallel workers (default: 3)
        case_sensitive: Whether search is case-sensitive
        
    Returns:
        Number of sections created, or None if there was an error
    """
    
    print("=" * 70)
    print("PDF Splitting with LlamaOCR (LlamaParse)")
    print("=" * 70)
    print(f"Input PDF: {input_pdf_path}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find matching pages
    detection_pages = find_matching_pages(input_pdf_path, filter_strings, max_workers, case_sensitive)
    
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
    MAX_WORKERS = 3  # Be gentle with API
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
            print("⚠️  Warning: Invalid max_workers, using default of 3")
    
    # Validate input
    if not os.path.exists(INPUT_PDF):
        print(f"❌ Error: Input PDF '{INPUT_PDF}' not found!")
        sys.exit(1)
    
    if not FILTER_STRINGS or all(not s.strip() for s in FILTER_STRINGS):
        print("❌ Error: Filter strings cannot be empty!")
        sys.exit(1)
    
    # Run the splitting
    result = split_pdf_with_llamaocr(INPUT_PDF, OUTPUT_FOLDER, FILTER_STRINGS, MAX_WORKERS, CASE_SENSITIVE)
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
