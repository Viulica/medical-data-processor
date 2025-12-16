#!/usr/bin/env python3
"""
PDF Splitting using Gemini Flash Latest
========================================
This script splits PDFs by asking Gemini to identify pages containing specific text.
Process pages in batches per API call for efficiency.
"""

import os
import sys
import json
import time
import random
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter
import google.genai as genai
from google.genai import types
from pathlib import Path


def is_openrouter_model(model_name):
    """Check if model name indicates OpenRouter (contains '/' or starts with 'google/')"""
    return '/' in model_name or model_name.startswith('google/')


def ask_gemini_about_pages(client, pdf_path, page_start, page_end, filter_strings, model="gemini-flash-latest", max_retries=3):
    """
    Ask Gemini which pages in the given range contain the exact filter string.
    
    Args:
        client: Google GenAI client
        pdf_path: Path to the PDF file
        page_start: Starting page index (0-based)
        page_end: Ending page index (exclusive, 0-based)
        filter_strings: List containing a single string to match exactly (case-sensitive)
        model: Model name to use
        max_retries: Maximum number of retry attempts for parsing
    
    Returns:
        List of page numbers (0-based) that contain the exact filter string, or None on failure
    """
    
    # Create a temporary PDF with just these pages
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    actual_pages = []
    for page_idx in range(page_start, min(page_end, len(reader.pages))):
        writer.add_page(reader.pages[page_idx])
        actual_pages.append(page_idx)
    
    if not actual_pages:
        return []
    
    # Write temporary PDF
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        writer.write(temp_pdf)
        temp_pdf_path = temp_pdf.name
    
    try:
        # Create the prompt - filter_strings should be a single string
        filter_display = filter_strings[0] if isinstance(filter_strings, list) and filter_strings else str(filter_strings)
        
        prompt = f"""You are analyzing a PDF document to find pages that contain specific text.

I need you to identify which pages contain the following text string EXACTLY as written:
"{filter_display}"

CRITICAL INSTRUCTIONS:
1. CAREFULLY REVIEW EACH PAGE in this batch - examine every page thoroughly
2. A page matches ONLY if it contains the text EXACTLY as written above (case-sensitive match)
3. The text must appear exactly as: "{filter_display}"
4. Return your answer as a JSON object with this EXACT structure:
{{
    "matching_pages": [1, 3, 5]
}}

5. Page numbers should be 1-based (first page is 1, second page is 2, etc.)
6. If NO pages match, return: {{"matching_pages": []}}
7. Return ONLY the JSON object, no other text or explanation
8. The JSON must be valid and parseable

Example:
If pages 2 and 4 contain the exact text "{filter_display}", return:
{{"matching_pages": [2, 4]}}
"""
        
        # Read the PDF data
        with open(temp_pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        
        # Create the API request
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="application/pdf",
                        data=pdf_data,
                    ),
                    types.Part.from_text(text=prompt)
                ],
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
        )
        
        # Try up to max_retries times to get valid JSON
        for attempt in range(max_retries):
            try:
                # Call Gemini
                full_response = ""
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text is not None:
                        full_response += chunk.text
                
                response_text = full_response.strip()
                
                # Validate response
                if not response_text or len(response_text) < 2:
                    raise ValueError(f"Response too short: {response_text}")
                
                # Clean up response (remove markdown code blocks if present)
                cleaned_response = response_text
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # Parse JSON
                result = json.loads(cleaned_response)
                
                # Validate structure
                if "matching_pages" not in result:
                    raise ValueError(f"Missing 'matching_pages' key in response: {result}")
                
                if not isinstance(result["matching_pages"], list):
                    raise ValueError(f"'matching_pages' must be a list: {result}")
                
                # Convert batch-relative page numbers (1-based) to document-absolute page numbers (0-based)
                matching_pages = []
                
                for page_num in result["matching_pages"]:
                    if not isinstance(page_num, int):
                        raise ValueError(f"Page number must be integer: {page_num}")
                    if page_num < 1 or page_num > len(actual_pages):
                        print(f"  ⚠️  Warning: Page number {page_num} out of range for batch, skipping")
                        continue
                    # Convert: batch 1-based -> batch 0-based -> document 0-based
                    doc_page_idx = actual_pages[page_num - 1]
                    matching_pages.append(doc_page_idx)
                
                print(f"  ✅ Successfully parsed response for pages {page_start + 1}-{page_end} (attempt {attempt + 1})")
                if matching_pages:
                    # Convert to 1-based for display
                    display_pages = [p + 1 for p in matching_pages]
                    print(f"     Found matches on pages: {display_pages}")
                
                return matching_pages
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parsing failed for pages {page_start + 1}-{page_end} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    print(f"  ❌ Final JSON parsing failure for pages {page_start + 1}-{page_end}")
                    print(f"     Raw response: {response_text[:200]}...")
                    return None
            except ValueError as e:
                print(f"  ⚠️  Response validation failed for pages {page_start + 1}-{page_end} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    print(f"  ❌ Final validation failure for pages {page_start + 1}-{page_end}")
                    return None
            except Exception as e:
                print(f"  ⚠️  API call failed for pages {page_start + 1}-{page_end} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    print(f"  ❌ Final API failure for pages {page_start + 1}-{page_end}")
                    return None
            
            # Exponential backoff with jitter for retries
            if attempt < max_retries - 1:
                base_delay = 2 ** attempt  # 1, 2, 4 seconds
                jitter = random.uniform(0.5, 1.5)
                delay = base_delay * jitter
                print(f"  ⏳ Retrying pages {page_start + 1}-{page_end} in {delay:.1f} seconds...")
                time.sleep(delay)
        
        return None
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_pdf_path)
        except:
            pass


def find_matching_pages_with_gemini(pdf_path, filter_strings, batch_size=5, model="gemini-flash-latest", max_workers=12):
    """
    Find all pages that contain the exact filter string using Gemini with parallel processing.
    
    Args:
        pdf_path: Path to the PDF file
        filter_strings: List containing a single string to match exactly (case-sensitive)
        batch_size: Number of pages to process per API call
        model: Model name to use
        max_workers: Number of parallel threads (default: 3)
    
    Returns:
        List of page numbers (0-based) that match, or None if there was an error
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Initialize Gemini client
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set!")
        return None
    
    client = genai.Client(api_key=api_key)
    
    # Get total pages
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    filter_display = filter_strings[0] if isinstance(filter_strings, list) and filter_strings else str(filter_strings)
    print(f"📄 Scanning {total_pages} pages for: \"{filter_display}\"")
    print(f"🤖 Using model: {model}")
    print(f"📦 Processing {batch_size} pages per API call")
    print(f"🧵 Using {max_workers} parallel threads")
    
    # Create list of batch ranges with order indices
    batch_tasks = []
    batch_index = 0
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_tasks.append((batch_index, batch_start, batch_end))
        batch_index += 1
    
    print(f"🚀 Processing {len(batch_tasks)} batches in parallel...")
    
    # Store results with their order index to prevent mixing
    results_with_order = []
    failed_batches = []
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {}
        for order_idx, batch_start, batch_end in batch_tasks:
            future = executor.submit(
                ask_gemini_about_pages,
                client, pdf_path, batch_start, batch_end, filter_strings, model
            )
            future_to_batch[future] = (order_idx, batch_start, batch_end)
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            order_idx, batch_start, batch_end = future_to_batch[future]
            
            try:
                matching_pages = future.result()
                
                if matching_pages is None:
                    print(f"❌ Failed to process batch {batch_start + 1}-{batch_end} after all retries")
                    failed_batches.append((batch_start, batch_end))
                else:
                    # Store with order index to sort later
                    results_with_order.append((order_idx, matching_pages))
                    
            except Exception as e:
                print(f"❌ Exception processing batch {batch_start + 1}-{batch_end}: {str(e)}")
                failed_batches.append((batch_start, batch_end))
    
    # Check if any batches failed
    if failed_batches:
        print(f"\n❌ {len(failed_batches)} batch(es) failed:")
        for batch_start, batch_end in failed_batches:
            print(f"   - Pages {batch_start + 1}-{batch_end}")
        return None
    
    # Sort results by order index to maintain correct sequence
    results_with_order.sort(key=lambda x: x[0])
    
    # Collect all matching pages in correct order
    all_matching_pages = []
    for order_idx, matching_pages in results_with_order:
        all_matching_pages.extend(matching_pages)
    
    print(f"\n✅ Scan complete! Found {len(all_matching_pages)} matching pages")
    if all_matching_pages:
        # Convert to 1-based for display
        display_pages = [p + 1 for p in sorted(all_matching_pages)]
        print(f"   Matching pages: {display_pages}")
    
    return sorted(all_matching_pages)


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
                end_page = detection_pages[i + 1]  # Stop before next detection
            else:
                end_page = total_pages  # Last section goes to end
            
            # Create PDF for this section
            writer = PdfWriter()
            pages_in_section = end_page - start_page
            
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


def split_pdf_with_gemini(input_pdf_path, output_folder, filter_strings, batch_size=5, model="gemini-flash-latest", max_workers=12):
    """
    Main function to split a PDF using Gemini with parallel processing.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_folder: Path to the output folder for split PDFs
        filter_strings: List containing a single string to match exactly (case-sensitive)
        batch_size: Number of pages to process per API call (default: 30)
        model: Model name to use (default: gemini-flash-latest)
        max_workers: Number of parallel threads (default: 3)
    
    Returns:
        Number of sections created, or None if there was an error
    """
    
    print("=" * 70)
    print("PDF Splitting (New Method)")
    print("=" * 70)
    print(f"Input PDF: {input_pdf_path}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find matching pages with parallel processing
    detection_pages = find_matching_pages_with_gemini(
        input_pdf_path, filter_strings, batch_size, model, max_workers
    )
    
    if detection_pages is None:
        print("\n❌ Failed to find matching pages")
        return None
    
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
    """Command-line interface for the script"""
    
    # Default configuration (can be overridden by command-line args)
    INPUT_PDF = "input.pdf"
    OUTPUT_FOLDER = "output"
    FILTER_STRINGS = ["Patient Address"]  # Default filter
    BATCH_SIZE = 5  # Process 5 pages per API call
    MODEL = "gemini-flash-latest"
    MAX_WORKERS = 12  # Process 12 batches in parallel
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        INPUT_PDF = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FOLDER = sys.argv[2]
    if len(sys.argv) > 3:
        # Filter strings can be provided as a single comma-separated argument
        filter_arg = sys.argv[3]
        if ',' in filter_arg:
            FILTER_STRINGS = [s.strip() for s in filter_arg.split(',')]
        else:
            FILTER_STRINGS = [filter_arg.strip()]
    if len(sys.argv) > 4:
        try:
            BATCH_SIZE = int(sys.argv[4])
        except ValueError:
            print("⚠️  Warning: Invalid batch size, using default of 10")
    if len(sys.argv) > 5:
        MODEL = sys.argv[5]
    if len(sys.argv) > 6:
        try:
            MAX_WORKERS = int(sys.argv[6])
        except ValueError:
            print("⚠️  Warning: Invalid max_workers, using default of 3")
    
    # Validate input
    if not os.path.exists(INPUT_PDF):
        print(f"❌ Error: Input PDF '{INPUT_PDF}' not found!")
        sys.exit(1)
    
    if not FILTER_STRINGS or all(not s.strip() for s in FILTER_STRINGS):
        print("❌ Error: Filter strings cannot be empty!")
        sys.exit(1)
    
    # Run the splitting with parallel processing
    result = split_pdf_with_gemini(
        INPUT_PDF, OUTPUT_FOLDER, FILTER_STRINGS, BATCH_SIZE, MODEL, MAX_WORKERS
    )
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

