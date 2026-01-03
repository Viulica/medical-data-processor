#!/usr/bin/env python3
"""
PDF Splitting using Gemini 3 Flash Preview with Custom Prompt
==============================================================
This script splits PDFs by asking Gemini to identify pages based on a custom prompt.
The prompt describes what to look for (e.g., patient record starts with "25..." IDs).
Returns page indexes directly.
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


def ask_gemini_for_page_indexes(client, pdf_path, page_start, page_end, custom_prompt, model="gemini-3-flash-preview", max_retries=3):
    """
    Ask Gemini to identify page indexes based on a custom prompt.
    
    Args:
        client: Google GenAI client
        pdf_path: Path to the PDF file
        page_start: Starting page index (0-based)
        page_end: Ending page index (exclusive, 0-based)
        custom_prompt: Custom prompt describing what to look for
        model: Model name to use
        max_retries: Maximum number of retry attempts for parsing
    
    Returns:
        List of page numbers (0-based) that match the prompt, or None on failure
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
        num_pages_in_batch = len(actual_pages)
        
        # Build the prompt
        prompt = f"""{custom_prompt}

This batch contains {num_pages_in_batch} page(s) from the PDF (pages {page_start + 1} to {page_end} in the full document).

CRITICAL INSTRUCTIONS:
1. Analyze ALL {num_pages_in_batch} pages in this batch carefully
2. Identify which pages match the criteria described above
3. Return your answer as a JSON object with page indexes (1-based within this batch):
{{
    "matching_pages": [1, 3, 5]
}}

4. Page numbers should be 1-based within this batch (1 = first page in batch, 2 = second page, etc.)
5. Return ONLY the JSON object, no other text or explanation
6. The JSON must be valid and parseable
7. If no pages match, return: {{"matching_pages": []}}
8. Only include pages that clearly match the criteria

Example: If pages 2 and 4 in this batch match, return:
{{
    "matching_pages": [2, 4]
}}
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
        
        # Use thinking_level="HIGH" for Gemini 3 models
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
                if not isinstance(result, dict):
                    raise ValueError(f"Response must be a JSON object: {result}")
                
                if "matching_pages" not in result:
                    raise ValueError(f"Response must contain 'matching_pages' key: {result}")
                
                matching_batch_pages = result["matching_pages"]
                if not isinstance(matching_batch_pages, list):
                    raise ValueError(f"'matching_pages' must be a list: {matching_batch_pages}")
                
                # Convert batch-relative page numbers (1-based) to document-absolute page numbers (0-based)
                matching_pages = []
                for batch_page_num in matching_batch_pages:
                    if not isinstance(batch_page_num, int):
                        continue
                    if batch_page_num < 1 or batch_page_num > len(actual_pages):
                        print(f"  ⚠️  Warning: Page {batch_page_num} out of range (1-{len(actual_pages)}), skipping")
                        continue
                    # Convert: batch 1-based -> batch 0-based -> document 0-based
                    doc_page_idx = actual_pages[batch_page_num - 1]
                    matching_pages.append(doc_page_idx)
                
                print(f"  ✅ Successfully parsed response for pages {page_start + 1}-{page_end} (attempt {attempt + 1})")
                if matching_pages:
                    # Convert to 1-based for display
                    display_pages = [p + 1 for p in matching_pages]
                    print(f"     Found matches on pages: {display_pages}")
                else:
                    print(f"     No matches found in this batch")
                
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


def shift_detection_pages(detection_pages, shift_amount, total_pages):
    """Shift detection pages by the specified amount, clamping to valid page range."""
    if shift_amount == 0:
        return detection_pages
    
    shifted_pages = []
    for page_num in detection_pages:
        shifted_page = page_num + shift_amount
        # Clamp to valid page range (0 to total_pages - 1)
        shifted_page = max(0, min(shifted_page, total_pages - 1))
        shifted_pages.append(shifted_page)
    
    # Remove duplicates and sort (in case shifting caused overlaps)
    shifted_pages = sorted(list(set(shifted_pages)))
    
    if shift_amount != 0:
        shift_direction = "down" if shift_amount > 0 else "up"
        print(f"  🔄 Shifted detections {abs(shift_amount)} page(s) {shift_direction}")
        if detection_pages != shifted_pages:
            print(f"     Original: {[p+1 for p in detection_pages]}")
            print(f"     Shifted:  {[p+1 for p in shifted_pages]}")
    
    return shifted_pages


def find_matching_pages_with_gemini_prompt(pdf_path, custom_prompt, batch_size=5, model="gemini-3-flash-preview", max_workers=12):
    """
    Find all pages that match the custom prompt using Gemini with parallel processing.
    
    Args:
        pdf_path: Path to the PDF file
        custom_prompt: Custom prompt describing what to look for
        batch_size: Number of pages to process per API call
        model: Model name to use
        max_workers: Number of parallel threads
    
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
    
    print(f"📄 Scanning {total_pages} pages with custom prompt")
    print(f"🤖 Using model: {model}")
    print(f"📦 Processing {batch_size} pages per API call")
    print(f"🧵 Using {max_workers} parallel threads")
    print(f"📝 Prompt: {custom_prompt[:100]}..." if len(custom_prompt) > 100 else f"📝 Prompt: {custom_prompt}")
    
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
                ask_gemini_for_page_indexes,
                client, pdf_path, batch_start, batch_end, custom_prompt, model
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


def split_pdf_with_gemini_prompt(input_pdf_path, output_folder, custom_prompt, batch_size=5, model="gemini-3-flash-preview", max_workers=12, detection_shift=0):
    """
    Main function to split a PDF using Gemini with a custom prompt.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_folder: Path to the output folder for split PDFs
        custom_prompt: Custom prompt describing what to look for
        batch_size: Number of pages to process per API call
        model: Model name to use
        max_workers: Number of parallel threads
        detection_shift: Shift detections by N pages (positive = down, negative = up)
    
    Returns:
        Number of sections created, or None if there was an error
    """
    
    print("=" * 70)
    print("PDF Splitting with Gemini (Custom Prompt)")
    print("=" * 70)
    print(f"Input PDF: {input_pdf_path}")
    print(f"Output folder: {output_folder}")
    if detection_shift != 0:
        shift_direction = "down" if detection_shift > 0 else "up"
        print(f"Detection shift: {abs(detection_shift)} page(s) {shift_direction}")
    print()
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find matching pages with parallel processing
    detection_pages = find_matching_pages_with_gemini_prompt(
        input_pdf_path, custom_prompt, batch_size, model, max_workers
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
    
    # Apply shift if specified
    if detection_shift != 0:
        detection_pages = shift_detection_pages(detection_pages, detection_shift, total_pages)
    
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
    
    # Default configuration
    INPUT_PDF = "input.pdf"
    OUTPUT_FOLDER = "output"
    CUSTOM_PROMPT = 'your task is to identify starts of a unique patient record in a big pdf file, a patient record start will always have a pasted looking id that starts with "25..." somewhere on the first page of that patient, your output is ONLY the indexes of the patient starting pages'
    BATCH_SIZE = 5
    MODEL = "gemini-3-flash-preview"
    MAX_WORKERS = 12
    DETECTION_SHIFT = 0
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        INPUT_PDF = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FOLDER = sys.argv[2]
    if len(sys.argv) > 3:
        CUSTOM_PROMPT = sys.argv[3]
    if len(sys.argv) > 4:
        try:
            BATCH_SIZE = int(sys.argv[4])
        except ValueError:
            print("⚠️  Warning: Invalid batch size, using default of 5")
    if len(sys.argv) > 5:
        MODEL = sys.argv[5]
    if len(sys.argv) > 6:
        try:
            MAX_WORKERS = int(sys.argv[6])
        except ValueError:
            print("⚠️  Warning: Invalid max_workers, using default of 12")
    if len(sys.argv) > 7:
        try:
            DETECTION_SHIFT = int(sys.argv[7])
        except ValueError:
            print("⚠️  Warning: Invalid detection_shift, using default of 0")
    
    # Validate input
    if not os.path.exists(INPUT_PDF):
        print(f"❌ Error: Input PDF '{INPUT_PDF}' not found!")
        sys.exit(1)
    
    if not CUSTOM_PROMPT or not CUSTOM_PROMPT.strip():
        print("❌ Error: Custom prompt cannot be empty!")
        sys.exit(1)
    
    # Run the splitting
    result = split_pdf_with_gemini_prompt(
        INPUT_PDF, OUTPUT_FOLDER, CUSTOM_PROMPT, BATCH_SIZE, MODEL, MAX_WORKERS, DETECTION_SHIFT
    )
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

