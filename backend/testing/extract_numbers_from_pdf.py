#!/usr/bin/env python3
"""
Extract only numbers from PDF files.
This script extracts all selectable text and filters for numbers.
"""

import fitz  # PyMuPDF
import re
import sys
from pathlib import Path
from collections import defaultdict


def extract_numbers_from_pdf(pdf_path, output_file=None):
    """
    Extract all numbers from selectable text in a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Optional path to save extracted numbers. If None, prints to stdout.
    
    Returns:
        Dictionary with page numbers as keys and lists of numbers as values
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    extracted_numbers = {}
    all_numbers = []
    
    print(f"Total pages: {len(doc)}")
    print("=" * 60)
    
    # Pattern to match numbers (integers, decimals, dates, etc.)
    # This will match:
    # - Simple integers: 123, 4567
    # - Decimals: 12.34, 0.5
    # - Dates: 03/24/25, 03-24-25
    # - Numbers with special chars: 25032428
    number_pattern = re.compile(r'\d+\.?\d*')
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text from the page
        text = page.get_text("text")
        
        # Find all numbers in the text
        numbers = number_pattern.findall(text)
        
        # Also get text blocks with positions for context
        text_dict = page.get_text("dict")
        
        extracted_numbers[page_num + 1] = numbers
        
        print(f"\nPage {page_num + 1}:")
        print("-" * 60)
        
        if numbers:
            print(f"Found {len(numbers)} number(s):")
            for num in numbers:
                print(f"  - {num}")
            all_numbers.append(f"=== Page {page_num + 1} ===\n{', '.join(numbers)}\n")
        else:
            print("(No numbers found on this page)")
            all_numbers.append(f"=== Page {page_num + 1} ===\n(No numbers found)\n")
        
        # Show context for numbers (what text surrounds them)
        if text.strip():
            print(f"\nFull text context:")
            print(text[:200] + "..." if len(text) > 200 else text)
    
    total_pages = len(doc)
    doc.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_numbers = sum(len(nums) for nums in extracted_numbers.values())
    pages_with_numbers = sum(1 for nums in extracted_numbers.values() if nums)
    print(f"Total numbers found: {total_numbers}")
    print(f"Pages with numbers: {pages_with_numbers} out of {total_pages}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("EXTRACTED NUMBERS FROM PDF\n")
            f.write("=" * 60 + "\n\n")
            f.write('\n'.join(all_numbers))
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"SUMMARY: {total_numbers} numbers found on {pages_with_numbers} pages\n")
        print(f"\nExtracted numbers saved to: {output_path}")
    
    return extracted_numbers


def extract_numbers_with_context(pdf_path):
    """
    Extract numbers with their surrounding context (nearby text).
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    print(f"\nExtracting numbers with context from: {pdf_path}")
    print("=" * 60)
    
    number_pattern = re.compile(r'\d+\.?\d*')
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        if not text.strip():
            continue
        
        print(f"\nPage {page_num + 1}:")
        print("-" * 60)
        
        # Find numbers with their positions in the text
        for match in number_pattern.finditer(text):
            num = match.group()
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].replace('\n', ' ')
            print(f"  {num}: ...{context}...")
    
    doc.close()


if __name__ == "__main__":
    # Default to INJE.pdf in the same directory
    script_dir = Path(__file__).parent
    default_pdf = script_dir / "INJE.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = default_pdf
    
    # Check if user wants context information
    show_context = "--context" in sys.argv or "-c" in sys.argv
    
    if show_context:
        extract_numbers_with_context(pdf_path)
    else:
        # Extract numbers
        output_file = None
        if "--output" in sys.argv or "-o" in sys.argv:
            try:
                idx = sys.argv.index("--output") if "--output" in sys.argv else sys.argv.index("-o")
                output_file = sys.argv[idx + 1]
            except (IndexError, ValueError):
                pass
        
        extract_numbers_from_pdf(pdf_path, output_file)

