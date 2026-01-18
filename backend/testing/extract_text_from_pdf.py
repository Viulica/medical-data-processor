#!/usr/bin/env python3
"""
Extract selectable text from PDF files.
This script extracts all text that can be selected in a PDF viewer.
"""

import fitz  # PyMuPDF
import sys
from pathlib import Path


def extract_text_from_pdf(pdf_path, output_file=None):
    """
    Extract all selectable text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Optional path to save extracted text. If None, prints to stdout.
    
    Returns:
        Dictionary with page numbers as keys and extracted text as values
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    extracted_text = {}
    all_text = []
    
    print(f"Total pages: {len(doc)}")
    print("-" * 60)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text from the page
        # Using "text" parameter gets selectable text
        text = page.get_text("text")
        
        # Also try "dict" format to get more structured information
        text_dict = page.get_text("dict")
        
        extracted_text[page_num + 1] = text
        
        print(f"\nPage {page_num + 1}:")
        print("-" * 60)
        
        if text.strip():
            print(text)
            all_text.append(f"=== Page {page_num + 1} ===\n{text}\n")
        else:
            print("(No selectable text found on this page)")
            all_text.append(f"=== Page {page_num + 1} ===\n(No selectable text found)\n")
        
        # Print some statistics about text blocks
        if text_dict.get("blocks"):
            text_blocks = [b for b in text_dict["blocks"] if b.get("type") == 0]  # type 0 = text
            print(f"\nFound {len(text_blocks)} text block(s) on this page")
            
            # Show coordinates of text blocks (useful for debugging)
            for i, block in enumerate(text_blocks[:5]):  # Show first 5 blocks
                bbox = block.get("bbox", [])
                if bbox:
                    print(f"  Block {i+1}: bbox=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
    
    doc.close()
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        print(f"\n\nExtracted text saved to: {output_path}")
    
    return extracted_text


def extract_text_with_positions(pdf_path):
    """
    Extract text with position information (coordinates).
    Useful for understanding where text appears on the page.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    print(f"\nExtracting text with positions from: {pdf_path}")
    print("=" * 60)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_dict = page.get_text("dict")
        
        print(f"\nPage {page_num + 1}:")
        print("-" * 60)
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                bbox = block.get("bbox", [])
                print(f"\nText block at ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f}):")
                
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    if line_text.strip():
                        print(f"  {line_text}")
    
    doc.close()


if __name__ == "__main__":
    # Default to INJE.pdf in the same directory
    script_dir = Path(__file__).parent
    default_pdf = script_dir / "INJE.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = default_pdf
    
    # Check if user wants position information
    show_positions = "--positions" in sys.argv or "-p" in sys.argv
    
    if show_positions:
        extract_text_with_positions(pdf_path)
    else:
        # Extract text
        output_file = None
        if "--output" in sys.argv or "-o" in sys.argv:
            try:
                idx = sys.argv.index("--output") if "--output" in sys.argv else sys.argv.index("-o")
                output_file = sys.argv[idx + 1]
            except (IndexError, ValueError):
                pass
        
        extract_text_from_pdf(pdf_path, output_file)

