#!/usr/bin/env python3
"""
Simple script to extract only the TEXT content from pasted annotations.
This extracts just the text values, without all the metadata.
"""

import fitz  # PyMuPDF
import sys
from pathlib import Path


def extract_pasted_text_simple(pdf_path, output_file=None):
    """
    Extract only the text content from FreeText annotations (pasted text).
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Optional path to save extracted text. If None, prints to stdout.
    
    Returns:
        List of extracted text strings
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    
    print(f"Extracting pasted text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    all_text = []
    text_by_page = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        annotations = list(page.annots())
        
        page_texts = []
        
        for annot in annotations:
            # Only extract FreeText annotations (these are the "pasted" text)
            if annot.type[1] == 'FreeText':
                try:
                    # Get the text content
                    text = annot.get_text()
                    if text and text.strip():
                        all_text.append(text.strip())
                        page_texts.append(text.strip())
                        print(f"Page {page_num + 1}: {text.strip()}")
                except:
                    # Fallback to content field
                    try:
                        content = annot.info.get('content', '')
                        if content and content.strip():
                            all_text.append(content.strip())
                            page_texts.append(content.strip())
                            print(f"Page {page_num + 1}: {content.strip()}")
                    except:
                        pass
        
        if page_texts:
            text_by_page[page_num + 1] = page_texts
    
    doc.close()
    
    print(f"\n{'='*60}")
    print(f"Total pasted text items found: {len(all_text)}")
    print(f"Pages with pasted text: {len(text_by_page)}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("PASTED TEXT FROM PDF\n")
            f.write("=" * 60 + "\n\n")
            
            for page_num, texts in text_by_page.items():
                f.write(f"Page {page_num}:\n")
                for text in texts:
                    f.write(f"  - {text}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("ALL TEXT (one per line):\n")
            for text in all_text:
                f.write(f"{text}\n")
        
        print(f"\nExtracted text saved to: {output_path}")
    
    return all_text


if __name__ == "__main__":
    # Default to INJE.pdf in the same directory
    script_dir = Path(__file__).parent
    default_pdf = script_dir / "INJE.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = default_pdf
    
    output_file = None
    if "--output" in sys.argv or "-o" in sys.argv:
        try:
            idx = sys.argv.index("--output") if "--output" in sys.argv else sys.argv.index("-o")
            output_file = sys.argv[idx + 1]
        except (IndexError, ValueError):
            pass
    
    extract_pasted_text_simple(pdf_path, output_file)

