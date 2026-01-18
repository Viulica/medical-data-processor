#!/usr/bin/env python3
"""
Extract text that was PASTED or added as annotations on a PDF.
This includes text annotations, form fields, and other annotation types.
"""

import fitz  # PyMuPDF
import sys
from pathlib import Path
from collections import defaultdict


def extract_annotations_from_pdf(pdf_path, output_file=None):
    """
    Extract all annotations (including pasted text) from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Optional path to save extracted annotations. If None, prints to stdout.
    
    Returns:
        Dictionary with page numbers as keys and lists of annotations as values
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    all_annotations = defaultdict(list)
    annotation_texts = []
    
    print(f"Total pages: {len(doc)}")
    print("=" * 60)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        annotations = list(page.annots())  # Convert generator to list
        
        print(f"\nPage {page_num + 1}:")
        print("-" * 60)
        
        if annotations:
            print(f"Found {len(annotations)} annotation(s):")
            
            for annot in annotations:
                annot_type = annot.type[1]  # Get annotation type name
                annot_info = {
                    'type': annot_type,
                    'rect': annot.rect,  # Bounding box
                    'content': None,
                    'text': None,
                    'info': {}
                }
                
                # Get annotation content/text
                try:
                    # Try to get the annotation content
                    content = annot.info.get('content', '')
                    title = annot.info.get('title', '')
                    
                    # For text annotations, get the actual text
                    if annot_type == 'Text' or annot_type == 'FreeText':
                        # Try to get the text content
                        try:
                            text_content = annot.get_text()
                            if text_content:
                                annot_info['text'] = text_content
                        except:
                            pass
                    
                    # Get all info
                    annot_info['content'] = content
                    annot_info['info'] = dict(annot.info)
                    
                    # Print annotation details
                    print(f"\n  Annotation Type: {annot_type}")
                    print(f"    Position: ({annot.rect.x0:.1f}, {annot.rect.y0:.1f}) to ({annot.rect.x1:.1f}, {annot.rect.y1:.1f})")
                    
                    if content:
                        print(f"    Content: {content}")
                    if title:
                        print(f"    Title: {title}")
                    if annot_info['text']:
                        print(f"    Text: {annot_info['text']}")
                    
                    # Show all info fields
                    if annot.info:
                        print(f"    Info fields: {list(annot.info.keys())}")
                    
                    all_annotations[page_num + 1].append(annot_info)
                    
                    # Collect text for output
                    text_parts = []
                    if title:
                        text_parts.append(f"Title: {title}")
                    if content:
                        text_parts.append(f"Content: {content}")
                    if annot_info['text']:
                        text_parts.append(f"Text: {annot_info['text']}")
                    
                    if text_parts:
                        annotation_texts.append(f"Page {page_num + 1} - {annot_type}: {' | '.join(text_parts)}")
                
                except Exception as e:
                    print(f"    Error reading annotation: {e}")
                    annot_info['error'] = str(e)
                    all_annotations[page_num + 1].append(annot_info)
        else:
            print("(No annotations found on this page)")
            annotation_texts.append(f"Page {page_num + 1}: (No annotations)")
    
    total_pages = len(doc)
    doc.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_annotations = sum(len(annots) for annots in all_annotations.values())
    pages_with_annotations = len(all_annotations)
    print(f"Total annotations found: {total_annotations}")
    print(f"Pages with annotations: {pages_with_annotations} out of {total_pages}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("EXTRACTED ANNOTATIONS (PASTED TEXT) FROM PDF\n")
            f.write("=" * 60 + "\n\n")
            f.write('\n'.join(annotation_texts))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"SUMMARY: {total_annotations} annotations found on {pages_with_annotations} pages\n")
        print(f"\nExtracted annotations saved to: {output_path}")
    
    return dict(all_annotations)


def extract_form_fields_from_pdf(pdf_path):
    """
    Extract text from form fields (another type of "pasted" text).
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    print(f"\nExtracting form fields from: {pdf_path}")
    print("=" * 60)
    
    form_fields = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        widgets = page.widgets()
        
        if widgets:
            print(f"\nPage {page_num + 1}:")
            print("-" * 60)
            print(f"Found {len(widgets)} form field(s):")
            
            for widget in widgets:
                field_type = widget.field_type_string
                field_name = widget.field_name
                field_value = widget.field_value
                
                print(f"\n  Field Name: {field_name}")
                print(f"    Type: {field_type}")
                print(f"    Value: {field_value}")
                print(f"    Position: {widget.rect}")
                
                if field_value:
                    form_fields.append(f"Page {page_num + 1} - {field_name}: {field_value}")
    
    doc.close()
    
    if not form_fields:
        print("\n(No form fields found)")
    else:
        print(f"\n\nTotal form fields with values: {len(form_fields)}")
    
    return form_fields


def extract_all_pasted_content(pdf_path, output_file=None):
    """
    Extract all types of "pasted" content: annotations, form fields, etc.
    """
    pdf_path = Path(pdf_path)
    
    print("=" * 60)
    print("EXTRACTING ALL PASTED/ANNOTATED CONTENT")
    print("=" * 60)
    
    # Extract annotations
    annotations = extract_annotations_from_pdf(pdf_path)
    
    # Extract form fields
    form_fields = extract_form_fields_from_pdf(pdf_path)
    
    # Combine results
    all_content = []
    
    if annotations:
        all_content.append("\n=== ANNOTATIONS ===")
        for page_num, annots in annotations.items():
            for annot in annots:
                parts = []
                if annot.get('content'):
                    parts.append(annot['content'])
                if annot.get('text'):
                    parts.append(annot['text'])
                if parts:
                    all_content.append(f"Page {page_num}: {' | '.join(parts)}")
    
    if form_fields:
        all_content.append("\n=== FORM FIELDS ===")
        all_content.extend(form_fields)
    
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ALL PASTED/ANNOTATED CONTENT FROM PDF\n")
            f.write("=" * 60 + "\n\n")
            f.write('\n'.join(all_content))
        print(f"\nAll pasted content saved to: {output_path}")
    
    return all_content


if __name__ == "__main__":
    # Default to INJE.pdf in the same directory
    script_dir = Path(__file__).parent
    default_pdf = script_dir / "INJE.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = default_pdf
    
    # Check what to extract
    if "--form-fields" in sys.argv or "-f" in sys.argv:
        extract_form_fields_from_pdf(pdf_path)
    elif "--all" in sys.argv or "-a" in sys.argv:
        output_file = None
        if "--output" in sys.argv or "-o" in sys.argv:
            try:
                idx = sys.argv.index("--output") if "--output" in sys.argv else sys.argv.index("-o")
                output_file = sys.argv[idx + 1]
            except (IndexError, ValueError):
                pass
        extract_all_pasted_content(pdf_path, output_file)
    else:
        # Extract annotations by default
        output_file = None
        if "--output" in sys.argv or "-o" in sys.argv:
            try:
                idx = sys.argv.index("--output") if "--output" in sys.argv else sys.argv.index("-o")
                output_file = sys.argv[idx + 1]
            except (IndexError, ValueError):
                pass
        
        extract_annotations_from_pdf(pdf_path, output_file)

