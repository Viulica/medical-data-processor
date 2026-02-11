"""
Utility functions for extracting provider codes from PDF annotations.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available, provider annotation extraction will not work")

logger = logging.getLogger(__name__)


def parse_provider_mapping(provider_mapping_text: str) -> Dict[str, Dict[str, str]]:
    """
    Parse provider mapping text into a dictionary mapping mednet codes to provider info.
    
    Input format example:
    '''
    Billable CRNA providers:
    Smith, John, CRNA (MedNet Code: 7)
    Doe, Jane M, CRNA (MedNet Code: 12)
    
    Billable MD providers:
    Johnson, Robert, MD (MedNet Code: 1)
    Williams, Sarah K, MD (MedNet Code: 5)
    '''
    
    Returns:
        Dict mapping mednet_code -> {'name': str, 'title': str (CRNA/MD)}
        Example: {'7': {'name': 'Smith, John, CRNA', 'title': 'CRNA'}, ...}
    """
    if not provider_mapping_text:
        return {}
    
    providers = {}
    current_title = None
    
    lines = provider_mapping_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Check for section headers
        if 'CRNA' in line.upper() and 'PROVIDER' in line.upper():
            current_title = 'CRNA'
            continue
        elif 'MD' in line.upper() and 'PROVIDER' in line.upper():
            current_title = 'MD'
            continue
        
        # Skip empty lines
        if not line:
            continue
        
        # Extract provider info
        # Pattern: {name}, {title} (MedNet Code: {code})
        match = re.search(r'\(MedNet Code:\s*(\d+)\)', line, re.IGNORECASE)
        if match:
            mednet_code = match.group(1)
            
            # Extract the full name and title (everything before the MedNet Code part)
            name_part = line[:match.start()].strip()
            
            # Determine title from the line
            if ', CRNA' in name_part:
                title = 'CRNA'
            elif ', MD' in name_part:
                title = 'MD'
            else:
                # Fallback to section title
                title = current_title if current_title else 'UNKNOWN'
            
            providers[mednet_code] = {
                'name': name_part,
                'title': title
            }
            
            logger.debug(f"Parsed provider: Code {mednet_code} -> {name_part} ({title})")
    
    logger.info(f"Parsed {len(providers)} providers from mapping")
    return providers


def extract_annotations_from_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from FreeText annotations (pasted text) in a PDF.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of extracted text strings from annotations
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available, cannot extract annotations")
        return []
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.warning(f"PDF file not found: {pdf_path}")
        return []
    
    try:
        doc = fitz.open(pdf_path)
        annotation_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            annotations = list(page.annots())
            
            for annot in annotations:
                # Only extract FreeText annotations (pasted text)
                if annot.type[1] == 'FreeText':
                    try:
                        # Try to get the text content
                        text = annot.get_text()
                        if text and text.strip():
                            annotation_texts.append(text.strip())
                    except:
                        # Fallback to content field
                        try:
                            content = annot.info.get('content', '')
                            if content and content.strip():
                                annotation_texts.append(content.strip())
                        except:
                            pass
        
        doc.close()
        
        logger.debug(f"Extracted {len(annotation_texts)} annotations from {pdf_path.name}")
        return annotation_texts
    
    except Exception as e:
        logger.error(f"Error extracting annotations from {pdf_path}: {e}")
        return []


def match_providers_from_annotations(
    annotation_texts: List[str],
    provider_mapping: Dict[str, Dict[str, str]]
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Match provider codes from annotation texts against the provider mapping.

    Handles cases like:
    - Single code: "7" -> CRNA provider with code 7
    - Two providers: "7/1" -> CRNA (7) and MD (1)
    - With SRNA suffix: "1/SRNA" or "1/7/SRNA" -> providers + SRNA flag

    Args:
        annotation_texts: List of text strings extracted from annotations
        provider_mapping: Dict mapping mednet_code -> provider info

    Returns:
        Tuple of (responsible_provider, md_provider, crna_provider, has_srna)
        - responsible_provider: The first provider found (or None)
        - md_provider: MD provider name (or None)
        - crna_provider: CRNA provider name (or None)
        - has_srna: True if SRNA was present in the annotation
    """
    if not annotation_texts or not provider_mapping:
        return None, None, None, False

    # Pattern to match:
    # - Single number: "7", "12", "509"
    # - Two numbers separated by /: "7/1", "12/5"
    # - Numbers can have leading zeros: "07", "01"
    # - Any pattern can end with /SRNA: "1/SRNA", "1/7/SRNA"

    for text in annotation_texts:
        text = text.strip()

        # Check for SRNA suffix and strip it
        has_srna = False
        if re.search(r'/\s*SRNA\s*$', text, re.IGNORECASE):
            has_srna = True
            text = re.sub(r'\s*/\s*SRNA\s*$', '', text, flags=re.IGNORECASE).strip()

        # Try to match "X/Y" pattern (two providers)
        match_two = re.match(r'^(\d+)\s*/\s*(\d+)$', text)
        if match_two:
            code1 = match_two.group(1).lstrip('0') or '0'
            code2 = match_two.group(2).lstrip('0') or '0'

            provider1 = provider_mapping.get(code1)
            provider2 = provider_mapping.get(code2)

            if provider1 or provider2:
                # Determine which is MD and which is CRNA
                md_provider = None
                crna_provider = None
                responsible_provider = None

                if provider1:
                    if provider1['title'] == 'MD':
                        md_provider = provider1['name']
                    elif provider1['title'] == 'CRNA':
                        crna_provider = provider1['name']

                    # First provider is responsible
                    responsible_provider = provider1['name']

                if provider2:
                    if provider2['title'] == 'MD':
                        md_provider = provider2['name']
                    elif provider2['title'] == 'CRNA':
                        crna_provider = provider2['name']

                    # If we didn't find a responsible provider yet, use provider2
                    if not responsible_provider:
                        responsible_provider = provider2['name']

                logger.info(f"Matched two providers from annotation '{text}': Responsible={responsible_provider}, MD={md_provider}, CRNA={crna_provider}, SRNA={has_srna}")
                return responsible_provider, md_provider, crna_provider, has_srna

        # Try to match single number
        match_one = re.match(r'^(\d+)$', text)
        if match_one:
            code = match_one.group(1).lstrip('0') or '0'
            provider = provider_mapping.get(code)

            if provider:
                responsible_provider = provider['name']
                md_provider = provider['name'] if provider['title'] == 'MD' else None
                crna_provider = provider['name'] if provider['title'] == 'CRNA' else None

                logger.info(f"Matched single provider from annotation '{text}': {responsible_provider} ({provider['title']}), SRNA={has_srna}")
                return responsible_provider, md_provider, crna_provider, has_srna

    # No matches found
    logger.debug(f"No provider matches found in annotations: {annotation_texts}")
    return None, None, None, False


def extract_and_match_providers(
    pdf_path: str,
    provider_mapping_text: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Extract annotations from a PDF and match them against provider mapping.

    This is the main function to use for provider extraction from PDF annotations.

    Args:
        pdf_path: Path to the PDF file
        provider_mapping_text: Provider mapping text (output from provider-mapping endpoint)

    Returns:
        Tuple of (responsible_provider, md_provider, crna_provider, has_srna)
    """
    if not provider_mapping_text:
        return None, None, None, False

    # Parse the provider mapping
    provider_mapping = parse_provider_mapping(provider_mapping_text)

    if not provider_mapping:
        logger.warning("No providers parsed from mapping text")
        return None, None, None, False

    # Extract annotations from PDF
    annotation_texts = extract_annotations_from_pdf(pdf_path)

    if not annotation_texts:
        logger.debug(f"No annotations found in {pdf_path}")
        return None, None, None, False

    # Match providers
    return match_providers_from_annotations(annotation_texts, provider_mapping)


if __name__ == "__main__":
    # Test the parsing function
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        # Test extracting annotations
        print(f"Extracting annotations from: {pdf_path}")
        annotations = extract_annotations_from_pdf(pdf_path)
        print(f"\nFound {len(annotations)} annotations:")
        for i, text in enumerate(annotations, 1):
            print(f"  {i}. {text}")
        
        # Test with sample provider mapping
        sample_mapping = """Billable CRNA providers:
Smith, John, CRNA (MedNet Code: 7)
Doe, Jane M, CRNA (MedNet Code: 12)
Brown, Alice, CRNA (MedNet Code: 509)

Billable MD providers:
Johnson, Robert, MD (MedNet Code: 1)
Williams, Sarah K, MD (MedNet Code: 5)
"""
        
        print("\n\nTesting with sample provider mapping...")
        responsible, md, crna = extract_and_match_providers(pdf_path, sample_mapping)
        print(f"\nResults:")
        print(f"  Responsible Provider: {responsible}")
        print(f"  MD Provider: {md}")
        print(f"  CRNA Provider: {crna}")
    else:
        print("Usage: python provider_annotation_utils.py <pdf_path>")

