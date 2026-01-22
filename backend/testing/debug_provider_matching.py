#!/usr/bin/env python3
"""
Debug script to understand provider matching logic.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from provider_annotation_utils import (
    extract_annotations_from_pdf,
    parse_provider_mapping,
    match_providers_from_annotations
)

# Sample provider mapping
sample_mapping = """Billable CRNA providers:
Smith, John, CRNA (MedNet Code: 7)
Doe, Jane M, CRNA (MedNet Code: 12)
Brown, Alice, CRNA (MedNet Code: 509)

Billable MD providers:
Johnson, Robert, MD (MedNet Code: 1)
Williams, Sarah K, MD (MedNet Code: 5)
"""

# Test PDF
pdf_path = "testing/INJE.pdf"

print("=" * 80)
print("DEBUGGING PROVIDER EXTRACTION")
print("=" * 80)

# Step 1: Extract annotations
print("\n1. Extracting annotations from PDF...")
annotations = extract_annotations_from_pdf(pdf_path)
print(f"   Found {len(annotations)} annotations:")
for i, text in enumerate(annotations, 1):
    print(f"   {i:2d}. '{text}'")

# Step 2: Parse provider mapping
print("\n2. Parsing provider mapping...")
provider_mapping = parse_provider_mapping(sample_mapping)
print(f"   Found {len(provider_mapping)} providers:")
for code, info in provider_mapping.items():
    print(f"   Code {code:3s} -> {info['name']} ({info['title']})")

# Step 3: Test matching logic manually
print("\n3. Testing matching logic for each annotation...")
import re

for text in annotations:
    text = text.strip()
    
    # Check if it matches single number pattern
    match_one = re.match(r'^(\d+)$', text)
    if match_one:
        code = match_one.group(1).lstrip('0') or '0'
        provider = provider_mapping.get(code)
        
        if provider:
            print(f"   ✅ '{text}' -> Code {code} -> {provider['name']} ({provider['title']})")
        else:
            print(f"   ❌ '{text}' -> Code {code} -> NOT FOUND in mapping")
    else:
        # Check if it matches two-number pattern
        match_two = re.match(r'^(\d+)\s*/\s*(\d+)$', text)
        if match_two:
            code1 = match_two.group(1).lstrip('0') or '0'
            code2 = match_two.group(2).lstrip('0') or '0'
            print(f"   📋 '{text}' -> Two codes: {code1}, {code2}")

# Step 4: Run the actual matching function
print("\n4. Running actual matching function...")
responsible, md, crna = match_providers_from_annotations(annotations, provider_mapping)
print(f"   Responsible Provider: {responsible}")
print(f"   MD Provider: {md}")
print(f"   CRNA Provider: {crna}")

print("\n" + "=" * 80)
print("EXPECTED RESULT:")
print("   Responsible Provider: Brown, Alice, CRNA")
print("   MD Provider: None")
print("   CRNA Provider: Brown, Alice, CRNA")
print("=" * 80)
