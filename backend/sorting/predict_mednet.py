#!/usr/bin/env python3
"""
Production script for predicting MedNet codes for insurance companies.
Takes Primary/Secondary/Tertiary insurance info and predicts MedNet codes.
"""

import pandas as pd
import re
import json
import google.genai as genai
from google.genai import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_po_box(address):
    """Extract PO Box number from address using regex"""
    if pd.isna(address) or address == '':
        return None
    
    # Pattern to match PO BOX followed by numbers
    po_box_pattern = r'PO\s+BOX\s+(\d+)'
    match = re.search(po_box_pattern, str(address).upper())
    
    if match:
        return match.group(1)
    return None


def clean_address_for_matching(address):
    """Clean address for better matching"""
    if pd.isna(address) or address == '':
        return ''
    
    # Convert to uppercase and remove extra spaces
    cleaned = str(address).upper().strip()
    # Remove commas and semicolons for consistency
    cleaned = cleaned.replace(',', '').replace(';', '')
    # Normalize spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned


def use_ai_to_check_match(input_data, mednet_matches, client):
    """Use Gemini AI to check if a match is correct and provide confidence level"""
    
    # Create the prompt for AI matching
    if len(mednet_matches) == 1:
        # Single match case
        match = mednet_matches[0]
        prompt = f"""
You are a medical billing expert tasked with verifying if an insurance company match is correct.

INPUT DATA:
- Insurance Name: "{input_data['insurance_name']}"
- Insurance Address: "{input_data['address']}"

FOUND MATCH:
- MedNet Name: "{match['name']}"
- MedNet Code: "{match['mednet_code']}"
- MedNet Address: "{match['address']}"

TASK:
Determine if this is the correct match and your confidence level.

Consider:
1. Name similarity (exact matches are best, but consider variations, abbreviations, subsidiaries)
2. Address similarity (city, state, zip code patterns)
3. Business context (same company but different divisions/locations)

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{{
    "confidence": "<high|low>",
    "mednet_code": "{match['mednet_code']}"
}}

CRITICAL: Only use HIGH confidence if you are ABSOLUTELY CERTAIN this is the right match.
- HIGH confidence: Exact name match or very clear business relationship
- LOW confidence: Any uncertainty, partial matches, or when you're not 100% sure
- When in doubt, choose LOW confidence

IMPORTANT WARNINGS:
- EVERCARE and MEDICARE are COMPLETELY DIFFERENT companies - do NOT confuse them
- EVERCARE is a separate insurance company, not related to Medicare
- If you see "MEDICARE" in the insurance name, NEVER match it to EVERCARE
"""
    else:
        # Multiple matches case
        prompt = f"""
You are a medical billing expert tasked with matching insurance companies to their correct MedNet codes.

INPUT DATA:
- Insurance Name: "{input_data['insurance_name']}"
- Insurance Address: "{input_data['address']}"

AVAILABLE MATCHES (all have the same PO Box number):
"""
        
        for i, match in enumerate(mednet_matches, 1):
            prompt += f"""
Option {i}:
- MedNet Name: "{match['name']}"
- MedNet Code: "{match['mednet_code']}"
- MedNet Address: "{match['address']}"
"""
        
        prompt += """
TASK:
Compare the input insurance name and address with each available MedNet option and select the BEST MATCH.

Consider:
1. Name similarity (exact matches are best, but consider variations, abbreviations, subsidiaries)
2. Address similarity (city, state, zip code patterns)
3. Business context (same company but different divisions/locations)

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{
    "selected_option": <number>,
    "confidence": "<high|low>",
    "mednet_code": "<code>"
}

Where selected_option is the number (1, 2, 3, etc.) of the best match, and mednet_code is the code for that option.

CRITICAL: Only use HIGH confidence if you are ABSOLUTELY CERTAIN this is the right match.
- HIGH confidence: Exact name match or very clear business relationship
- LOW confidence: Any uncertainty, partial matches, or when you're not 100% sure
- When in doubt, choose LOW confidence

IMPORTANT WARNINGS:
- EVERCARE and MEDICARE are COMPLETELY DIFFERENT companies - do NOT confuse them
- EVERCARE is a separate insurance company, not related to Medicare
- If you see "MEDICARE" in the insurance name, NEVER match it to EVERCARE
"""
    
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
        )

        # Get AI response
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=generate_content_config,
        )
        
        response_text = response.text.strip()
        
        # Clean the response by removing markdown code block formatting
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith('```'):
            response_text = response_text[3:]   # Remove ```
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove trailing ```
        response_text = response_text.strip()
        
        # Parse the JSON response
        ai_decision = json.loads(response_text)
        
        confidence = ai_decision.get('confidence', 'low')
        mednet_code = ai_decision.get('mednet_code', None)
        
        if len(mednet_matches) == 1:
            # Single match case
            return {
                'mednet_code': mednet_code if confidence == 'high' else None,
                'confidence': confidence
            }
        else:
            # Multiple matches case
            selected_option = ai_decision.get('selected_option', 1)
            
            # Validate selected option is within range
            if 1 <= selected_option <= len(mednet_matches):
                return {
                    'mednet_code': mednet_code if confidence == 'high' else None,
                    'confidence': confidence
                }
            else:
                # Fallback if AI selection is invalid
                return {
                    'mednet_code': None,
                    'confidence': 'low'
                }
            
    except Exception as e:
        logger.error(f"AI matching failed: {str(e)}")
        return {
            'mednet_code': None,
            'confidence': 'low'
        }


def predict_mednet_code(insurance_name, insurance_address, po_box_to_mednet, special_cases, client):
    """Predict MedNet code for a single insurance entry"""
    
    # Skip if no insurance name
    if pd.isna(insurance_name) or insurance_name == '':
        return None
    
    # Check special cases first (before address validation)
    insurance_name_upper = str(insurance_name).upper().strip()
    if insurance_name_upper in special_cases:
        logger.info(f"Special case match for: {insurance_name}")
        return special_cases[insurance_name_upper]
    
    # Now check if address is present for regular matching
    if pd.isna(insurance_address) or insurance_address == '':
        logger.info(f"No address provided for: {insurance_name}")
        return None
    
    # Extract PO Box
    po_box = extract_po_box(insurance_address)
    
    if not po_box:
        logger.info(f"No PO Box found for: {insurance_name}")
        return None
    
    if po_box not in po_box_to_mednet:
        logger.info(f"PO Box {po_box} not found in MedNet database for: {insurance_name}")
        return None
    
    matches = po_box_to_mednet[po_box]
    
    # Prepare input data for AI
    input_data = {
        'insurance_name': insurance_name,
        'address': insurance_address
    }
    
    # Use AI to check the match
    ai_result = use_ai_to_check_match(input_data, matches, client)
    
    if ai_result['confidence'] == 'high' and ai_result['mednet_code']:
        logger.info(f"AI matched {insurance_name} to MedNet code {ai_result['mednet_code']}")
        return ai_result['mednet_code']
    else:
        logger.info(f"Low confidence for {insurance_name}, not mapping")
        return None


def process_insurance_predictions(input_csv, mednet_csv, output_csv, special_cases_csv=None, max_workers=10):
    """
    Main function to predict MedNet codes for primary, secondary, and tertiary insurance.
    
    Args:
        input_csv: Path to input CSV with patient insurance data
        mednet_csv: Path to MedNet codes CSV
        output_csv: Path to save output CSV with predicted codes
        special_cases_csv: Optional path to CSV with special case mappings (Company name, Mednet code)
        max_workers: Number of concurrent workers for AI predictions
    """
    
    logger.info("Loading data files...")
    
    # Load input CSV
    df = pd.read_csv(input_csv, dtype=str)
    logger.info(f"Loaded input CSV: {len(df)} rows")
    
    # Load MedNet CSV
    mednet_df = pd.read_csv(mednet_csv, dtype=str)
    logger.info(f"Loaded MedNet CSV: {len(mednet_df)} rows")
    
    # Load special cases if provided
    special_cases = {}
    if special_cases_csv:
        try:
            special_df = pd.read_csv(special_cases_csv, dtype=str)
            if 'Company name' in special_df.columns and 'Mednet code' in special_df.columns:
                for _, row in special_df.iterrows():
                    company_name = str(row['Company name']).upper().strip()
                    mednet_code = str(row['Mednet code']).strip()
                    special_cases[company_name] = mednet_code
                logger.info(f"Loaded {len(special_cases)} special case mappings")
            else:
                logger.warning("Special cases CSV missing required columns: 'Company name' and 'Mednet code'")
        except Exception as e:
            logger.error(f"Failed to load special cases CSV: {str(e)}")
    
    # Create a mapping of PO Box numbers to MedNet codes
    po_box_to_mednet = defaultdict(list)
    
    for _, row in mednet_df.iterrows():
        po_box = extract_po_box(row.get('Address', ''))
        if po_box:
            po_box_to_mednet[po_box].append({
                'mednet_code': row.get('MedNet Code', ''),
                'name': row.get('Name', ''),
                'address': row.get('Address', '')
            })
    
    logger.info(f"Created PO Box mapping for {len(po_box_to_mednet)} unique PO Boxes")
    
    # Initialize Google AI client
    client = genai.Client(
        api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
    )
    
    # Add MedNet code columns if they don't exist
    if 'Primary Mednet Code' not in df.columns:
        df['Primary Mednet Code'] = ''
    if 'Secondary Mednet Code' not in df.columns:
        df['Secondary Mednet Code'] = ''
    if 'Tertiary Mednet Code' not in df.columns:
        df['Tertiary Mednet Code'] = ''
    
    # Process primary insurance
    logger.info("Processing primary insurance...")
    primary_tasks = []
    for idx, row in df.iterrows():
        insurance_name = row.get('Primary Company Name', '')
        insurance_address = row.get('Primary Company Address 1', '')
        primary_tasks.append((idx, insurance_name, insurance_address))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                predict_mednet_code, 
                name, 
                address, 
                po_box_to_mednet, 
                special_cases, 
                client
            ): idx 
            for idx, name, address in primary_tasks
        }
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                mednet_code = future.result()
                if mednet_code:
                    df.at[idx, 'Primary Mednet Code'] = mednet_code
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Primary: Processed {completed}/{len(primary_tasks)}")
            except Exception as e:
                logger.error(f"Error processing primary insurance at row {idx}: {str(e)}")
    
    # Process secondary insurance
    logger.info("Processing secondary insurance...")
    secondary_tasks = []
    for idx, row in df.iterrows():
        insurance_name = row.get('Secondary Company Name', '')
        insurance_address = row.get('Secondary Company Address 1', '')
        if insurance_name or insurance_address:
            secondary_tasks.append((idx, insurance_name, insurance_address))
    
    if secondary_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    predict_mednet_code, 
                    name, 
                    address, 
                    po_box_to_mednet, 
                    special_cases, 
                    client
                ): idx 
                for idx, name, address in secondary_tasks
            }
            
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    mednet_code = future.result()
                    if mednet_code:
                        df.at[idx, 'Secondary Mednet Code'] = mednet_code
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Secondary: Processed {completed}/{len(secondary_tasks)}")
                except Exception as e:
                    logger.error(f"Error processing secondary insurance at row {idx}: {str(e)}")
    
    # Process tertiary insurance
    logger.info("Processing tertiary insurance...")
    tertiary_tasks = []
    for idx, row in df.iterrows():
        insurance_name = row.get('Tertiary Company Name', '')
        insurance_address = row.get('Tertiary Company Address 1', '')
        if insurance_name or insurance_address:
            tertiary_tasks.append((idx, insurance_name, insurance_address))
    
    if tertiary_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    predict_mednet_code, 
                    name, 
                    address, 
                    po_box_to_mednet, 
                    special_cases, 
                    client
                ): idx 
                for idx, name, address in tertiary_tasks
            }
            
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    mednet_code = future.result()
                    if mednet_code:
                        df.at[idx, 'Tertiary Mednet Code'] = mednet_code
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Tertiary: Processed {completed}/{len(tertiary_tasks)}")
                except Exception as e:
                    logger.error(f"Error processing tertiary insurance at row {idx}: {str(e)}")
    
    # Save output
    logger.info(f"Saving results to {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    primary_mapped = (df['Primary Mednet Code'] != '').sum()
    secondary_mapped = (df['Secondary Mednet Code'] != '').sum()
    tertiary_mapped = (df['Tertiary Mednet Code'] != '').sum()
    
    logger.info("=" * 60)
    logger.info("SUMMARY RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total rows processed: {len(df)}")
    logger.info(f"Primary insurance mapped: {primary_mapped}/{len(df)} ({primary_mapped/len(df)*100:.1f}%)")
    if secondary_tasks:
        logger.info(f"Secondary insurance mapped: {secondary_mapped}/{len(secondary_tasks)} ({secondary_mapped/len(secondary_tasks)*100:.1f}%)")
    if tertiary_tasks:
        logger.info(f"Tertiary insurance mapped: {tertiary_mapped}/{len(tertiary_tasks)} ({tertiary_mapped/len(tertiary_tasks)*100:.1f}%)")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python predict_mednet.py <input_csv> <mednet_csv> <output_csv> [special_cases_csv] [max_workers]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    mednet_csv = sys.argv[2]
    output_csv = sys.argv[3]
    special_cases_csv = sys.argv[4] if len(sys.argv) > 4 else None
    max_workers = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    
    process_insurance_predictions(input_csv, mednet_csv, output_csv, special_cases_csv, max_workers)

