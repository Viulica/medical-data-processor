#!/usr/bin/env python3
"""
Python script to replicate Excel macro functionality for data conversion.
This script processes CSV data according to hardcoded mapping rules.
"""

import pandas as pd
import re
import sys
from pathlib import Path
import os
import json
import google.genai as genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import export_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from export_utils import save_dataframe_dual_format


def find_header(df, header_name):
    """
    Find header by partial/case-insensitive match (returns column name, or None if not found)
    Replicates the FindHeader function from the Excel macro.
    """
    header_name = header_name.lower().strip()
    for col in df.columns:
        if header_name in col.lower().strip():
            return col
    return None


def split_patient_name(patient_name, target_field):
    """
    Split patient name into components based on target field.
    Handles both "Last, First" and "First Last" formats.
    """
    if not patient_name or pd.isna(patient_name):
        return ""
    
    patient_name = str(patient_name).strip()
    
    if "last" in target_field.lower():
        # Extract last name
        if "," in patient_name:
            # Format: "Last, First" - take part before comma
            parts = patient_name.split(",")
            return parts[0].strip()
        else:
            # Format: "First Last" or "First Middle Last" - take last word
            parts = patient_name.split()
            if len(parts) > 0:
                return parts[-1].strip()
            return ""
    
    elif "first" in target_field.lower():
        # Extract first name
        if "," in patient_name:
            # Format: "Last, First" - take part after comma
            parts = patient_name.split(",")
            if len(parts) > 1:
                first_part = parts[1].strip()
                # If there are multiple names after comma, take the first word
                first_words = first_part.split()
                return first_words[0] if first_words else ""
            return ""
        else:
            # Format: "First Last" or "First Middle Last" - take first word
            parts = patient_name.split()
            if len(parts) > 1:
                # Only return first name if there are multiple parts (indicating first + last)
                return parts[0].strip()
            else:
                # Single word - assume it's a last name only, so no first name
                return ""
    
    elif "middle" in target_field.lower():
        # Extract middle name
        if "," in patient_name:
            # Format: "Last, First Middle" - extract middle from after comma
            parts = patient_name.split(",")
            if len(parts) > 1:
                after_comma = parts[1].strip()
                words = after_comma.split()
                # Look for middle name (single letter or second word)
                if len(words) > 1:
                    # Check if last word is single letter
                    if len(words[-1]) == 1 and words[-1].isupper() and words[-1].isalpha():
                        return words[-1]
                    # Otherwise return second word as middle name
                    return words[1] if len(words) > 1 else ""
                return ""
        else:
            # Format: "First Middle Last" - look for middle name
            parts = patient_name.split()
            if len(parts) >= 3:
                # If we have 3+ words, middle could be the second word or a single letter
                middle_candidate = parts[1]
                if len(middle_candidate) == 1 and middle_candidate.isupper() and middle_candidate.isalpha():
                    return middle_candidate
                # For compound names like "First Middle Last", return the middle word
                return middle_candidate
            elif len(parts) == 2:
                # Only First Last, no middle name
                return ""
            return ""
    
    else:
        # Return full name
        return patient_name
    
    return ""


def process_dob_gender_logic(df, row_idx, mapping, target_field, is_dob=True):
    """
    Process DOB or Gender with 'if self' logic based on relationship.
    Replicates the DOB and Gender logic from the Excel macro.
    """
    # Find relationship column
    rel_col = find_header(df, "Relationship")
    if not rel_col:
        rel_col = find_header(df, "Guarantor Relation")
    
    # Find patient and guarantor columns
    if is_dob:
        pat_col = find_header(df, "DOB")
        if not pat_col:
            pat_col = find_header(df, "Patient DOB")
        guar_col = find_header(df, "Guarantor DOB")
        if not guar_col:
            guar_col = find_header(df, "Subscriber DOB")
    else:
        pat_col = find_header(df, "Sex")
        if not pat_col:
            pat_col = find_header(df, "Gender")
        guar_col = find_header(df, "Guarantor Sex")
        if not guar_col:
            guar_col = find_header(df, "Subscriber Sex")
    
    if rel_col and not pd.isna(df.iloc[row_idx][rel_col]):
        rel_val = str(df.iloc[row_idx][rel_col]).lower().strip()
        if "self" in rel_val:
            if pat_col and not pd.isna(df.iloc[row_idx][pat_col]):
                return df.iloc[row_idx][pat_col]
        else:
            if guar_col and not pd.isna(df.iloc[row_idx][guar_col]):
                return df.iloc[row_idx][guar_col]
    
    # Fallback to patient value
    if pat_col and not pd.isna(df.iloc[row_idx][pat_col]):
        return df.iloc[row_idx][pat_col]
    
    return ""


def extract_icd_codes(text):
    """
    Extract ICD codes from text. Codes are in brackets like [E66.01] or [B18.1, K74.60].
    Returns a list of ICD codes.
    Only keeps codes that start with a letter (valid ICD codes).
    """
    if pd.isna(text) or not text:
        return []
    
    text = str(text)
    codes = []
    
    # Find all patterns in square brackets
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    
    for match in matches:
        # Split by comma in case there are multiple codes in one bracket
        codes_in_bracket = [code.strip() for code in match.split(',')]
        # Only keep codes that start with a letter (valid ICD codes)
        for code in codes_in_bracket:
            if code and code[0].isalpha():
                codes.append(code)
    
    return codes


def process_icd_reordering_task(args):
    """
    Process a single ICD reordering task (for threading).
    
    Args:
        args: Tuple of (row_idx, icd_codes, procedure, post_op_diagnosis, post_op_coded)
    
    Returns:
        Tuple of (row_idx, reordered_codes, success_flag, final_prompt, final_response)
    """
    row_idx, icd_codes, procedure, post_op_diagnosis, post_op_coded = args
    
    # Initialize client for this thread
    try:
        client = genai.Client(
            api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
        )
    except Exception as e:
        print(f"    ⚠️  Row {row_idx}: Could not initialize AI client: {str(e)}")
        return (row_idx, icd_codes, False, "", "")
    
    # Reorder codes
    reordered_codes, success_flag, final_prompt, final_response = reorder_icd_codes_with_ai(icd_codes, procedure, post_op_diagnosis, post_op_coded, client)
    return (row_idx, reordered_codes, success_flag, final_prompt, final_response)


def reorder_icd_codes_with_ai(icd_codes, procedure, post_op_diagnosis, post_op_coded, client=None):
    """
    Use Gemini AI to reorder ICD codes by relevance to the procedure.
    The primary diagnosis (reason for procedure) should come first.
    
    Args:
        icd_codes: List of ICD codes (up to 4)
        procedure: Procedure description string
        post_op_diagnosis: POST-OP DIAGNOSIS string (supplementary)
        post_op_coded: Post-op Diagnosis - Coded string (supplementary)
        client: Google AI client (optional, will create if not provided)
    
    Returns:
        Tuple of (reordered_codes, success_flag, final_prompt, final_response) where:
        - reordered_codes: List of reordered ICD codes
        - success_flag: Boolean indicating if AI reordering was successful
        - final_prompt: The exact prompt sent to the AI
        - final_response: The exact response received from the AI
    """
    # If no codes or only one code, no need to reorder
    if not icd_codes or len(icd_codes) <= 1:
        return (icd_codes, True, "", "")  # Success = True since no reordering needed
    
    # Initialize Google AI client if not provided
    if client is None:
        try:
            client = genai.Client(
                api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
            )
        except Exception as e:
            print(f"    ⚠️  Could not initialize AI client: {str(e)}")
            return (icd_codes, False, "", "")  # Return original order if AI fails
    
    # Prepare the prompt
    prompt = f"""
You are a medical coding expert tasked with ordering ICD diagnosis codes by relevance to a surgical procedure.

PROCEDURE:
{procedure if procedure and not pd.isna(procedure) else "Not specified"}

SUPPLEMENTARY INFORMATION:
Post-op Diagnosis: {post_op_diagnosis if post_op_diagnosis and not pd.isna(post_op_diagnosis) else "Not specified"}
Post-op Diagnosis - Coded: {post_op_coded if post_op_coded and not pd.isna(post_op_coded) else "Not specified"}

ICD CODES TO ORDER:
{', '.join(icd_codes)}

TASK:
Reorder these ICD codes by relevance to the procedure, where:
1. ICD1 should be the PRIMARY diagnosis (the main reason for the procedure)
2. ICD2 is often also related to the procedure itself (but mostly not)
3. ICD3 and ICD4 should be ordered by decreasing relevance to the procedure

Consider:
- Which diagnosis is the primary reason the procedure was performed?
- Which diagnoses are directly related to the surgical intervention?
- Which diagnoses are secondary or comorbid conditions?

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{{
    "ordered_codes": ["ICD1", "ICD2", "ICD3", "ICD4"],
    "reasoning": "Brief explanation of why you ordered the codes this way, focusing on why the primary diagnosis was chosen."
}}

Where:
- "ordered_codes" contains all {len(icd_codes)} codes in order of relevance (most relevant first)
- "reasoning" explains your decision-making process (1-3 sentences)

CRITICAL RULES:
1. Return ONLY valid JSON, no markdown formatting, no code blocks, no other text
2. Include ALL {len(icd_codes)} codes in "ordered_codes"
3. Use the EXACT code strings provided above (preserve case and format)
4. Your entire response must be parseable as JSON
"""
    
    # Retry logic - attempt up to 5 times
    max_retries = 5
    for attempt in range(max_retries):
        raw_response_text = ""  # Initialize to handle early exceptions
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            tools = [
                types.Tool(googleSearch=types.GoogleSearch(
                )),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,  # Lower temperature for more consistent output
                tools=tools,
            )

            # Get AI response
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=contents,
                config=generate_content_config,
            )
            
            # Capture the RAW response exactly as returned by the model
            raw_response_text = response.text
            
            # Now clean the response for parsing
            response_text = response.text.strip()
            
            # Clean the response by removing any markdown code block formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove trailing ```
            response_text = response_text.strip()
            
            # Parse the JSON response
            ai_decision = json.loads(response_text)
            ordered_codes = ai_decision.get('ordered_codes', icd_codes)
            reasoning = ai_decision.get('reasoning', 'No reasoning provided')
            
            # Normalize codes for comparison (case-insensitive, strip whitespace)
            original_codes_normalized = [c.strip().upper() for c in icd_codes]
            ordered_codes_normalized = [c.strip().upper() for c in ordered_codes]
            
            # Validate that all original codes are present (case-insensitive comparison)
            if len(ordered_codes) == len(icd_codes) and set(ordered_codes_normalized) == set(original_codes_normalized):
                # Return the RAW response exactly as received from the model for logging
                return (ordered_codes, True, prompt, raw_response_text)  # Success!
            else:
                if attempt < max_retries - 1:
                    print(f"    ⚠️  AI returned invalid code list (attempt {attempt + 1}/{max_retries})")
                    print(f"        Expected: {icd_codes}")
                    print(f"        Received: {ordered_codes}")
                    continue
                else:
                    print(f"    ⚠️  AI returned invalid code list after {max_retries} attempts, using original order")
                    print(f"        Expected: {icd_codes}")
                    print(f"        Received: {ordered_codes}")
                    return (icd_codes, False, prompt, raw_response_text)
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️  AI reordering failed (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...")
                continue
            else:
                print(f"    ⚠️  AI reordering failed after {max_retries} attempts: {str(e)}")
                return (icd_codes, False, prompt, raw_response_text)  # Return original order and any captured response
    
    # This should never be reached, but just in case
    return (icd_codes, False, prompt, "")


def update_icd_codes_to_latest(icd_codes, procedure, post_op_diagnosis, post_op_coded, client=None):
    """
    Use Gemini AI with web search to update ICD codes to their latest versions.
    Verifies each code is the most current version for the specific diagnosis.
    Maintains the original order and diagnoses - only updates code versions.
    
    Args:
        icd_codes: List of ICD codes (up to 4) - these will be updated to latest versions
        procedure: Procedure description string
        post_op_diagnosis: POST-OP DIAGNOSIS string (supplementary)
        post_op_coded: Post-op Diagnosis - Coded string (supplementary)
        client: Google AI client (optional, will create if not provided)
    
    Returns:
        Tuple of (updated_codes, success_flag, final_prompt, final_response) where:
        - updated_codes: List of updated ICD codes in the same order as input
        - success_flag: Boolean indicating if AI update was successful
        - final_prompt: The exact prompt sent to the AI
        - final_response: The exact response received from the AI
    """
    # If no codes, return empty list
    if not icd_codes or len(icd_codes) == 0:
        return (icd_codes, True, "", "")
    
    # Initialize Google AI client if not provided
    if client is None:
        try:
            client = genai.Client(
                api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
            )
        except Exception as e:
            print(f"    ⚠️  Could not initialize AI client: {str(e)}")
            return (icd_codes, False, "", "")
    
    # Prepare the prompt with emphasis on using Google search to verify latest versions
    prompt = f"""
You are a medical coding expert tasked with updating ICD diagnosis codes to their LATEST CURRENT VERSIONS.

PROCEDURE:
{procedure if procedure and not pd.isna(procedure) else "Not specified"}

SUPPLEMENTARY INFORMATION:
Post-op Diagnosis: {post_op_diagnosis if post_op_diagnosis and not pd.isna(post_op_diagnosis) else "Not specified"}
Post-op Diagnosis - Coded: {post_op_coded if post_op_coded and not pd.isna(post_op_coded) else "Not specified"}

CURRENT ICD CODES (in order):
{chr(10).join([f"{i+1}. {code}" for i, code in enumerate(icd_codes)])}

TASK:
For EACH ICD code above, you must:
1. Use Google Search to verify if the code is the LATEST CURRENT VERSION for that specific diagnosis
2. If a newer version exists, replace it with the latest version
3. If the code is already the latest version, keep it unchanged
4. MAINTAIN THE EXACT SAME ORDER as provided above
5. DO NOT change the diagnosis - only update the code version if outdated

IMPORTANT RULES:
- Use Google Search to check the latest ICD-10-CM code versions for each diagnosis
- Preserve the order: Code 1 stays first, Code 2 stays second, etc.
- Only update codes that have newer versions available
- If a code is already current, return it exactly as provided
- Do NOT reorder codes or change diagnoses
- Do NOT add or remove codes

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{{
    "updated_codes": ["ICD1", "ICD2", "ICD3", "ICD4"]
}}

Where:
- "updated_codes" contains all {len(icd_codes)} codes in THE SAME ORDER as input (updated to latest versions)

CRITICAL RULES:
1. Return ONLY valid JSON, no markdown formatting, no code blocks, no other text
2. Include ALL {len(icd_codes)} codes in "updated_codes" in THE SAME ORDER
3. Use Google Search to verify each code is the latest version
4. If a code is already current, return it exactly as provided (preserve format/case)
5. Your entire response must be parseable as JSON
"""
    
    # Retry logic - attempt up to 5 times
    max_retries = 5
    for attempt in range(max_retries):
        raw_response_text = ""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            tools = [
                types.Tool(googleSearch=types.GoogleSearch(
                )),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,  # Lower temperature for more consistent/accurate code updates
                tools=tools,
            )

            # Get AI response using gemini-3-flash-preview with web search
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=contents,
                config=generate_content_config,
            )
            
            # Capture the RAW response exactly as returned by the model
            raw_response_text = response.text
            
            # Now clean the response for parsing
            response_text = response.text.strip()
            
            # Clean the response by removing any markdown code block formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove trailing ```
            response_text = response_text.strip()
            
            # Parse the JSON response
            ai_decision = json.loads(response_text)
            updated_codes = ai_decision.get('updated_codes', icd_codes)
            
            # Validate that we have the same number of codes
            if len(updated_codes) != len(icd_codes):
                if attempt < max_retries - 1:
                    print(f"    ⚠️  AI returned wrong number of codes (attempt {attempt + 1}/{max_retries})")
                    print(f"        Expected: {len(icd_codes)} codes")
                    print(f"        Received: {len(updated_codes)} codes")
                    continue
                else:
                    print(f"    ⚠️  AI returned wrong number of codes after {max_retries} attempts, using original codes")
                    return (icd_codes, False, prompt, raw_response_text)
            
            # Success - return updated codes in same order
            return (updated_codes, True, prompt, raw_response_text)
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️  AI code update failed (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...")
                continue
            else:
                print(f"    ⚠️  AI code update failed after {max_retries} attempts: {str(e)}")
                return (icd_codes, False, prompt, raw_response_text)
    
    # This should never be reached, but just in case
    return (icd_codes, False, prompt, "")


def process_icd_update_task(args):
    """
    Process a single ICD update task (for threading).
    
    Args:
        args: Tuple of (row_idx, icd_codes, procedure, post_op_diagnosis, post_op_coded)
    
    Returns:
        Tuple of (row_idx, updated_codes, success_flag, final_prompt, final_response)
    """
    row_idx, icd_codes, procedure, post_op_diagnosis, post_op_coded = args
    
    # Initialize client for this thread
    try:
        client = genai.Client(
            api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
        )
    except Exception as e:
        print(f"    ⚠️  Row {row_idx}: Could not initialize AI client: {str(e)}")
        return (row_idx, icd_codes, False, "", "")
    
    # Update codes to latest versions
    updated_codes, success_flag, final_prompt, final_response = update_icd_codes_to_latest(icd_codes, procedure, post_op_diagnosis, post_op_coded, client)
    return (row_idx, updated_codes, success_flag, final_prompt, final_response)


def extract_mednet_code(value):
    """
    Extract only numeric part from Mednet Code.
    Replicates the Mednet Code processing logic from the Excel macro.
    """
    if pd.isna(value) or not value:
        return ""
    
    value = str(value)
    only_num = ""
    
    for char in value:
        if char.isdigit():
            only_num += char
        elif only_num:  # Stop after first numeric block
            break
    
    return only_num


def process_company_name(value):
    """
    Process company name: strip numeric code, keep only alphabetic part.
    Replicates the Company Name processing logic from the Excel macro.
    """
    if pd.isna(value) or not value:
        return ""
    
    value = str(value).strip()
    
    # If value contains "-", return everything after first dash
    if "-" in value:
        dash_pos = value.find("-")
        return value[dash_pos + 1:].strip()
    else:
        # Fallback: return as-is if no dash found
        return value


def extract_phone_by_type(phone_text, phone_type):
    """
    Extract phone number by type from a text that may contain multiple phone numbers.
    phone_type: "home", "work", or "mobile"
    Returns only numeric digits.
    """
    if pd.isna(phone_text) or not phone_text:
        return ""
    
    phone_text = str(phone_text).strip()
    
    # Split by newlines to get individual phone entries
    phone_entries = phone_text.split('\n')
    
    for entry in phone_entries:
        entry = entry.strip()
        if not entry:
            continue
            
        # Check for phone type indicators
        if phone_type == "home" and ("Home Phone" in entry or "Home" in entry or "(Home Phone)" in entry):
            # Extract only numeric digits
            phone = ''.join(filter(str.isdigit, entry))
            return phone
        elif phone_type == "work" and ("Work Phone" in entry or "Work" in entry or "(Work Phone)" in entry):
            # Extract only numeric digits
            phone = ''.join(filter(str.isdigit, entry))
            return phone
        elif phone_type == "mobile" and ("Mobile" in entry or "Cell" in entry or "(Mobile)" in entry):
            # Extract only numeric digits
            phone = ''.join(filter(str.isdigit, entry))
            return phone
    
    # If no specific type found, return empty
    return ""


def extract_time_as_number(time_str):
    """
    Extract time as a numeric value for comparison.
    Normalizes to 4-digit format: "31" -> 31, "331" -> 331, "1430" -> 1430, "0200" -> 200.
    Returns None if time cannot be extracted.
    """
    if not time_str or pd.isna(time_str):
        return None
    
    time_str = str(time_str).strip()
    
    # If it's all digits, extract the numeric value
    if time_str.isdigit():
        # Return as integer - this works because:
        # - "31" = 31 (00:31)
        # - "331" = 331 (03:31)
        # - "1430" = 1430 (14:30)
        # - "0200" = 200 (02:00) - but this is fine for comparison
        # Comparison: 2200 > 200 (correct for rollover), 1430 > 31 (correct, same day)
        return int(time_str)
    
    # If it has a colon (already formatted), extract HHMM
    if ':' in time_str:
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                hours = int(parts[0])
                minutes = int(parts[1])
                return hours * 100 + minutes  # Convert to HHMM format
        except:
            return None
    
    return None


def format_anesthesia_time(df, row_idx, time_value, time_type):
    """
    Format anesthesia time by converting 2, 3, or 4-digit time to HH:MM format
    and combine with the date from the same row.
    Handles multi-day scenarios intelligently by detecting time rollovers.
    time_type: "start" or "stop"
    """
    if pd.isna(time_value) or not time_value:
        return ""
    
    time_str = str(time_value).strip()
    
    # Handle different digit lengths
    if time_str.isdigit():
        if len(time_str) == 1:
            # Convert 2 to 00:02
            formatted_time = f"00:0{time_str}"
        elif len(time_str) == 2:
            # Convert 31 to 00:31
            formatted_time = f"00:{time_str}"
        elif len(time_str) == 3:
            # Convert 331 to 03:31
            formatted_time = f"0{time_str[0]}:{time_str[1:]}"
        elif len(time_str) == 4:
            # Convert 1331 to 13:31
            formatted_time = f"{time_str[:2]}:{time_str[2:]}"
        else:
            # If not 1, 2, 3, or 4 digits, return as-is
            formatted_time = time_str
    else:
        # If not all digits, return as-is
        formatted_time = time_str
    
    # Get the date from the same row
    date_value = df.iloc[row_idx].get('Date', '')
    if pd.isna(date_value) or not date_value:
        return formatted_time
    
    # Parse the base date intelligently
    base_date = parse_date_intelligently(str(date_value).strip())
    if not base_date:
        return formatted_time
    
    # For stop times, check if we need to roll over to the next day
    if time_type == "stop":
        # Get the start time from the same row to detect rollover
        start_time_value = df.iloc[row_idx].get('An Start', '')
        if not pd.isna(start_time_value) and start_time_value:
            start_time_str = str(start_time_value).strip()
            
            # Simple comparison: if An Start > An Stop (as 4-digit numbers), it's a rollover
            # Extract numeric values for comparison
            start_numeric = extract_time_as_number(start_time_str)
            stop_numeric = extract_time_as_number(time_str)
            
            # If start > stop, it means stop is on the next day
            if start_numeric is not None and stop_numeric is not None and start_numeric > stop_numeric:
                # Add one day to the base date
                base_date = add_one_day(base_date)
    
    # Format the final date-time string
    date_str = format_date_for_output(base_date)
    return f"{date_str} {formatted_time}"


def format_time_only(time_value):
    """
    Format time value to HH:MM format only (no date).
    Returns time in HH:MM format for comparison purposes.
    """
    if pd.isna(time_value) or not time_value:
        return "00:00"
    
    time_str = str(time_value).strip()
    
    if time_str.isdigit():
        if len(time_str) == 1:
            return f"00:0{time_str}"
        elif len(time_str) == 2:
            return f"00:{time_str}"
        elif len(time_str) == 3:
            return f"0{time_str[0]}:{time_str[1:]}"
        elif len(time_str) == 4:
            return f"{time_str[:2]}:{time_str[2:]}"
    
    # If already has colon, try to extract time part
    if ':' in time_str:
        return time_str
    
    return "00:00"


def is_time_rollover(start_time, stop_time):
    """
    Check if stop time represents a rollover to the next day.
    Returns True if stop_time is earlier than start_time (indicating next day).
    """
    try:
        # Convert times to minutes since midnight for easy comparison
        start_minutes = time_to_minutes(start_time)
        stop_minutes = time_to_minutes(stop_time)
        
        # If stop time is significantly earlier than start time, it's likely next day
        # Use a threshold of 4 hours (240 minutes) to avoid false positives
        if stop_minutes < start_minutes and (start_minutes - stop_minutes) > 240:
            return True
        
        return False
    except:
        return False


def time_to_minutes(time_str):
    """
    Convert HH:MM time string to minutes since midnight.
    """
    try:
        if ':' in time_str:
            hours, minutes = time_str.split(':')
            return int(hours) * 60 + int(minutes)
        return 0
    except:
        return 0


def parse_date_intelligently(date_str):
    """
    Parse date string in various formats.
    Returns a datetime object or None if parsing fails.
    Supports: MM/DD/YYYY, MM/DD/YY, YYYY-MM-DD, YYYY-MM-DD HH:MM:SS
    """
    from datetime import datetime
    
    if not date_str:
        return None
    
    date_str = str(date_str).strip()
    
    # List of formats to try, in order of preference
    formats = [
        '%m/%d/%Y',           # MM/DD/YYYY
        '%m/%d/%y',           # MM/DD/YY
        '%Y-%m-%d',           # YYYY-MM-DD
        '%Y-%m-%d %H:%M:%S',  # YYYY-MM-DD HH:MM:SS
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If all parsing fails, return None
    return None


def add_one_day(date_obj):
    """
    Add one day to a datetime object, handling month/year rollovers.
    """
    from datetime import timedelta
    
    try:
        return date_obj + timedelta(days=1)
    except:
        # Fallback: return original date if addition fails
        return date_obj


def format_date_for_output(date_obj):
    """
    Format datetime object back to MM/DD/YYYY string for output.
    """
    try:
        return date_obj.strftime('%m/%d/%Y')
    except:
        # Fallback: return original date as string
        return str(date_obj)


def fix_time_format(time_str):
    """
    Fix malformed time formats by converting them to HH:MM format.
    
    Args:
        time_str: Time string that may be malformed (e.g., "1534", "734", "34")
    
    Returns:
        Properly formatted time string (e.g., "15:34", "07:34", "00:34")
    """
    if not time_str or pd.isna(time_str):
        return time_str
    
    time_str = str(time_str).strip()
    
    # If it already has a colon, it's probably already formatted correctly
    if ':' in time_str:
        return time_str
    
    # Check if it's all digits
    if not time_str.isdigit():
        return time_str
    
    # Handle different digit lengths
    if len(time_str) == 2:
        # Convert "34" to "00:34"
        return f"00:{time_str}"
    elif len(time_str) == 3:
        # Convert "734" to "07:34"
        return f"0{time_str[0]}:{time_str[1:]}"
    elif len(time_str) == 4:
        # Convert "1534" to "15:34"
        return f"{time_str[:2]}:{time_str[2:]}"
    else:
        # If not 2, 3, or 4 digits, return as-is
        return time_str


def fix_concurrent_providers_dates(concurrent_providers_value, charge_date):
    """
    Fix Concurrent Providers format by adding date prefix to times that don't have it
    and fixing malformed time formats. Also filters out any records containing "CRNA".
    
    Args:
        concurrent_providers_value: The Concurrent Providers string
        charge_date: The Charge Date value to use for missing dates
    
    Returns:
        Fixed Concurrent Providers string with dates added where needed, times formatted correctly,
        and CRNA records removed
    """
    if not concurrent_providers_value or pd.isna(concurrent_providers_value):
        return concurrent_providers_value
    
    if not charge_date or pd.isna(charge_date):
        return concurrent_providers_value
    
    # Convert charge_date to the format used in Concurrent Providers (MM/DD/YY)
    try:
        # Parse the charge date (could be in various formats)
        charge_date_str = str(charge_date).strip()
        
        # Try to parse as datetime
        if '/' in charge_date_str:
            # Format: M/D/YYYY or MM/DD/YYYY
            parts = charge_date_str.split('/')
            if len(parts) == 3:
                month, day, year = parts
                # Convert to MM/DD/YY format
                date_prefix = f"{month.zfill(2)}/{day.zfill(2)}/{year[-2:]}"
            else:
                return concurrent_providers_value
        else:
            # Can't parse the date, return as-is
            return concurrent_providers_value
    except:
        return concurrent_providers_value
    
    # Split by pipe to get individual provider entries
    provider_entries = concurrent_providers_value.split('|')
    fixed_entries = []
    
    for entry in provider_entries:
        # Skip entries that contain "CRNA"
        if "CRNA" in entry:
            continue
        
        # Each entry format: Name;Role;Time1;Time2
        parts = entry.split(';')
        
        if len(parts) >= 4:
            name = parts[0]
            role = parts[1]
            time1 = parts[2]
            time2 = parts[3]
            
            # Check if time1 already has a date (contains '/')
            if '/' in time1:
                # Extract date and time parts
                date_time_parts = time1.split(' ', 1)
                if len(date_time_parts) == 2:
                    date_part = date_time_parts[0]
                    time_part = date_time_parts[1]
                    # Fix the time part and recombine
                    fixed_time_part = fix_time_format(time_part)
                    time1 = f"{date_part} {fixed_time_part}"
                else:
                    # If no space, just fix the time format
                    time1 = fix_time_format(time1)
            else:
                # No date, fix time format and add date prefix
                time1 = fix_time_format(time1)
                time1 = f"{date_prefix} {time1}"
            
            # Check if time2 already has a date (contains '/')
            if '/' in time2:
                # Extract date and time parts
                date_time_parts = time2.split(' ', 1)
                if len(date_time_parts) == 2:
                    date_part = date_time_parts[0]
                    time_part = date_time_parts[1]
                    # Fix the time part and recombine
                    fixed_time_part = fix_time_format(time_part)
                    time2 = f"{date_part} {fixed_time_part}"
                else:
                    # If no space, just fix the time format
                    time2 = fix_time_format(time2)
            else:
                # No date, fix time format and add date prefix
                time2 = fix_time_format(time2)
                time2 = f"{date_prefix} {time2}"
            
            # Reconstruct the entry
            fixed_entry = f"{name};{role};{time1};{time2}"
            fixed_entries.append(fixed_entry)
        else:
            # Entry doesn't have expected format, keep as-is
            fixed_entries.append(entry)
    
    return '|'.join(fixed_entries)


def get_header_mapping():
    """
    Returns the hardcoded header mapping from old headers to new headers.
    This replaces the need for a mapping file.
    """
    return {
            # Date and Patient Info
            "Date": "Charge Date",
            "Patient Name": "Patient Last Name",  # Will be split into Last, First, Middle
            "MRN": "Patient MRN",
            "Location": "Location",
        "Procedure": "Procedure Description",
        "POST-OP DIAGNOSIS": "POST-OP DIAGNOSIS",
        "Post-op Diagnosis - Coded": "POST-OP DIAGNOSIS",
        "Responsible Provider": "Responsible Provider",
        "Concurrent Providers": "Concurrent Providers",
        "An Start": "An Start",
        "An Stop": "An Stop",
        
        # Patient Address
        "Street Address": "Patient Street Address",
        "City": "Patient City",
        "State": "Patient State",
        "ZIP Code": "Patient ZIP Code",
        "SSN": "Patient SSN",
        "DOB": "Patient DOB",
        "Sex": "Patient Sex",
        "Marital Status": "Patient Marital Status",
        "Pt. E-mail Address": "Patient E-mail Address",
        
        # Guarantor Info
        "Guarantor First": "Guarantor First Name",
        "Guarantor Last": "Guarantor Last Name",
        "Guarantor Middle": "Guarantor Middle Name",
        "Guarantor Relation": "Guarantor Relation",
        "Guarantor Address": "Guarantor Address",
        "Guarantor City": "Guarantor City",
        "Guarantor State": "Guarantor State",
        "Guarantor ZIP": "Guarantor ZIP",
        
        # Surgeon and Provider Info
        "Surgeons": "Surgeon",
        
        # Primary Insurance
        "Primary Plan": "Primary Company Name",
        "Primary Subsc ID": "Primary Sub ID",
        "Primary CVG Sub Name": "Primary Sub Name",
        "Primary Cvg Mem Rel to Sub": "Primary Cvg Mem Rel to Sub",
        "Primary Cvg Sub Address": "Primary Company Address 1",
        "Primary Cvg Sub City": "Primary Company City",
        "Primary Cvg Sub State": "Primary Company State",
        "Primary CVG Sub ZIP": "Primary Company ZIP",
        "Primary CVG Group Num": "Primary Sub Group Num",
        "Cvg 1 Auth Num": "Primary Sub Auth Num",
        "Primary CVG Address 1": "Primary Company Address 1",
        "Primary CVG City": "Primary Company City",
        "Primary CVG State": "Primary Company State",
        "Primary CVG ZIP": "Primary Company ZIP",
        
        # Secondary Insurance
        "Secondary Plan": "Secondary Company Name",
        "Secondary Subsc ID": "Secondary Sub ID",
        "Secondary CVG Sub Name": "Secondary Sub Name",
        "Secondary Cvg Mem Rel to Sub": "Secondary Cvg Mem Rel to Sub",
        "Secondary Cvg Sub Address": "Secondary Company Address 1",
        "Secondary Cvg Sub City": "Secondary Company City",
        "Secondary Cvg Sub State": "Secondary Company State",
        "Secondary CVG Sub ZIP": "Secondary Company ZIP",
        "Secondary CVG Group Num": "Secondary Sub Group Num",
        "Cvg 2 Auth Num": "Secondary Auth Num",
        "Secondary CVG Address 1": "Secondary Company Address 1",
        "Secondary CVG City": "Secondary Company City",
        "Secondary CVG State": "Secondary Company State",
        "Secondary CVG ZIP": "Secondary Company ZIP",
        
        # Tertiary Insurance
        "Tertiary Plan": "Tertiary Company Name",
        "Tertiary Subsc ID": "Tertiary Sub ID",
        "Tertiary Cvg Mem Rel to Sub": "Tertiary Cvg Mem Rel to Sub",
        "Tertiary Cvg Sub Address": "Tertiary Company Address 1",
        "Tertiary Cvg Sub City": "Tertiary Company City",
        "Tertiary Cvg Sub State": "Tertiary Company State",
        "Tertiary CVG Sub ZIP": "Tertiary Company ZIP",
        "Tertiary CVG Group Num": "Tertiary Sub Group",
        "Cvg 3 Auth Num": "Tertiary Sub Auth Num",
        "Tertiary CVG Address 1": "Tertiary Company Address 1",
        "Tertiary CVG City": "Tertiary Company City",
        "Tertiary CVG State": "Tertiary Company State",
        "Tertiary CVG ZIP": "Tertiary Company ZIP",
        
        # Additional fields
        "Phone": "Patient CellPhone",
        "Patient Employer": "Patient Employer",
        "Patient Class": "PatientClass",
        "Billing Status": "Notes",
        "Prim. Member #": "Primary Sub ID",
        "Primary Cvg Payer": "Primary Company Name",
        "Pat Primary CVG Payer ID": "Pat Primary CVG Payer ID",
        "Primary CVG Address 2": "Primary Company Address 1",
        "Primary CVG Phone": "Primary CVG Phone",
        "Primary CVG Sub ID": "Primary Sub ID",
        "Anes Type": "Anesthesia Type",
        "CSN": "Case #",
        "Admit Date": "Admit Date",
        "Discharge Date": "Discharge Date",
        "Encounter Client": "PatientClass",
        "Admission Comments": "Notes",
        "Admit Status": "PatientClass",
        
        # DOB and Gender (will use "if self" logic based on relationship)
        "DOB": "Patient DOB",
        "Sex": "Patient Sex",
        
        # Company Name fields  
        "Primary Plan": "Primary Company Name",
        "Secondary Plan": "Secondary Company Name",
        "Tertiary Plan": "Tertiary Company Name",
        
        # Additional fields that don't exist in original but needed in output
        # These will be handled in processing logic to check if they exist
        # Fields to check: Responsible Provider, MD, CRNA, SRNA, Locum, Resident, 
        # Concurrent Providers, Physical Status, ICD1, ICD2, ICD3, ICD4
    }


def load_mednet_mapping(mapping_file="mednet-maping.csv"):
    """
    Load the mednet code mapping from database first, with CSV fallback.
    Returns a dictionary mapping UNI codes to internal codes.
    """
    try:
        # TRY 1: Load from database (preferred method)
        from db_utils import get_insurance_mappings_dict
        mapping_dict = get_insurance_mappings_dict()
        
        if mapping_dict:
            print(f"✅ Loaded {len(mapping_dict)} mednet code mappings from DATABASE")
            return mapping_dict
        else:
            print("⚠️  No mappings found in database, trying CSV fallback...")
    except Exception as e:
        print(f"⚠️  Could not load from database ({e}), trying CSV fallback...")
    
    # TRY 2: Fallback to CSV file (legacy method)
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        mapping_path = script_dir / mapping_file
        
        print(f"Loading mednet mapping from CSV: {mapping_path}")
        print(f"File exists: {mapping_path.exists()}")
        
        # Read CSV with dtype=str to preserve leading zeros
        mapping_df = pd.read_csv(mapping_path, dtype=str)
        # Ensure both keys and values are strings for consistent lookup and preserve leading zeros
        mapping_dict = {str(k): str(v) for k, v in zip(mapping_df['InputValue'], mapping_df['OutputValue'])}
        print(f"✅ Loaded {len(mapping_dict)} mednet code mappings from CSV")
        return mapping_dict
    except FileNotFoundError:
        print(f"❌ Warning: {mapping_file} not found at {mapping_path}. Mednet codes will not be mapped.")
        return {}
    except Exception as e:
        print(f"❌ Warning: Error loading {mapping_file}: {e}. Mednet codes will not be mapped.")
        import traceback
        traceback.print_exc()
        return {}


def convert_data(input_file, output_file=None):
    """
    Main function to convert data according to Excel macro logic.
    """
    try:
        # Load mednet mapping
        mednet_mapping = load_mednet_mapping()
        
        # Initialize AI client for ICD code reordering
        ai_client = None
        try:
            ai_client = genai.Client(
                api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
            )
            print("AI client initialized for ICD code reordering")
        except Exception as e:
            print(f"Warning: Could not initialize AI client: {e}. ICD codes will not be reordered.")
        
        # Read the file (CSV or Excel) with dtype=str to preserve leading zeros in codes
        df = None
        last_error = None
        
        # Check if file is Excel or CSV
        if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            try:
                print(f"Reading Excel file: {input_file}")
                df = pd.read_excel(input_file, dtype=str)
                print(f"Successfully read Excel file")
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                last_error = e
        else:
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    print(f"Attempting to read CSV file with encoding: {encoding}")
                    df = pd.read_csv(input_file, dtype=str, encoding=encoding)
                    print(f"Successfully read CSV file with encoding: {encoding}")
                    break
                except UnicodeDecodeError as e:
                    print(f"Failed to read with {encoding}: {e}")
                    last_error = e
                    continue
        
        if df is None:
            print(f"Error: Could not read file. Last error: {last_error}")
            return False
        
        if len(df) < 1:
            print("Error: CSV file must have at least 1 row (headers)")
            return False
        
        # Get the hardcoded mapping
        header_mapping = get_header_mapping()
        
        # Create new dataframe with mapped headers
        result_data = []
        
        # PHASE 1: Extract ICD codes for all rows and prepare for parallel AI reordering
        print("\n🔍 Phase 1: Extracting ICD codes from all rows...")
        icd_extraction_data = []  # Store (row_idx, unique_codes, procedure, post_op_diag, post_op_coded)
        
        for row_idx in range(len(df)):
            # Extract ICD codes from both POST-OP DIAGNOSIS columns
            icd_codes = []
            
            # PRIORITY 1: Get POST-OP DIAGNOSIS column codes first
            post_op_diag_col = find_header(df, "POST-OP DIAGNOSIS")
            post_op_diag_value = ''
            codes_from_post_op_diag = []
            if post_op_diag_col:
                post_op_diag_value = df.iloc[row_idx].get(post_op_diag_col, '')
                codes_from_post_op_diag = extract_icd_codes(post_op_diag_value)
                icd_codes.extend(codes_from_post_op_diag)
            
            # PRIORITY 2: Get Post-op Diagnosis - Coded column codes
            post_op_coded_col = find_header(df, "Post-op Diagnosis - Coded")
            post_op_coded_value = ''
            codes_from_post_op_coded = []
            if post_op_coded_col:
                post_op_coded_value = df.iloc[row_idx].get(post_op_coded_col, '')
                codes_from_post_op_coded = extract_icd_codes(post_op_coded_value)
                icd_codes.extend(codes_from_post_op_coded)
            
            # PRIORITY 3: Add any existing ICD1-ICD4 codes last (as fallback)
            codes_from_existing_icd = []
            for i in range(4):
                icd_field = f"ICD{i+1}"
                if icd_field in df.columns:
                    existing_value = df.iloc[row_idx].get(icd_field, '')
                    if not pd.isna(existing_value) and str(existing_value).strip():
                        code_value = str(existing_value).strip()
                        codes_from_existing_icd.append(code_value)
                        icd_codes.append(code_value)
            
            # Remove duplicates while preserving order
            unique_codes = []
            seen = set()
            for code in icd_codes:
                if code and code not in seen:
                    unique_codes.append(code)
                    seen.add(code)
            
            # Get procedure value for AI reordering
            procedure_value = df.iloc[row_idx].get('Procedure', '')
            
            # Store for AI reordering with detailed logging
            icd_extraction_data.append({
                'row_idx': row_idx,
                'unique_codes': unique_codes,
                'procedure': procedure_value,
                'post_op_diag': post_op_diag_value,
                'post_op_coded': post_op_coded_value,
                'codes_from_post_op_diag': codes_from_post_op_diag,
                'codes_from_post_op_coded': codes_from_post_op_coded,
                'codes_from_existing_icd': codes_from_existing_icd
            })
        
        print(f"✓ Extracted ICD codes from {len(icd_extraction_data)} rows")
        
        # PHASE 2: Parallel AI reordering of ICD codes
        icd_reordered_map = {}  # Map row_idx -> reordered_codes
        icd_success_map = {}    # Map row_idx -> success_flag
        icd_prompt_map = {}     # Map row_idx -> final_prompt
        icd_response_map = {}   # Map row_idx -> final_response
        
        if ai_client is not None:
            # Prepare tasks for rows that need reordering (2+ codes)
            reordering_tasks = []
            for data in icd_extraction_data:
                if len(data['unique_codes']) >= 2:
                    reordering_tasks.append((
                        data['row_idx'],
                        data['unique_codes'],
                        data['procedure'],
                        data['post_op_diag'],
                        data['post_op_coded']
                    ))
            
            if reordering_tasks:
                print(f"\n🤖 Phase 2: AI reordering {len(reordering_tasks)} rows with 2+ ICD codes (10 workers)...")
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    # Submit all tasks
                    future_to_row = {executor.submit(process_icd_reordering_task, task): task[0] for task in reordering_tasks}
                    
                    completed = 0
                    total = len(future_to_row)
                    successful_reorders = 0
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_row):
                        completed += 1
                        row_idx = future_to_row[future]
                        
                        try:
                            result_row_idx, reordered_codes, success_flag, final_prompt, final_response = future.result()
                            icd_reordered_map[result_row_idx] = reordered_codes
                            icd_success_map[result_row_idx] = success_flag
                            icd_prompt_map[result_row_idx] = final_prompt
                            icd_response_map[result_row_idx] = final_response
                            
                            if success_flag:
                                successful_reorders += 1
                            
                            # Progress update every 10%
                            if completed % max(1, total // 10) == 0:
                                progress = (completed / total) * 100
                                print(f"    📊 Progress: {completed}/{total} ({progress:.1f}%) - {successful_reorders} successful")
                        except Exception as e:
                            print(f"    ⚠️  Row {row_idx}: Exception during AI reordering: {str(e)}")
                            icd_success_map[row_idx] = False
                            icd_prompt_map[row_idx] = ""
                            icd_response_map[row_idx] = ""
                
                print(f"✓ AI reordering complete for {len(icd_reordered_map)} rows ({successful_reorders} successful)")
            else:
                print("\n⏭️  Phase 2: No rows with 2+ ICD codes, skipping AI reordering")
        else:
            print("\n⏭️  Phase 2: AI client not available, skipping ICD reordering")
        
        # PHASE 2.5: Parallel AI update of ICD codes to latest versions
        icd_updated_map = {}  # Map row_idx -> updated_codes
        icd_update_success_map = {}    # Map row_idx -> success_flag
        icd_update_prompt_map = {}     # Map row_idx -> final_prompt
        icd_update_response_map = {}   # Map row_idx -> final_response
        
        if ai_client is not None:
            # Prepare tasks for all rows with ICD codes (use reordered codes if available, otherwise original)
            update_tasks = []
            for data in icd_extraction_data:
                row_idx = data['row_idx']
                # Use reordered codes if available, otherwise use original extracted codes
                codes_to_update = icd_reordered_map.get(row_idx, data['unique_codes'])
                
                # Only process rows that have at least one ICD code
                if len(codes_to_update) >= 1:
                    update_tasks.append((
                        row_idx,
                        codes_to_update,
                        data['procedure'],
                        data['post_op_diag'],
                        data['post_op_coded']
                    ))
            
            if update_tasks:
                print(f"\n🔄 Phase 2.5: AI updating {len(update_tasks)} rows with ICD codes to latest versions (10 workers)...")
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    # Submit all tasks
                    future_to_row = {executor.submit(process_icd_update_task, task): task[0] for task in update_tasks}
                    
                    completed = 0
                    total = len(future_to_row)
                    successful_updates = 0
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_row):
                        completed += 1
                        row_idx = future_to_row[future]
                        
                        try:
                            result_row_idx, updated_codes, success_flag, final_prompt, final_response = future.result()
                            icd_updated_map[result_row_idx] = updated_codes
                            icd_update_success_map[result_row_idx] = success_flag
                            icd_update_prompt_map[result_row_idx] = final_prompt
                            icd_update_response_map[result_row_idx] = final_response
                            
                            if success_flag:
                                successful_updates += 1
                            
                            # Progress update every 10%
                            if completed % max(1, total // 10) == 0:
                                progress = (completed / total) * 100
                                print(f"    📊 Progress: {completed}/{total} ({progress:.1f}%) - {successful_updates} successful")
                        except Exception as e:
                            print(f"    ⚠️  Row {row_idx}: Exception during AI code update: {str(e)}")
                            icd_update_success_map[row_idx] = False
                            icd_update_prompt_map[row_idx] = ""
                            icd_update_response_map[row_idx] = ""
                
                print(f"✓ AI code update complete for {len(icd_updated_map)} rows ({successful_updates} successful)")
            else:
                print("\n⏭️  Phase 2.5: No rows with ICD codes, skipping AI code update")
        else:
            print("\n⏭️  Phase 2.5: AI client not available, skipping ICD code update")
        
        # PHASE 3: Process each data row with regular conversion + apply updated ICD codes
        print(f"\n📝 Phase 3: Processing {len(df)} rows with field mapping...")
        
        for row_idx in range(len(df)):
            new_row = {}
            
            # Process each column in the original data
            for old_col in df.columns:
                # Clean the column name (remove quotes if present)
                clean_old_col = old_col.strip('"')
                # Find the new header name
                new_header = header_mapping.get(clean_old_col, old_col)  # Use original if no mapping
                
                # Get the value
                value = df.iloc[row_idx][old_col]
                
                # Apply special processing based on column type
                if pd.isna(value):
                    new_row[new_header] = ""
                elif str(value).lower().strip() == "none":
                    # Global "None" value cleaning - leave empty for all columns
                    new_row[new_header] = ""
                elif old_col.lower() == "phone":
                    # Phone splitting logic - split into Home, Work, Cell
                    phone_value = str(value).strip()
                    new_row["Patient HomePhone"] = extract_phone_by_type(phone_value, "home")
                    new_row["Patient WorkPhone"] = extract_phone_by_type(phone_value, "work")
                    new_row["Patient CellPhone"] = extract_phone_by_type(phone_value, "mobile")
                    # Skip regular processing for Phone column to avoid overwriting
                    continue
                elif "patient name" in old_col.lower():
                    # Patient name splitting logic - create separate columns
                    patient_name = str(value)
                    new_row["Patient Last Name"] = split_patient_name(patient_name, "last")
                    new_row["Patient First Name"] = split_patient_name(patient_name, "first")
                    new_row["Patient Middle Name"] = split_patient_name(patient_name, "middle")
                elif old_col.lower() == "dob" or old_col.lower() == "patient dob":
                    # DOB with "if self" logic - check relationship to determine patient vs guarantor DOB
                    new_row[new_header] = process_dob_gender_logic(df, row_idx, old_col, new_header, is_dob=True)
                elif old_col.lower() == "sex" or old_col.lower() == "patient sex" or old_col.lower() == "gender":
                    # Gender with "if self" logic - check relationship to determine patient vs guarantor gender
                    new_row[new_header] = process_dob_gender_logic(df, row_idx, old_col, new_header, is_dob=False)
                elif "nill" in str(value).lower() or "nil" in str(value).lower():
                    # Explicit "Nill" (leave empty)
                    new_row[new_header] = ""
                elif "mednet code" in new_header.lower() or "mednet code" in old_col.lower():
                    # SKIP mednet code mapping for CVG Payer ID fields - just pass through the value as-is
                    # Only Primary/Secondary/Tertiary Plan processing should populate mednet codes
                    if "payer id" in old_col.lower() or "cvg payer" in old_col.lower():
                        # Keep the original value, don't do any mapping
                        new_row[new_header] = value if not pd.isna(value) else ""
                    else:
                        # For other mednet code fields (if any), do the mapping
                        extracted_code = extract_mednet_code(value)
                        if extracted_code and mednet_mapping:
                            # Convert to string for mapping lookup
                            code_str = str(extracted_code)
                            # Try exact match first
                            if code_str in mednet_mapping:
                                mapped_code = mednet_mapping[code_str]
                                # Debug logging for TRAN mappings only
                                if mapped_code == "TRAN":
                                    print(f"[DEBUG Row {row_idx}] GENERAL MEDNET TRAN MAPPING:")
                                    print(f"[DEBUG Row {row_idx}] Column: {old_col}")
                                    print(f"[DEBUG Row {row_idx}] Original value: {value}")
                                    print(f"[DEBUG Row {row_idx}] Extracted code: {code_str}")
                                    print(f"[DEBUG Row {row_idx}] Mapped to: {mapped_code}")
                                new_row[new_header] = mapped_code
                            else:
                                # Try to find best prefix match (4-digit code matching 6-digit codes)
                                matches = [str(k) for k in mednet_mapping.keys() if str(k).startswith(code_str)]
                                if matches:
                                    # Use the first match (most common pattern)
                                    mapped_code = mednet_mapping[matches[0]]
                                    # Debug logging for TRAN mappings only
                                    if mapped_code == "TRAN":
                                        print(f"[DEBUG Row {row_idx}] GENERAL MEDNET TRAN MAPPING (PREFIX):")
                                        print(f"[DEBUG Row {row_idx}] Column: {old_col}")
                                        print(f"[DEBUG Row {row_idx}] Original value: {value}")
                                        print(f"[DEBUG Row {row_idx}] Extracted code: {code_str}")
                                        print(f"[DEBUG Row {row_idx}] Prefix match: {matches[0]}")
                                        print(f"[DEBUG Row {row_idx}] Mapped to: {mapped_code}")
                                    new_row[new_header] = mapped_code
                                else:
                                    # No match found, leave empty
                                    new_row[new_header] = ""
                        else:
                            # No extracted code or no mapping available, leave empty
                            new_row[new_header] = ""
                elif "primary plan" in old_col.lower():
                    # Primary Plan: extract code before dash, map to mednet code, keep text after dash as company name
                    if not pd.isna(value) and value:
                        value_str = str(value).strip()
                        # Find the code part (before the first dash)
                        if '-' in value_str:
                            code_part = value_str.split('-')[0].strip()
                            text_part = value_str.split('-', 1)[1].strip()
                            
                            # Debug logging for TRAN mappings only
                            if mednet_mapping and code_part in mednet_mapping:
                                mapped_value = mednet_mapping[code_part]
                                if mapped_value == "TRAN":
                                    print(f"[DEBUG Row {row_idx}] PRIMARY TRAN MAPPING:")
                                    print(f"[DEBUG Row {row_idx}] Primary Plan value: {value_str}")
                                    print(f"[DEBUG Row {row_idx}] Extracted code: {code_part}")
                                    print(f"[DEBUG Row {row_idx}] Extracted text: {text_part}")
                                    print(f"[DEBUG Row {row_idx}] Mapped to: {mapped_value}")
                            else:
                                # Only show if no mapping found
                                print(f"[DEBUG Row {row_idx}] PRIMARY NO MAPPING:")
                                print(f"[DEBUG Row {row_idx}] Primary Plan value: {value_str}")
                                print(f"[DEBUG Row {row_idx}] Extracted code: {code_part}")
                                print(f"[DEBUG Row {row_idx}] No mapping found for code: {code_part}")
                            
                            # Try to map the code
                            if mednet_mapping and code_part in mednet_mapping:
                                # Found mapping - use mapped code
                                new_row["Primary Mednet Code"] = mednet_mapping[code_part]
                            else:
                                # No mapping found - leave mednet code empty
                                new_row["Primary Mednet Code"] = ""
                            
                            # Keep the text part as company name
                            new_row[new_header] = text_part
                        else:
                            # No dash found, keep original as company name
                            new_row[new_header] = value_str
                            new_row["Primary Mednet Code"] = ""
                    else:
                        # No value
                        new_row[new_header] = ""
                        new_row["Primary Mednet Code"] = ""
                elif "secondary plan" in old_col.lower():
                    # Secondary Plan: extract code before dash, map to mednet code, keep text after dash as company name
                    if not pd.isna(value) and value:
                        value_str = str(value).strip()
                        # Find the code part (before the first dash)
                        if '-' in value_str:
                            code_part = value_str.split('-')[0].strip()
                            text_part = value_str.split('-', 1)[1].strip()
                            
                            # Debug logging for TRAN mappings only
                            if mednet_mapping and code_part in mednet_mapping:
                                mapped_value = mednet_mapping[code_part]
                                if mapped_value == "TRAN":
                                    print(f"[DEBUG Row {row_idx}] SECONDARY TRAN MAPPING:")
                                    print(f"[DEBUG Row {row_idx}] Secondary Plan value: {value_str}")
                                    print(f"[DEBUG Row {row_idx}] Extracted code: {code_part}")
                                    print(f"[DEBUG Row {row_idx}] Extracted text: {text_part}")
                                    print(f"[DEBUG Row {row_idx}] Mapped to: {mapped_value}")
                            else:
                                # Only show if no mapping found
                                print(f"[DEBUG Row {row_idx}] SECONDARY NO MAPPING:")
                                print(f"[DEBUG Row {row_idx}] Secondary Plan value: {value_str}")
                                print(f"[DEBUG Row {row_idx}] Extracted code: {code_part}")
                                print(f"[DEBUG Row {row_idx}] No secondary mapping found for code: {code_part}")
                            
                            # Try to map the code
                            if mednet_mapping and code_part in mednet_mapping:
                                # Found mapping - use mapped code
                                new_row["Secondary Mednet Code"] = mednet_mapping[code_part]
                            else:
                                # No mapping found - leave mednet code empty
                                new_row["Secondary Mednet Code"] = ""
                            
                            # Keep the text part as company name
                            new_row[new_header] = text_part
                        else:
                            # No dash found, keep original as company name
                            new_row[new_header] = value_str
                            new_row["Secondary Mednet Code"] = ""
                    else:
                        # No value
                        new_row[new_header] = ""
                        new_row["Secondary Mednet Code"] = ""
                elif "tertiary plan" in old_col.lower():
                    # Tertiary Plan: extract code before dash, map to mednet code, keep text after dash as company name
                    if not pd.isna(value) and value:
                        value_str = str(value).strip()
                        # Find the code part (before the first dash)
                        if '-' in value_str:
                            code_part = value_str.split('-')[0].strip()
                            text_part = value_str.split('-', 1)[1].strip()
                            
                            # Debug logging for TRAN mappings only
                            if mednet_mapping and code_part in mednet_mapping:
                                mapped_value = mednet_mapping[code_part]
                                if mapped_value == "TRAN":
                                    print(f"[DEBUG Row {row_idx}] TERTIARY TRAN MAPPING:")
                                    print(f"[DEBUG Row {row_idx}] Tertiary Plan value: {value_str}")
                                    print(f"[DEBUG Row {row_idx}] Extracted code: {code_part}")
                                    print(f"[DEBUG Row {row_idx}] Extracted text: {text_part}")
                                    print(f"[DEBUG Row {row_idx}] Mapped to: {mapped_value}")
                            else:
                                # Only show if no mapping found
                                print(f"[DEBUG Row {row_idx}] TERTIARY NO MAPPING:")
                                print(f"[DEBUG Row {row_idx}] Tertiary Plan value: {value_str}")
                                print(f"[DEBUG Row {row_idx}] Extracted code: {code_part}")
                                print(f"[DEBUG Row {row_idx}] No tertiary mapping found for code: {code_part}")
                            
                            # Try to map the code
                            if mednet_mapping and code_part in mednet_mapping:
                                # Found mapping - use mapped code
                                new_row["Tertiary Mednet Code"] = mednet_mapping[code_part]
                            else:
                                # No mapping found - leave mednet code empty
                                new_row["Tertiary Mednet Code"] = ""
                            
                            # Keep the text part as company name
                            new_row[new_header] = text_part
                        else:
                            # No dash found, keep original as company name
                            new_row[new_header] = value_str
                            new_row["Tertiary Mednet Code"] = ""
                    else:
                        # No value
                        new_row[new_header] = ""
                        new_row["Tertiary Mednet Code"] = ""
                elif "company name" in new_header.lower() or "company name" in old_col.lower():
                    # Company Names: strip numeric code, keep only alphabetic part
                    new_row[new_header] = process_company_name(value)
                elif "marital status" in new_header.lower():
                    # Clean marital status - remove brackets and numbers
                    clean_value = str(value).strip()
                    if '[' in clean_value and ']' in clean_value:
                        # Remove everything from '[' to ']' including the brackets
                        clean_value = clean_value.split('[')[0].strip()
                    new_row[new_header] = clean_value
                elif "employer" in new_header.lower() or "employer" in old_col.lower():
                    # Patient Employer: leave empty if "None" (case insensitive)
                    if str(value).lower().strip() == "none":
                        new_row[new_header] = ""
                    else:
                        new_row[new_header] = value
                elif "an start" in new_header.lower() or "an start" in old_col.lower():
                    # Anesthesia Start: format time and combine with date
                    new_row[new_header] = format_anesthesia_time(df, row_idx, value, "start")
                elif "an stop" in new_header.lower() or "an stop" in old_col.lower():
                    # Anesthesia Stop: format time and combine with date
                    new_row[new_header] = format_anesthesia_time(df, row_idx, value, "stop")
                elif "surgeons" in old_col.lower():
                    # Surgeon field: create both Surgeon and Referring fields
                    new_row[new_header] = value  # Original Surgeon field
                    new_row["Referring"] = value  # Duplicate as Referring field
                elif "csn" in old_col.lower():
                    # CSN field: create both Case # and CSN fields
                    new_row[new_header] = value  # Original Case # field (via mapping)
                    new_row["CSN"] = value  # Also populate CSN field
                elif "patient class" in old_col.lower():
                    # Patient Class: take first word, add "hospital", and capitalize
                    if pd.isna(value) or not value:
                        new_row[new_header] = ""
                        new_row["Place of Service"] = ""
                    else:
                        value_str = str(value).strip()
                        # Get first word (split by space or dash)
                        first_word = ""
                        for char in value_str:
                            if char in [' ', '-']:
                                break
                            first_word += char
                        
                        # Add "hospital" and capitalize
                        formatted_value = f"{first_word} hospital".title()
                        new_row[new_header] = value  # Keep original value
                        new_row["Place Of Service"] = formatted_value
                elif "primary cvg mem rel to sub" in old_col.lower():
                    # Primary Cvg Mem Rel to Sub: if "self", auto-populate Primary Sub DOB and Gender
                    new_row[new_header] = value
                    
                    # Check if relationship is "self"
                    if not pd.isna(value) and str(value).lower().strip() == "self":
                        # Get DOB and Sex values from the same row
                        dob_value = df.iloc[row_idx].get('DOB', '')
                        sex_value = df.iloc[row_idx].get('Sex', '')
                        
                        # Auto-populate Primary Sub DOB and Primary Sub Gender
                        new_row["Primary Sub DOB"] = dob_value if not pd.isna(dob_value) else ""
                        new_row["Primary Sub Gender"] = sex_value if not pd.isna(sex_value) else ""
                    else:
                        # If not "self", leave these fields empty
                        new_row["Primary Sub DOB"] = ""
                        new_row["Primary Sub Gender"] = ""
                elif "secondary cvg mem rel to sub" in old_col.lower():
                    # Secondary Cvg Mem Rel to Sub: if "self", auto-populate Secondary Sub DOB and Gender
                    new_row[new_header] = value
                    
                    # Check if relationship is "self"
                    if not pd.isna(value) and str(value).lower().strip() == "self":
                        # Get DOB and Sex values from the same row
                        dob_value = df.iloc[row_idx].get('DOB', '')
                        sex_value = df.iloc[row_idx].get('Sex', '')
                        
                        # Auto-populate Secondary Sub DOB and Secondary Sub Gender
                        new_row["Secondary Sub DOB"] = dob_value if not pd.isna(dob_value) else ""
                        new_row["Secondary Sub Gender"] = sex_value if not pd.isna(sex_value) else ""
                    else:
                        # If not "self", leave these fields empty
                        new_row["Secondary Sub DOB"] = ""
                        new_row["Secondary Sub Gender"] = ""
                elif "tertiary cvg mem rel to sub" in old_col.lower():
                    # Tertiary Cvg Mem Rel to Sub: if "self", auto-populate Tertiary Sub DOB and Gender
                    new_row[new_header] = value
                    
                    # Check if relationship is "self"
                    if not pd.isna(value) and str(value).lower().strip() == "self":
                        # Get DOB and Sex values from the same row
                        dob_value = df.iloc[row_idx].get('DOB', '')
                        sex_value = df.iloc[row_idx].get('Sex', '')
                        
                        # Auto-populate Tertiary Sub DOB and Tertiary Sub Gender
                        new_row["Tertiary Sub DOB"] = dob_value if not pd.isna(dob_value) else ""
                        new_row["Tertiary Sub Gender"] = sex_value if not pd.isna(sex_value) else ""
                    else:
                        # If not "self", leave these fields empty
                        new_row["Tertiary Sub DOB"] = ""
                        new_row["Tertiary Sub Gender"] = ""
                    
                    # Check if Tertiary Sub Name exists in original data, if not create as empty
                    tertiary_sub_name_value = df.iloc[row_idx].get('Tertiary Sub Name', '')
                    if pd.isna(tertiary_sub_name_value):
                        new_row["Tertiary Sub Name"] = ""  # Create empty if not found
                    else:
                        new_row["Tertiary Sub Name"] = tertiary_sub_name_value  # Use original value if found
                elif "guarantor relation" in old_col.lower():
                    # Guarantor Relation: if "self", auto-populate Guarantor DOB and Gender
                    new_row[new_header] = value
                    
                    # Check if relationship is "self"
                    if not pd.isna(value) and str(value).lower().strip() == "self":
                        # Get DOB and Sex values from the same row
                        dob_value = df.iloc[row_idx].get('DOB', '')
                        sex_value = df.iloc[row_idx].get('Sex', '')
                        
                        # Auto-populate Guarantor DOB and Guarantor Gender
                        new_row["Guarantor DOB"] = dob_value if not pd.isna(dob_value) else ""
                        new_row["Guarantor Gender"] = sex_value if not pd.isna(sex_value) else ""
                    else:
                        # If not "self", leave these fields empty
                        new_row["Guarantor DOB"] = ""
                        new_row["Guarantor Gender"] = ""
                elif "anesthesia staff" in old_col.lower():
                    # Skip Anesthesia Staff field - don't include in output
                    pass
                else:
                    # Direct mapping
                    new_row[new_header] = value
            
            # Check for additional fields that may not exist in original data
            additional_fields = [
                "Responsible Provider", "MD", "CRNA", "SRNA", "Locum", "Resident",
                "Physical Status", "ICD1", "ICD2", "ICD3", "ICD4"
            ]
            
            for field in additional_fields:
                if field not in new_row:  # Only add if not already processed
                    field_value = df.iloc[row_idx].get(field, '')
                    if pd.isna(field_value):
                        new_row[field] = ""  # Create empty if not found
                    else:
                        new_row[field] = field_value  # Use original value if found
            
            # Special handling for Concurrent Providers - fix date format if present
            if "Concurrent Providers" in df.columns:
                concurrent_value = df.iloc[row_idx].get("Concurrent Providers", '')
                if not pd.isna(concurrent_value) and concurrent_value:
                    # Get the date from the INPUT CSV "Date" column (which gets renamed to "Charge Date" in output)
                    date_value = df.iloc[row_idx].get("Date", '')
                    new_row["Concurrent Providers"] = fix_concurrent_providers_dates(concurrent_value, date_value)
                elif "Concurrent Providers" not in new_row:
                    new_row["Concurrent Providers"] = ""
            elif "Concurrent Providers" not in new_row:
                new_row["Concurrent Providers"] = ""
            
            # Get ICD codes: Priority 1 = Updated codes (Phase 2.5), Priority 2 = Reordered codes (Phase 2), Priority 3 = Original extracted codes
            # Also track codes before update for logging
            codes_before_update = None
            if row_idx in icd_updated_map:
                # Use updated codes from Phase 2.5
                unique_codes = icd_updated_map[row_idx]
                # Codes before update: use reordered codes if available, otherwise original extracted codes
                if row_idx in icd_reordered_map:
                    codes_before_update = icd_reordered_map[row_idx]
                else:
                    codes_before_update = icd_extraction_data[row_idx]['unique_codes']
                # Check if AI update was successful for this row
                ai_update_success = icd_update_success_map.get(row_idx, False)
                update_prompt = icd_update_prompt_map.get(row_idx, "")
                update_response = icd_update_response_map.get(row_idx, "")
                # Also check reordering status if it was done
                ai_reorder_success = icd_success_map.get(row_idx, False)
                reorder_prompt = icd_prompt_map.get(row_idx, "")
                reorder_response = icd_response_map.get(row_idx, "")
            elif row_idx in icd_reordered_map:
                # Use reordered codes from Phase 2
                unique_codes = icd_reordered_map[row_idx]
                codes_before_update = unique_codes  # No update happened, so before = after
                # Check if AI reordering was successful for this row
                ai_reorder_success = icd_success_map.get(row_idx, False)
                reorder_prompt = icd_prompt_map.get(row_idx, "")
                reorder_response = icd_response_map.get(row_idx, "")
                ai_update_success = False
                update_prompt = ""
                update_response = ""
            else:
                # Use original extracted codes
                unique_codes = icd_extraction_data[row_idx]['unique_codes']
                codes_before_update = unique_codes  # No update happened, so before = after
                # If no AI processing attempted (less than 2 codes), mark as successful
                ai_reorder_success = True if len(unique_codes) <= 1 else False
                reorder_prompt = ""
                reorder_response = ""
                ai_update_success = False
                update_prompt = ""
                update_response = ""
            
            # Add ICD AI Reordering Success column
            new_row["ICD AI Reordering Success"] = "Yes" if ai_reorder_success else "No"
            
            # Add ICD AI Update Success column
            new_row["ICD AI Update Success"] = "Yes" if ai_update_success else "No"
            
            # Add logging columns for AI reordering interaction
            new_row["ICD AI Reordering Prompt"] = reorder_prompt
            new_row["ICD AI Reordering Response"] = reorder_response
            
            # Add logging columns for AI update interaction
            new_row["ICD AI Update Prompt"] = update_prompt
            new_row["ICD AI Update Response"] = update_response
            
            # Add logging columns for codes before and after update
            new_row["ICD Codes Before Update"] = "; ".join(codes_before_update) if codes_before_update else ""
            new_row["ICD Codes After Update"] = "; ".join(unique_codes) if unique_codes else ""
            
            # Add detailed ICD extraction logging columns
            extraction_data = icd_extraction_data[row_idx]
            new_row["ICD Codes from POST-OP DIAGNOSIS"] = "; ".join(extraction_data['codes_from_post_op_diag']) if extraction_data['codes_from_post_op_diag'] else ""
            new_row["ICD Codes from Post-op Diagnosis - Coded"] = "; ".join(extraction_data['codes_from_post_op_coded']) if extraction_data['codes_from_post_op_coded'] else ""
            new_row["ICD Codes from Existing ICD1-ICD4"] = "; ".join(extraction_data['codes_from_existing_icd']) if extraction_data['codes_from_existing_icd'] else ""
            
            # Fill ICD1-ICD4 slots sequentially (max 4 codes)
            for i in range(4):
                icd_field = f"ICD{i+1}"
                if i < len(unique_codes):
                    new_row[icd_field] = unique_codes[i]
                else:
                    # If no code available, set to empty
                    new_row[icd_field] = ""
            
            result_data.append(new_row)
        
        # Create result dataframe
        result_df = pd.DataFrame(result_data)
        
        # Replace any NaN values with empty strings before converting to string
        result_df = result_df.fillna('')
        # Ensure all data is treated as strings to preserve leading zeros
        result_df = result_df.astype(str)
        # Replace any remaining 'nan' strings with empty strings
        result_df = result_df.replace('nan', '')
        
        # Save to output file(s)
        if output_file is None:
            output_file = input_file.replace('.csv', '_converted.csv')
        
        # Save in both CSV and XLSX formats
        try:
            # Remove extension to use base path
            base_path = Path(output_file).with_suffix('')
            csv_path, xlsx_path = save_dataframe_dual_format(result_df, base_path)
            print(f"Conversion complete.")
            print(f"CSV output saved to: {csv_path}")
            if xlsx_path:
                print(f"XLSX output saved to: {xlsx_path}")
        except Exception as e:
            # Fallback to CSV only if dual format fails
            print(f"Warning: Could not save XLSX format ({e}), saving CSV only")
            result_df.to_csv(output_file, index=False)
            print(f"Conversion complete. Output saved to: {output_file}")
        
        print(f"Processed {len(result_data)} rows of data.")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False


def main():
    """
    Main entry point for the script.
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_data.py <input_csv_file> [output_csv_file]")
        print("Example: python convert_data.py uni.csv")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    print(f"Processing file: {input_file}")
    success = convert_data(input_file, output_file)
    
    if success:
        print("Data conversion completed successfully!")
    else:
        print("Data conversion failed!")


if __name__ == "__main__":
    main()