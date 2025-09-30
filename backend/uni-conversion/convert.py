#!/usr/bin/env python3
"""
Python script to replicate Excel macro functionality for data conversion.
This script processes CSV data according to hardcoded mapping rules.
"""

import pandas as pd
import re
import sys
from pathlib import Path


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
        if phone_type == "home" and ("Home Phone" in entry or "Home" in entry):
            # Extract only numeric digits
            phone = ''.join(filter(str.isdigit, entry))
            return phone
        elif phone_type == "work" and ("Work Phone" in entry or "Work" in entry):
            # Extract only numeric digits
            phone = ''.join(filter(str.isdigit, entry))
            return phone
        elif phone_type == "mobile" and ("Mobile" in entry or "Cell" in entry):
            # Extract only numeric digits
            phone = ''.join(filter(str.isdigit, entry))
            return phone
    
    # If no specific type found, return empty
    return ""


def format_anesthesia_time(df, row_idx, time_value, time_type):
    """
    Format anesthesia time by converting 2, 3, or 4-digit time to HH:MM format
    and combine with the date from the same row.
    time_type: "start" or "stop"
    """
    if pd.isna(time_value) or not time_value:
        return ""
    
    time_str = str(time_value).strip()
    
    # Handle different digit lengths
    if time_str.isdigit():
        if len(time_str) == 2:
            # Convert 31 to 00:31
            formatted_time = f"00:{time_str}"
        elif len(time_str) == 3:
            # Convert 331 to 03:31
            formatted_time = f"0{time_str[0]}:{time_str[1:]}"
        elif len(time_str) == 4:
            # Convert 1331 to 13:31
            formatted_time = f"{time_str[:2]}:{time_str[2:]}"
        else:
            # If not 2, 3, or 4 digits, return as-is
            formatted_time = time_str
    else:
        # If not all digits, return as-is
        formatted_time = time_str
    
    # Get the date from the same row
    date_value = df.iloc[row_idx].get('Date', '')
    if pd.isna(date_value) or not date_value:
        return formatted_time
    
    # Combine date and time with one space
    date_str = str(date_value).strip()
    return f"{date_str} {formatted_time}"


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
    and fixing malformed time formats.
    
    Args:
        concurrent_providers_value: The Concurrent Providers string
        charge_date: The Charge Date value to use for missing dates
    
    Returns:
        Fixed Concurrent Providers string with dates added where needed and times formatted correctly
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
            "Location": "Room",
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
        "Primary CVG Phone": "Primary Sub Auth Num",
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
    Load the mednet code mapping from CSV file.
    Returns a dictionary mapping UNI codes to internal codes.
    """
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        mapping_path = script_dir / mapping_file
        
        print(f"Loading mednet mapping from: {mapping_path}")
        print(f"File exists: {mapping_path.exists()}")
        
        # Read CSV with dtype=str to preserve leading zeros
        mapping_df = pd.read_csv(mapping_path, dtype=str)
        # Ensure both keys and values are strings for consistent lookup and preserve leading zeros
        mapping_dict = {str(k): str(v) for k, v in zip(mapping_df['InputValue'], mapping_df['OutputValue'])}
        print(f"Loaded {len(mapping_dict)} mednet code mappings")
        return mapping_dict
    except FileNotFoundError:
        print(f"Warning: {mapping_file} not found at {mapping_path}. Mednet codes will not be mapped.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading {mapping_file}: {e}. Mednet codes will not be mapped.")
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
        
        # Read the CSV file with dtype=str to preserve leading zeros in codes
        df = pd.read_csv(input_file, dtype=str)
        
        if len(df) < 1:
            print("Error: CSV file must have at least 1 row (headers)")
            return False
        
        # Get the hardcoded mapping
        header_mapping = get_header_mapping()
        
        # Create new dataframe with mapped headers
        result_data = []
        
        # Process each data row
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
                elif "patient name" in old_col.lower():
                    # Patient name splitting logic - create separate columns
                    patient_name = str(value)
                    new_row["Patient Last Name"] = split_patient_name(patient_name, "last")
                    new_row["Patient First Name"] = split_patient_name(patient_name, "first")
                    new_row["Patient Middle Name"] = split_patient_name(patient_name, "middle")
                elif "dob" in old_col.lower():
                    # DOB with "if self" logic - check relationship to determine patient vs guarantor DOB
                    new_row[new_header] = process_dob_gender_logic(df, row_idx, old_col, new_header, is_dob=True)
                elif "sex" in old_col.lower():
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
                elif "phone" in old_col.lower():
                    # Phone splitting logic - split into Home, Work, Cell
                    phone_value = str(value).strip()
                    new_row["Patient HomePhone"] = extract_phone_by_type(phone_value, "home")
                    new_row["Patient WorkPhone"] = extract_phone_by_type(phone_value, "work")
                    new_row["Patient CellPhone"] = extract_phone_by_type(phone_value, "mobile")
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
            
            result_data.append(new_row)
        
        # Create result dataframe
        result_df = pd.DataFrame(result_data)
        
        # Save to output file
        if output_file is None:
            output_file = input_file.replace('.csv', '_converted.csv')
        
        # Replace any NaN values with empty strings before converting to string
        result_df = result_df.fillna('')
        # Ensure all data is treated as strings to preserve leading zeros
        result_df = result_df.astype(str)
        # Replace any remaining 'nan' strings with empty strings
        result_df = result_df.replace('nan', '')
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