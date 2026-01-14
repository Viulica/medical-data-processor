#!/usr/bin/env python3
"""
Script to generate medical modifiers based on mednet codes and provider information.
This script processes CSV data and generates modifiers according to specific business rules.

Modifier Hierarchy:
1. M1 modifiers (AA/QK/QZ/QX) - Provider type modifiers
2. GC modifier - Teaching physician (when Resident is present and Medicare Modifiers = YES)
   NOTE: GC is NOT added when QK is present (to prevent QK + GC combination)
3. QS modifier - Monitored Anesthesia Care (when Anesthesia Type = MAC and Medicare Modifiers = YES)
4. P modifiers (P1-P6) - Physical status modifiers
5. PT modifier - Added to LAST available position (no gaps) when "Polyps found" = "FOUND" AND "colonoscopy_is_screening" = "TRUE"
   - PT modifier REQUIRES Medicare Modifiers = YES for the insurance
   - By default, PT modifier is only added for Medicare insurances (with Medicare Modifiers = YES)
   - If add_pt_for_non_medicare=True, PT modifier can also be added for non-Medicare insurances (but still requires Medicare Modifiers = YES)
   - PT modifier is placed in the last sequential position (M1, M2, M3, or M4) without gaps

Medical Direction Override:
- When turn_off_medical_direction=True, all medical direction YES values from the database
  are automatically overridden to NO during processing. This affects modifier generation
  by treating all cases as if medical direction is disabled.

Special Cases:
- Mednet Code 003 (Blue Cross): 
  * If turn_off_bcbs_medicare_modifiers is True: Only generate P modifiers (set Medicare Modifiers = NO and Medical Direction = NO)
  * If turn_off_bcbs_medicare_modifiers is False:
    - If BOTH MD and CRNA are present: Artificially set Medicare Modifiers = YES and Medical Direction = YES
      (unless turn_off_medical_direction is True, then Medical Direction stays NO)
    - If NOT both MD and CRNA: Artificially set Medicare Modifiers = NO and Medical Direction = NO
  This allows normal modifier generation (including GC) when both providers are present and BCBS modifiers are enabled,
  but limits to only P modifier when BCBS modifiers are turned off or when providers are not both present.

Peripheral Blocks Row Generation:
- Mode selection via peripheral_blocks_mode parameter:
  * "UNI": Generate blocks ONLY when "Anesthesia Type" = "General" (case insensitive)
  * "other": Generate blocks when "Anesthesia Type" is NOT "MAC" (case insensitive)
- If peripheral_blocks field contains "NONE DONE", NO blocks will be generated (regardless of mode)
- When conditions are met and "peripheral_blocks" field is non-empty, creates duplicate rows for each block
- Format: |cpt_code;MD;Resident;CRNA;M1;M2|cpt_code;MD;Resident;CRNA;M1;M2|...
- Each duplicate row:
  * ASA Code and Procedure Code = CPT code from block
  * MD/Resident/CRNA = values from block
  * Responsible Provider = MD value from block
  * Modifiers M1-M4 cleared, then M1 and M2 set from block
  * Concurrent Providers cleared
  * An Start and An Stop cleared
  * SRNA cleared
  * ICD codes:
    - ASA Code 01967 or 1967: Clear all, set ICD1 = "080"
    - Peripheral nerve blocks (644XX, 64488): Clear all, set ICD1 = "G89.18"
    - Arterial line (36620): Copy ICD1-ICD4 from original input row
    - Ultrasound guidance (76937): Keep ICD1-ICD4 ONLY if it comes after 36620, otherwise clear all
    - CVP (36556, 93503): Clear all ICD1-ICD4
    - Other codes: Clear all ICD1-ICD4

TEE (Transesophageal Echocardiogram) Row Generation:
- When "tee_was_done" = TRUE AND "tee_diagnosis" contains diagnosis codes (not empty)
- Creates a single duplicate row for the TEE procedure
- Diagnosis codes format: (I35.1, I34.0, I36.1, I37) or empty ()
- TEE row is created ONLY if diagnosis codes are present (not empty brackets)
- Each TEE row:
  * ASA Code and Procedure Code = 93312 (hardcoded)
  * M1 = "26" (hardcoded)
  * M2 = "59" (hardcoded)
  * M3 and M4 = cleared
  * ICD1-ICD4 = populated from tee_diagnosis codes in order
  * Concurrent Providers cleared
  * An Start and An Stop cleared
  * SRNA cleared
  * Anesthesia Type cleared
  * Other fields copied from original row

Emergent Case Row Generation:
- When "emergent_case" = TRUE
- Creates a single duplicate row for the emergent case procedure
- IMPORTANT: Code 99140 is ONLY added for commercial insurance (NOT Medicare, Medicare replacement, or Medicaid)
- Insurance check: If Insurance Name OR Insurance Plan contains "medicare" or "medicaid" (case-insensitive), 99140 is NOT added
- Each emergent_case row:
  * ASA Code and Procedure Code = 99140 (hardcoded, commercial insurance only)
  * M1, M2, M3, M4 = cleared
  * ICD1-ICD4 = copied from original row (not changed)
  * Concurrent Providers cleared
  * An Start and An Stop cleared
  * SRNA preserved (not cleared)
  * Anesthesia Type cleared
  * Other fields copied from original row
"""

import pandas as pd
import sys
from pathlib import Path
import os

# Add parent directory to path to import export_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from export_utils import save_dataframe_dual_format


def load_modifiers_definition(definition_file="modifiers_definition.csv"):
    """
    Load the modifiers definition from PostgreSQL database.
    Falls back to CSV file if database is not available.
    Returns a dictionary mapping mednet codes to (medicare_modifiers, medical_direction, enable_qs) tuples.
    """
    # Try to load from database first
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from db_utils import get_modifiers_dict
        
        modifiers_dict = get_modifiers_dict()
        if modifiers_dict:
            print(f"✅ Loaded {len(modifiers_dict)} modifiers definitions from database")
            return modifiers_dict
        else:
            print("⚠️  Database returned empty results, falling back to CSV...")
    except ImportError:
        print("⚠️  db_utils not available, falling back to CSV...")
    except Exception as e:
        print(f"⚠️  Database error ({e}), falling back to CSV...")
    
    # Fallback to CSV file
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        definition_path = script_dir / definition_file
        
        df = pd.read_csv(definition_path)
        
        # Create dictionary mapping MedNet Code to (Medicare Modifiers, Bill Medical Direction, Enable QS)
        modifiers_dict = {}
        for _, row in df.iterrows():
            mednet_code = str(row['MedNet Code']).strip()
            medicare_modifiers = str(row['Medicare Modifiers']).strip().upper() == 'YES'
            medical_direction = str(row['Bill Medical Direction']).strip().upper() == 'YES'
            # CSV fallback: default enable_qs to True for backward compatibility
            enable_qs = True
            modifiers_dict[mednet_code] = (medicare_modifiers, medical_direction, enable_qs)
        
        print(f"📄 Loaded {len(modifiers_dict)} modifiers definitions from CSV (fallback)")
        return modifiers_dict
    
    except FileNotFoundError:
        print(f"❌ Warning: {definition_file} not found. No modifiers will be generated.")
        return {}
    except Exception as e:
        print(f"❌ Warning: Error loading {definition_file}: {e}. No modifiers will be generated.")
        return {}


def parse_peripheral_blocks(peripheral_blocks_str):
    """
    Parse the peripheral_blocks field string into a list of block dictionaries.
    
    Format: |cpt_code;MD;Resident;CRNA;M1;M2|cpt_code;MD;Resident;CRNA;M1;M2|...
    
    Returns a list of dictionaries with keys: cpt_code, md, resident, crna, m1, m2
    Empty fields in the input are represented as empty strings in the output.
    """
    blocks = []
    
    if not peripheral_blocks_str or pd.isna(peripheral_blocks_str):
        return blocks
    
    # Split by | and filter out empty strings
    block_strings = [b.strip() for b in str(peripheral_blocks_str).split('|') if b.strip()]
    
    for block_str in block_strings:
        # Split by ; to get the 6 fields
        parts = block_str.split(';')
        
        # Ensure we have exactly 6 parts (pad with empty strings if needed)
        while len(parts) < 6:
            parts.append('')
        
        block = {
            'cpt_code': parts[0].strip(),
            'md': parts[1].strip(),
            'resident': parts[2].strip(),
            'crna': parts[3].strip(),
            'm1': parts[4].strip(),
            'm2': parts[5].strip()
        }
        
        # Only add block if it has a CPT code
        if block['cpt_code']:
            blocks.append(block)
    
    return blocks


def parse_tee_diagnosis(tee_diagnosis_str):
    """
    Parse the tee_diagnosis field string into a list of diagnosis codes.
    
    Format: (I35.1, I34.0, I36.1, I37) or empty ()
    
    Returns a list of diagnosis code strings.
    Empty brackets () return an empty list.
    """
    diagnosis_codes = []
    
    if not tee_diagnosis_str or pd.isna(tee_diagnosis_str):
        return diagnosis_codes
    
    # Convert to string and strip whitespace
    tee_diagnosis_str = str(tee_diagnosis_str).strip()
    
    # Remove parentheses
    tee_diagnosis_str = tee_diagnosis_str.strip('()')
    
    # If empty after removing parentheses, return empty list
    if not tee_diagnosis_str or tee_diagnosis_str.strip() == '':
        return diagnosis_codes
    
    # Split by comma and strip whitespace from each code
    codes = [code.strip() for code in tee_diagnosis_str.split(',') if code.strip()]
    
    return codes


def calculate_and_limit_anesthesia_time(an_start_str, an_stop_str, max_minutes=480):
    """
    Calculate anesthesia time from An Start and An Stop columns and limit to max_minutes.
    
    Supports two formats:
    1. HHMM format (4 digits), e.g., "1415" = 14:15 (2:15 PM)
    2. Full datetime string, e.g., "12/05/2025 07:14:00 AM" or "12/05/2025 10:44:00 PM"
    
    Args:
        an_start_str: Start time as string (e.g., "1415" or "12/05/2025 07:14:00 AM")
        an_stop_str: Stop time as string (e.g., "1813" or "12/05/2025 10:44:00 PM")
        max_minutes: Maximum allowed minutes (default 480)
    
    Returns:
        Tuple of (adjusted_an_stop_str, was_limited)
        - adjusted_an_stop_str: The adjusted stop time in original format, or original if no adjustment needed
        - was_limited: Boolean indicating if the time was limited
    """
    # Check if both values exist and are valid
    if not an_start_str or pd.isna(an_start_str) or str(an_start_str).strip() == '':
        return an_stop_str, False
    if not an_stop_str or pd.isna(an_stop_str) or str(an_stop_str).strip() == '':
        return an_stop_str, False
    
    try:
        # Convert to string and strip whitespace
        an_start_str = str(an_start_str).strip()
        an_stop_str = str(an_stop_str).strip()
        
        # Detect format: if it contains "/" or ":" with spaces, it's likely a datetime string
        is_datetime_format = '/' in an_start_str or (':' in an_start_str and ' ' in an_start_str)
        
        start_hour = None
        start_minute = None
        stop_hour = None
        stop_minute = None
        original_format_start = an_start_str
        original_format_stop = an_stop_str
        
        if is_datetime_format:
            # Parse as datetime string
            try:
                # Try to parse with pandas
                start_dt = pd.to_datetime(an_start_str, errors='coerce')
                stop_dt = pd.to_datetime(an_stop_str, errors='coerce')
                
                if pd.isna(start_dt) or pd.isna(stop_dt):
                    # If pandas parsing fails, try manual parsing
                    raise ValueError("Pandas parsing failed")
                
                start_hour = start_dt.hour
                start_minute = start_dt.minute
                stop_hour = stop_dt.hour
                stop_minute = stop_dt.minute
                
            except Exception:
                # Manual parsing for common formats like "12/05/2025 07:14:00 AM"
                # Extract time portion (after the space)
                start_parts = an_start_str.split()
                stop_parts = an_stop_str.split()
                
                if len(start_parts) >= 2 and len(stop_parts) >= 2:
                    start_time_str = start_parts[1]  # e.g., "07:14:00"
                    stop_time_str = stop_parts[1]     # e.g., "10:44:00"
                    start_ampm = start_parts[2] if len(start_parts) > 2 else ''
                    stop_ampm = stop_parts[2] if len(stop_parts) > 2 else ''
                    
                    # Parse time (HH:MM:SS or HH:MM)
                    start_time_parts = start_time_str.split(':')
                    stop_time_parts = stop_time_str.split(':')
                    
                    if len(start_time_parts) >= 2 and len(stop_time_parts) >= 2:
                        start_hour = int(start_time_parts[0])
                        start_minute = int(start_time_parts[1])
                        stop_hour = int(stop_time_parts[0])
                        stop_minute = int(stop_time_parts[1])
                        
                        # Handle AM/PM
                        if start_ampm.upper() == 'PM' and start_hour != 12:
                            start_hour += 12
                        elif start_ampm.upper() == 'AM' and start_hour == 12:
                            start_hour = 0
                        
                        if stop_ampm.upper() == 'PM' and stop_hour != 12:
                            stop_hour += 12
                        elif stop_ampm.upper() == 'AM' and stop_hour == 12:
                            stop_hour = 0
                else:
                    raise ValueError("Could not parse datetime format")
        else:
            # Parse as HHMM format
            if len(an_start_str) < 3 or len(an_stop_str) < 3:
                return an_stop_str, False
            
            # Handle 3-digit times (e.g., "013" = 00:13) or 4-digit times (e.g., "1415" = 14:15)
            if len(an_start_str) == 3:
                an_start_str = '0' + an_start_str
            if len(an_stop_str) == 3:
                an_stop_str = '0' + an_stop_str
            
            # Extract hours and minutes
            start_hour = int(an_start_str[:2])
            start_minute = int(an_start_str[2:4])
            stop_hour = int(an_stop_str[:2])
            stop_minute = int(an_stop_str[2:4])
        
        if start_hour is None or start_minute is None or stop_hour is None or stop_minute is None:
            return an_stop_str, False
        
        # Convert to total minutes from midnight
        start_total_minutes = start_hour * 60 + start_minute
        stop_total_minutes = stop_hour * 60 + stop_minute
        
        # Handle day rollover (if stop time is earlier than start time, assume next day)
        if stop_total_minutes < start_total_minutes:
            # Next day: add 24 hours (1440 minutes)
            stop_total_minutes += 1440
        
        # Calculate duration in minutes
        duration_minutes = stop_total_minutes - start_total_minutes
        
        # If duration exceeds max_minutes, adjust stop time
        if duration_minutes > max_minutes:
            # Calculate new stop time (start + max_minutes)
            new_stop_total_minutes = start_total_minutes + max_minutes
            
            # Handle day rollover for new stop time
            if new_stop_total_minutes >= 1440:
                new_stop_total_minutes = new_stop_total_minutes % 1440
            
            # Convert back to original format
            new_stop_hour = new_stop_total_minutes // 60
            new_stop_minute = new_stop_total_minutes % 60
            
            if is_datetime_format:
                # Reconstruct datetime string with adjusted time
                # Try to preserve the original format
                try:
                    # Parse original stop datetime to get date part
                    stop_dt = pd.to_datetime(original_format_stop, errors='coerce')
                    if not pd.isna(stop_dt):
                        # Create new datetime with adjusted time
                        new_stop_dt = stop_dt.replace(hour=new_stop_hour, minute=new_stop_minute, second=0)
                        # Format back to string (try to match original format)
                        adjusted_an_stop = new_stop_dt.strftime('%m/%d/%Y %I:%M:%S %p')
                    else:
                        # Fallback: reconstruct from original string
                        stop_parts = original_format_stop.split()
                        if len(stop_parts) >= 1:
                            date_part = stop_parts[0]  # e.g., "12/05/2025"
                            ampm = 'AM' if new_stop_hour < 12 else 'PM'
                            display_hour = new_stop_hour if new_stop_hour <= 12 else new_stop_hour - 12
                            if display_hour == 0:
                                display_hour = 12
                            adjusted_an_stop = f"{date_part} {display_hour:02d}:{new_stop_minute:02d}:00 {ampm}"
                        else:
                            adjusted_an_stop = original_format_stop
                except Exception:
                    # Fallback: reconstruct from original string
                    stop_parts = original_format_stop.split()
                    if len(stop_parts) >= 1:
                        date_part = stop_parts[0]  # e.g., "12/05/2025"
                        ampm = 'AM' if new_stop_hour < 12 else 'PM'
                        display_hour = new_stop_hour if new_stop_hour <= 12 else new_stop_hour - 12
                        if display_hour == 0:
                            display_hour = 12
                        adjusted_an_stop = f"{date_part} {display_hour:02d}:{new_stop_minute:02d}:00 {ampm}"
                    else:
                        adjusted_an_stop = original_format_stop
            else:
                # Format as 4-digit string (e.g., "1813")
                adjusted_an_stop = f"{new_stop_hour:02d}{new_stop_minute:02d}"
            
            return adjusted_an_stop, True
        else:
            return an_stop_str, False
            
    except (ValueError, IndexError) as e:
        # If parsing fails, return original stop time
        print(f"Warning: Could not parse anesthesia times (Start: {an_start_str}, Stop: {an_stop_str}): {e}")
        return an_stop_str, False


def apply_colonoscopy_correction(row, asa_code, insurances_df):
    """
    Apply colonoscopy-specific correction rules based on procedure type,
    insurance plan, and polyp findings.
    
    This changes 00812 to 00811 when appropriate based on Medicare rules.
    
    Args:
        row: DataFrame row containing procedure information
        asa_code: The current ASA Code (e.g., "00812")
        insurances_df: DataFrame containing insurance information
    
    Returns:
        Corrected ASA code or original code if no correction applies
    """
    try:
        # Only process if code is 00812
        if str(asa_code).strip() != '00812':
            return asa_code
        
        # Get the required fields from the row
        is_colonoscopy = str(row.get('is_colonoscopy', '')).strip().upper() == 'TRUE'
        colonoscopy_is_screening = str(row.get('colonoscopy_is_screening', '')).strip().upper() == 'TRUE'
        is_upper_endonoscopy = str(row.get('is_upper_endonoscopy', '')).strip().upper() == 'TRUE'
        polyps_found = str(row.get('Polyps found', '')).strip().upper() == 'FOUND'
        primary_mednet_code = str(row.get('Primary Mednet Code', '')).strip()
        
        # Priority rule: if both upper endoscopy and colonoscopy, always return 00813
        if is_upper_endonoscopy and is_colonoscopy:
            print(f"   Colonoscopy correction: Both upper endoscopy and colonoscopy detected -> 00813")
            return "00813"
        
        # Only proceed if it's a colonoscopy
        if not is_colonoscopy:
            return asa_code
        
        # Check if insurance is Medicare
        is_medicare = False
        if primary_mednet_code and not insurances_df.empty:
            # Find the insurance plan by MedNet Code
            insurance_match = insurances_df[insurances_df['MedNet Code'].astype(str).str.strip() == primary_mednet_code]
            if not insurance_match.empty:
                insurance_plan = str(insurance_match.iloc[0].get('Insurance Plan', '')).strip()
                if 'Medicare' in insurance_plan or 'MEDICARE' in insurance_plan or 'medicare' in insurance_plan:
                    is_medicare = True
        
        # Apply correction rules
        if is_medicare:
            # MEDICARE rules
            if colonoscopy_is_screening and not polyps_found:
                print(f"   Colonoscopy correction: Medicare + screening + no polyps -> 00812 (no change)")
                return "00812"
            elif colonoscopy_is_screening and polyps_found:
                print(f"   Colonoscopy correction: Medicare + screening + polyps found -> 00811 (CHANGED)")
                return "00811"
            else:  # not screening (polyps don't matter)
                print(f"   Colonoscopy correction: Medicare + not screening -> 00811 (CHANGED)")
                return "00811"
        else:
            # NOT MEDICARE rules - 00812 stays 00812 for screening with polyps
            if colonoscopy_is_screening and not polyps_found:
                print(f"   Colonoscopy correction: Non-Medicare + screening + no polyps -> 00812 (no change)")
                return "00812"
            elif colonoscopy_is_screening and polyps_found:
                print(f"   Colonoscopy correction: Non-Medicare + screening + polyps found -> 00812 (no change)")
                return "00812"
            else:  # not screening (polyps don't matter)
                print(f"   Colonoscopy correction: Non-Medicare + not screening -> 00811 (CHANGED)")
                return "00811"
        
    except Exception as e:
        print(f"   Warning: Colonoscopy correction failed: {str(e)}, keeping original code")
        return asa_code


def determine_modifier(has_md, has_crna, medicare_modifiers, medical_direction):
    """
    Determine the modifier to apply based on provider presence and rules.
    
    Returns the M1 modifier string.
    
    Rules:
    YES medical direction YES medicare modifiers
    - MD and CRNA -> QK
    - just MD -> AA
    - just CRNA -> QZ
    
    YES medical direction, NO medicare modifiers
    - MD and CRNA -> QK
    - just MD -> QK
    - just CRNA -> QX
    
    NO medical direction, YES medicare modifiers
    - MD and CRNA -> QZ
    - just MD -> AA
    - just CRNA -> QZ
    
    NO medical direction, NO medicare modifiers
    - MD and CRNA -> nothing
    - just MD -> nothing
    - just CRNA -> nothing
    """
    
    # NO medical direction, NO medicare modifiers
    if not medical_direction and not medicare_modifiers:
        return ''
    
    # YES medical direction YES medicare modifiers
    if medical_direction and medicare_modifiers:
        if has_md and has_crna:
            return 'QK'
        elif has_md:
            return 'AA'
        elif has_crna:
            return 'QZ'
        else:
            return ''
    
    # YES medical direction, NO medicare modifiers
    if medical_direction and not medicare_modifiers:
        if has_md and has_crna:
            return 'QK'
        elif has_md:
            return 'QK'
        elif has_crna:
            return 'QX'
        else:
            return ''
    
    # NO medical direction, YES medicare modifiers
    if not medical_direction and medicare_modifiers:
        if has_md and has_crna:
            return 'QZ'
        elif has_md:
            return 'AA'
        elif has_crna:
            return 'QZ'
        else:
            return ''
    
    # Default case (should not reach here)
    return ''


def generate_modifiers(input_file, output_file=None, turn_off_medical_direction=False, generate_qk_duplicate=False, limit_anesthesia_time=False, turn_off_bcbs_medicare_modifiers=True, peripheral_blocks_mode="other", add_pt_for_non_medicare=False, change_responsible_provider_to_md_if_p_only=False):
    """
    Main function to generate modifiers for medical billing.
    Reads input CSV, processes each row, and generates appropriate modifiers.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        turn_off_medical_direction: If True, override all medical direction YES to NO
        generate_qk_duplicate: If True, generate duplicate line when QK modifier is applied with CRNA as Responsible Provider
        limit_anesthesia_time: If True, limit anesthesia time to maximum 480 minutes based on An Start and An Stop columns
        turn_off_bcbs_medicare_modifiers: If True, for MedNet Code 003 (BCBS), only generate P modifiers (no M1/GC/QS)
        peripheral_blocks_mode: Mode for peripheral block generation
            - "UNI": Generate blocks ONLY when Anesthesia Type is "General"
            - "other": Generate blocks when Anesthesia Type is NOT "MAC" (default)
        add_pt_for_non_medicare: If True, add PT modifier for non-Medicare insurances when polyps found and screening colonoscopy
        change_responsible_provider_to_md_if_p_only: If True, change Responsible Provider to MD when only P modifier is used and both MD and CRNA are present
    """
    try:
        # Load modifiers definition
        modifiers_dict = load_modifiers_definition()
        
        # DEBUG: Check if specific codes are loaded
        if '3136' in modifiers_dict:
            print(f"\n✅ Code 3136 loaded from DB: medicare_modifiers={modifiers_dict['3136'][0]}, medical_direction={modifiers_dict['3136'][1]}")
        else:
            print(f"\n⚠️  Code 3136 NOT found in modifiers_dict")
        
        if '003' in modifiers_dict:
            print(f"✅ Code 003 loaded from DB: medicare_modifiers={modifiers_dict['003'][0]}, medical_direction={modifiers_dict['003'][1]}\n")
        else:
            print(f"⚠️  Code 003 NOT found in modifiers_dict\n")
        
        # Log medical direction override mode
        if turn_off_medical_direction:
            print("=" * 80)
            print("⚠️  MEDICAL DIRECTION OVERRIDE MODE ENABLED ⚠️")
            print("All medical direction YES values will be treated as NO")
            print("=" * 80)
        
        # Log QK duplicate generation mode
        if generate_qk_duplicate:
            print("=" * 80)
            print("🔄 QK DUPLICATE LINE GENERATION ENABLED 🔄")
            print("Duplicate lines will be created for QK modifiers with CRNA as Responsible Provider")
            print("=" * 80)
        
        # Log anesthesia time limiting mode
        if limit_anesthesia_time:
            print("=" * 80)
            print("⏱️  ANESTHESIA TIME LIMITING ENABLED ⏱️")
            print("Anesthesia time will be limited to maximum 480 minutes (8 hours)")
            print("Time limiting will ONLY apply to rows where ASA Code = 01967")
            print("=" * 80)
        
        # Log BCBS Medicare modifiers mode
        if turn_off_bcbs_medicare_modifiers:
            print("=" * 80)
            print("🔵 BCBS MEDICARE MODIFIERS TURNED OFF 🔵")
            print("For MedNet Code 003 (Blue Cross), ONLY P modifiers will be generated")
            print("No M1, GC, or QS modifiers will be generated for code 003")
            print("=" * 80)
        
        # Log peripheral blocks mode
        print("=" * 80)
        if peripheral_blocks_mode == "UNI":
            print("📋 PERIPHERAL BLOCKS MODE: UNI")
            print("Peripheral blocks will be generated ONLY when Anesthesia Type = 'General'")
        else:
            print("📋 PERIPHERAL BLOCKS MODE: Other Groups")
            print("Peripheral blocks will be generated when Anesthesia Type is NOT 'MAC'")
        print("=" * 80)
        
        # Log PT modifier for non-Medicare mode
        if add_pt_for_non_medicare:
            print("=" * 80)
            print("🏥 PT MODIFIER FOR NON-MEDICARE ENABLED 🏥")
            print("PT modifier will be added for non-Medicare insurances when polyps found and screening colonoscopy")
            print("=" * 80)
        
        # Log change Responsible Provider to MD if P only mode
        if change_responsible_provider_to_md_if_p_only:
            print("=" * 80)
            print("👤 CHANGE RESPONSIBLE PROVIDER TO MD IF P ONLY ENABLED 👤")
            print("Responsible Provider will be set to MD value when ONLY P modifier is used and both MD and CRNA are present")
            print("=" * 80)
        
        # Load insurances.csv for PT modifier Medicare check
        insurances_df = pd.DataFrame()
        try:
            script_dir = Path(__file__).parent
            insurances_path = script_dir.parent / "insurances.csv"
            if insurances_path.exists():
                insurances_df = pd.read_csv(insurances_path, dtype=str)
                print(f"Loaded {len(insurances_df)} insurance records for PT modifier Medicare check")
            else:
                print(f"Warning: insurances.csv not found at {insurances_path}, PT modifier Medicare check may not work properly")
        except Exception as e:
            print(f"Warning: Failed to load insurances.csv: {str(e)}, PT modifier Medicare check may not work properly")
        
        # Read the input CSV file with dtype=str to preserve leading zeros in MedNet codes
        try:
            df = pd.read_csv(input_file, encoding='utf-8', dtype=str)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(input_file, encoding='latin-1', dtype=str)
            except Exception as e:
                raise Exception(f"Could not read CSV with utf-8 or latin-1 encoding: {e}")
        
        if len(df) < 1:
            print("Error: CSV file must have at least 1 row (data)")
            return False
        
        # Check for required columns
        required_columns = ['Primary Mednet Code']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Missing required column '{col}'")
                return False
        
        # Check if MD and CRNA columns exist
        has_md_column = 'MD' in df.columns
        has_crna_column = 'CRNA' in df.columns
        
        if not has_md_column and not has_crna_column:
            print("Warning: Neither 'MD' nor 'CRNA' columns found. No modifiers will be generated.")
        
        # Ensure M1, M2, M3, M4 columns exist
        if 'M1' not in df.columns:
            df['M1'] = ''
        if 'M2' not in df.columns:
            df['M2'] = ''
        if 'M3' not in df.columns:
            df['M3'] = ''
        if 'M4' not in df.columns:
            df['M4'] = ''
        
        # Check for Anesthesia Type, Physical Status, Resident, Polyps found, and colonoscopy screening columns
        has_anesthesia_type = 'Anesthesia Type' in df.columns
        has_physical_status = 'Physical Status' in df.columns
        has_resident_column = 'Resident' in df.columns
        has_polyps_found_column = 'Polyps found' in df.columns
        has_colonoscopy_screening_column = 'colonoscopy_is_screening' in df.columns
        
        # Check for peripheral_blocks column
        has_peripheral_blocks = 'peripheral_blocks' in df.columns
        
        # Check for TEE columns
        has_tee_was_done = 'tee_was_done' in df.columns
        has_tee_diagnosis = 'tee_diagnosis' in df.columns
        
        # Check for emergent_case column
        has_emergent_case = 'emergent_case' in df.columns
        
        # Debug: Log column availability
        if not has_peripheral_blocks:
            print("⚠️  Warning: 'peripheral_blocks' column not found. Peripheral block rows will not be generated.")
        elif has_peripheral_blocks:
            if has_anesthesia_type:
                if peripheral_blocks_mode == "UNI":
                    print(f"✅ Found 'peripheral_blocks' column. Peripheral block rows will be generated ONLY when Anesthesia Type = 'General' (UNI mode)")
                else:
                    print(f"✅ Found 'peripheral_blocks' column. Peripheral block rows will be generated for all Anesthesia Types EXCEPT 'MAC' (Other Groups mode)")
            else:
                print(f"✅ Found 'peripheral_blocks' column. Peripheral block rows will be generated (Anesthesia Type column not found)")
        
        # Process each row
        result_rows = []
        successful_matches = 0
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            new_row = row.copy()
            
            # Reset M1, M2, M3, M4 for this row
            new_row['M1'] = ''
            new_row['M2'] = ''
            new_row['M3'] = ''
            new_row['M4'] = ''
            
            # Apply colonoscopy correction if ASA Code or Procedure Code is 00812
            asa_code = str(new_row.get('ASA Code', '')).strip()
            procedure_code = str(new_row.get('Procedure Code', '')).strip()
            
            if asa_code == '00812' or procedure_code == '00812':
                # Apply correction to ASA Code
                if asa_code == '00812':
                    corrected_asa = apply_colonoscopy_correction(row, asa_code, insurances_df)
                    if corrected_asa != asa_code:
                        new_row['ASA Code'] = corrected_asa
                        print(f"Row {idx + 1}: Colonoscopy correction applied - ASA Code changed from {asa_code} to {corrected_asa}")
                
                # Apply correction to Procedure Code (should match ASA Code)
                if procedure_code == '00812':
                    corrected_procedure = apply_colonoscopy_correction(row, procedure_code, insurances_df)
                    if corrected_procedure != procedure_code:
                        new_row['Procedure Code'] = corrected_procedure
                        print(f"Row {idx + 1}: Colonoscopy correction applied - Procedure Code changed from {procedure_code} to {corrected_procedure}")
            
            # Apply anesthesia time limiting if enabled AND ASA Code is 01967
            if limit_anesthesia_time and 'An Start' in new_row and 'An Stop' in new_row:
                # Check if ASA Code is 01967
                asa_code = str(new_row.get('ASA Code', '')).strip()
                if asa_code == '01967':
                    an_start = new_row.get('An Start', '')
                    an_stop = new_row.get('An Stop', '')
                    adjusted_stop, was_limited = calculate_and_limit_anesthesia_time(an_start, an_stop, max_minutes=480)
                    if was_limited:
                        new_row['An Stop'] = adjusted_stop
                        print(f"Row {idx + 1}: Limited anesthesia time (ASA Code 01967) - An Start: {an_start}, Original An Stop: {an_stop}, Adjusted An Stop: {adjusted_stop}")
            
            primary_mednet_code = str(row.get('Primary Mednet Code', '')).strip()
            
            # Variables to track modifiers
            m1_modifier = ''
            gc_modifier = ''
            qs_modifier = ''
            p_modifier = ''
            pt_modifier = ''
            medicare_modifiers = False
            has_md = False
            has_crna = False
            enable_qs = True  # Default to True if not set
            
            # Determine M1 modifier (AA/QK/QZ) based on mednet code
            if primary_mednet_code and primary_mednet_code != '' and primary_mednet_code.lower() != 'nan':
                # Look up the code in modifiers definition
                if primary_mednet_code in modifiers_dict:
                    # Code found in definition - increment successful matches
                    successful_matches += 1
                    
                    # Get the modifiers settings
                    modifiers_tuple = modifiers_dict[primary_mednet_code]
                    medicare_modifiers = modifiers_tuple[0]
                    medical_direction = modifiers_tuple[1]
                    enable_qs = modifiers_tuple[2] if len(modifiers_tuple) > 2 else True  # Default to True if not set
                    
                    # DEBUG LOGGING for specific codes
                    if primary_mednet_code in ['3136', '003']:
                        print(f"\n🔍 DEBUG Row {idx + 1} - MedNet Code: {primary_mednet_code}")
                        print(f"   From DB: medicare_modifiers={medicare_modifiers}, medical_direction={medical_direction}, enable_qs={enable_qs}")
                    
                    # Override medical direction if turn_off_medical_direction is enabled
                    if turn_off_medical_direction:
                        medical_direction = False
                        if primary_mednet_code in ['3136', '003']:
                            print(f"   Override: medical_direction={medical_direction} (turn_off_medical_direction=True)")
                    
                    # Check if MD and CRNA have values
                    if has_md_column:
                        md_value = row.get('MD', '')
                        if not pd.isna(md_value) and str(md_value).strip() != '':
                            has_md = True
                    
                    if has_crna_column:
                        crna_value = row.get('CRNA', '')
                        if not pd.isna(crna_value) and str(crna_value).strip() != '':
                            has_crna = True
                    
                    # DEBUG LOGGING for provider values
                    if primary_mednet_code in ['3136', '003']:
                        print(f"   Providers: has_md={has_md}, has_crna={has_crna}")
                        if has_md:
                            print(f"   MD value: '{row.get('MD', '')}'")
                        if has_crna:
                            print(f"   CRNA value: '{row.get('CRNA', '')}'")
                    
                    # SPECIAL CASE: Mednet code 003 (Blue Cross)
                    # Override modifiers settings based on MD and CRNA presence
                    # Note: If turn_off_medical_direction is True, we don't allow medical_direction to be set to True
                    # Note: If turn_off_bcbs_medicare_modifiers is True, only generate P modifiers (set both to NO)
                    if primary_mednet_code == '003':
                        if turn_off_bcbs_medicare_modifiers:
                            # BCBS Medicare modifiers turned off: only generate P modifiers
                            medicare_modifiers = False
                            medical_direction = False
                            print(f"   Special Case 003: BCBS Medicare Modifiers OFF - Only P modifiers will be generated")
                        elif has_md and has_crna:
                            # Both MD and CRNA present: artificially set both to YES
                            medicare_modifiers = True
                            # Only set medical_direction to True if turn_off_medical_direction is False
                            if not turn_off_medical_direction:
                                medical_direction = True
                        else:
                            # NOT both present: artificially set both to NO
                            medicare_modifiers = False
                            medical_direction = False
                        
                        if not turn_off_bcbs_medicare_modifiers:
                            print(f"   Special Case 003: medicare_modifiers={medicare_modifiers}, medical_direction={medical_direction}")
                    
                    # Determine M1 modifier (AA/QK/QZ)
                    m1_modifier = determine_modifier(has_md, has_crna, medicare_modifiers, medical_direction)
                    
                    # Check if original row has an existing M1 value - if so, use it instead of generated value
                    original_m1 = row.get('M1', '')
                    if original_m1 and not pd.isna(original_m1) and str(original_m1).strip() != '':
                        generated_m1 = m1_modifier  # Store the generated value for debug logging
                        m1_modifier = str(original_m1).strip()
                        if primary_mednet_code in ['3136', '003']:
                            print(f"   M1 modifier overridden with original value: '{m1_modifier}' (was generated: '{generated_m1}')")
                    
                    # DEBUG LOGGING for M1 modifier
                    if primary_mednet_code in ['3136', '003']:
                        print(f"   M1 modifier determined: '{m1_modifier}'")
                else:
                    # Code not found in modifiers_dict
                    if primary_mednet_code in ['3136', '003']:
                        print(f"\n❌ DEBUG Row {idx + 1} - MedNet Code: {primary_mednet_code}")
                        print(f"   Code NOT FOUND in modifiers_dict (loaded {len(modifiers_dict)} codes)")
            
            # Determine GC modifier based on Resident AND medicare modifiers
            # BUT NOT when m1_modifier is QK (prevent QK + GC combination)
            if has_resident_column and medicare_modifiers and m1_modifier != 'QK':
                resident_value = row.get('Resident', '')
                if not pd.isna(resident_value) and str(resident_value).strip() != '':
                    gc_modifier = 'GC'
                    if primary_mednet_code in ['3136', '003']:
                        print(f"   GC modifier added (Resident: '{resident_value}')")
            
            # Determine QS modifier based on Anesthesia Type AND medicare modifiers AND enable_qs setting
            if has_anesthesia_type and medicare_modifiers:
                if enable_qs:
                    anesthesia_type = str(row.get('Anesthesia Type', '')).strip().upper()
                    if anesthesia_type == 'MAC':
                        qs_modifier = 'QS'
                        if primary_mednet_code in ['3136', '003']:
                            print(f"   QS modifier added (Anesthesia Type: MAC, enable_qs=True)")
                    elif primary_mednet_code in ['3136', '003']:
                        print(f"   QS modifier NOT added (Anesthesia Type: '{anesthesia_type}', expected 'MAC')")
                else:
                    if primary_mednet_code in ['3136', '003']:
                        print(f"   QS modifier NOT added (enable_qs=False for this insurance)")
            
            # Determine P modifier based on Physical Status
            if has_physical_status:
                physical_status = str(row.get('Physical Status', '')).strip()
                if physical_status and physical_status != '' and physical_status.lower() != 'nan':
                    try:
                        # Convert to integer to validate it's a number
                        status_num = int(float(physical_status))
                        p_modifier = f'P{status_num}'
                        if primary_mednet_code in ['3136', '003']:
                            print(f"   P modifier: {p_modifier} (Physical Status: {physical_status})")
                    except (ValueError, TypeError):
                        # If Physical Status is not a valid number, skip P modifier
                        if primary_mednet_code in ['3136', '003']:
                            print(f"   P modifier NOT added (Invalid Physical Status: '{physical_status}')")
                        pass
            
            # Determine PT modifier based on Polyps found AND colonoscopy_is_screening = TRUE
            # PT modifier requires Medicare Modifiers = YES for the insurance
            # If add_pt_for_non_medicare is True, PT modifier can also be added for non-Medicare insurances (but still requires Medicare Modifiers = YES)
            if has_polyps_found_column and has_colonoscopy_screening_column:
                polyps_value = str(row.get('Polyps found', '')).strip().upper()
                colonoscopy_screening = str(row.get('colonoscopy_is_screening', '')).strip().upper()
                
                # Check if insurance is Medicare (for logging/debugging purposes)
                is_medicare = False
                if primary_mednet_code and not insurances_df.empty:
                    # Find the insurance plan by MedNet Code
                    insurance_match = insurances_df[insurances_df['MedNet Code'].astype(str).str.strip() == primary_mednet_code]
                    if not insurance_match.empty:
                        insurance_plan = str(insurance_match.iloc[0].get('Insurance Plan', '')).strip()
                        if 'Medicare' in insurance_plan or 'MEDICARE' in insurance_plan or 'medicare' in insurance_plan:
                            is_medicare = True
                
                # PT is added when:
                # 1. Polyps found = "FOUND" AND colonoscopy_is_screening = "TRUE"
                # 2. Medicare Modifiers = YES (from modifiers_dict lookup)
                # 3. AND either: (a) it's Medicare insurance, OR (b) add_pt_for_non_medicare is True and it's NOT Medicare
                should_add_pt = False
                if polyps_value == 'FOUND' and colonoscopy_screening == 'TRUE' and medicare_modifiers:
                    if is_medicare:
                        # Add PT for Medicare when conditions are met and Medicare Modifiers = YES
                        should_add_pt = True
                    elif add_pt_for_non_medicare:
                        # Add PT for non-Medicare when option is enabled and Medicare Modifiers = YES
                        should_add_pt = True
                
                if should_add_pt:
                    pt_modifier = 'PT'
                    if primary_mednet_code in ['3136', '003', '00812']:
                        medicare_status = "Medicare=True" if is_medicare else "Non-Medicare=True"
                        print(f"   PT modifier: added (Polyps=FOUND, Screening=TRUE, Medicare Modifiers=YES, {medicare_status})")
                elif primary_mednet_code in ['3136', '003', '00812'] and (polyps_value == 'FOUND' or colonoscopy_screening == 'TRUE'):
                    medicare_status = "Medicare=True" if is_medicare else "Non-Medicare=True"
                    medicare_modifiers_status = "YES" if medicare_modifiers else "NO"
                    print(f"   PT modifier: NOT added (Polyps={polyps_value}, Screening={colonoscopy_screening}, Medicare Modifiers={medicare_modifiers_status}, {medicare_status}, add_pt_for_non_medicare={add_pt_for_non_medicare})")
            
            # Apply hierarchy: M1 (AA/QK/QZ/QX) > M2 (GC) > M3 (QS) > M4 (P) > PT (goes in LAST position)
            # Place modifiers sequentially without gaps
            # PT modifier always goes in the LAST available position (no gaps allowed)
            
            # Collect all modifiers in priority order (excluding PT)
            modifiers_list = []
            if m1_modifier:
                modifiers_list.append(m1_modifier)
            if gc_modifier:
                modifiers_list.append(gc_modifier)
            if qs_modifier:
                modifiers_list.append(qs_modifier)
            if p_modifier:
                modifiers_list.append(p_modifier)
            
            # Add PT modifier to the end of the list (it goes in the last position)
            if pt_modifier:
                modifiers_list.append(pt_modifier)
            
            # Place modifiers sequentially in M1, M2, M3, M4 without gaps
            # PT will automatically be in the last position since it's added last
            if len(modifiers_list) >= 1:
                new_row['M1'] = modifiers_list[0]
            if len(modifiers_list) >= 2:
                new_row['M2'] = modifiers_list[1]
            if len(modifiers_list) >= 3:
                new_row['M3'] = modifiers_list[2]
            if len(modifiers_list) >= 4:
                new_row['M4'] = modifiers_list[3]
            
            # DEBUG LOGGING for final modifiers assigned
            if primary_mednet_code in ['3136', '003']:
                print(f"   Final Modifiers: M1='{new_row['M1']}', M2='{new_row['M2']}', M3='{new_row['M3']}', M4='{new_row['M4']}'")
                print(f"   medicare_modifiers={medicare_modifiers}, medical_direction={medical_direction}\n")
            
            # Check if we need to change Responsible Provider to MD when only P modifier is used
            # Conditions:
            # 1. change_responsible_provider_to_md_if_p_only is True
            # 2. Only P modifier is used (no provider type modifiers AA/QK/QZ/QX, no GC, no QS)
            #    - M1 has P modifier (P1-P6) - this means no AA/QK/QZ/QX modifiers
            #    - M3 is empty - this means no GC or QS modifiers
            #    - M2 can be empty or PT (if PT modifier is also present)
            # 3. Both MD and CRNA columns are filled out
            if change_responsible_provider_to_md_if_p_only:
                m1_value = str(new_row.get('M1', '')).strip()
                m2_value = str(new_row.get('M2', '')).strip()
                m3_value = str(new_row.get('M3', '')).strip()
                
                # Helper function to check if a value is a P modifier (P1-P6)
                def is_p_modifier(val):
                    return (
                        val and 
                        val.startswith('P') and 
                        len(val) == 2 and 
                        val[1].isdigit() and 
                        1 <= int(val[1]) <= 6
                    )
                
                # Check if only P modifier is used (no AA/QK/QZ/QX, no GC, no QS)
                # P modifier will be in M1 when it's the only modifier (or with PT)
                # M3 empty means no GC or QS modifiers
                # M2 can be empty or PT (if PT modifier is also present)
                only_p_modifier = (
                    is_p_modifier(m1_value) and  # M1 has P modifier (no AA/QK/QZ/QX)
                    not m3_value and  # M3 empty (no GC, no QS)
                    (not m2_value or m2_value == 'PT')  # M2 empty or PT (no GC modifier)
                )
                
                # Check if both MD and CRNA are filled
                md_value = ''
                crna_value = ''
                if has_md_column:
                    md_value = row.get('MD', '')
                    if pd.isna(md_value) or str(md_value).strip() == '':
                        md_value = ''
                if has_crna_column:
                    crna_value = row.get('CRNA', '')
                    if pd.isna(crna_value) or str(crna_value).strip() == '':
                        crna_value = ''
                
                both_providers_filled = bool(md_value and crna_value)
                
                # If conditions are met, set Responsible Provider to MD value
                if only_p_modifier and both_providers_filled:
                    if 'Responsible Provider' in new_row:
                        new_row['Responsible Provider'] = str(md_value).strip()
                        print(f"Row {idx + 1}: Changed Responsible Provider to MD value '{str(md_value).strip()}' (only P modifier used, both MD and CRNA present)")
            
            # Special case: If ASA Code is 01967 in main charge, clear ICD1-ICD4 and set ICD1 = "O80"
            asa_code = str(new_row.get('ASA Code', '')).strip()
            if asa_code == '01967':
                # Clear all ICD codes
                for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                    if icd_col in new_row:
                        new_row[icd_col] = ''
                # Set ICD1 to "O80"
                if 'ICD1' in new_row:
                    new_row['ICD1'] = 'O80'
                    print(f"Row {idx + 1}: ASA Code 01967 detected - cleared ICD1-ICD4 and set ICD1 = 'O80'")
            
            result_rows.append(new_row)
            
            # Check if we need to create a QK duplicate row
            # Conditions:
            # 1. generate_qk_duplicate is True
            # 2. M1 modifier is QK
            # 3. CRNA field has a value
            if generate_qk_duplicate and m1_modifier == 'QK' and has_crna:
                # Create a duplicate row
                qk_duplicate_row = new_row.copy()
                
                # Get the CRNA value
                crna_value = row.get('CRNA', '') if has_crna_column else ''
                
                # Set the Responsible Provider to the CRNA value
                if 'Responsible Provider' in qk_duplicate_row:
                    qk_duplicate_row['Responsible Provider'] = crna_value
                
                # Change M1 modifier from QK to QX
                qk_duplicate_row['M1'] = 'QX'
                
                # Add the duplicate row to results
                result_rows.append(qk_duplicate_row)
            
            # Check if we need to create a TEE row
            # Conditions:
            # 1. tee_was_done = TRUE
            # 2. tee_diagnosis has diagnosis codes (not empty)
            if has_tee_was_done and has_tee_diagnosis:
                tee_was_done_value = str(row.get('tee_was_done', '')).strip().upper()
                tee_diagnosis_value = row.get('tee_diagnosis', '')
                
                if tee_was_done_value == 'TRUE':
                    # Parse the TEE diagnosis codes
                    tee_diagnosis_codes = parse_tee_diagnosis(tee_diagnosis_value)
                    
                    # Only create the row if there are diagnosis codes
                    if tee_diagnosis_codes:
                        # Create a copy of the original input row (not the modified new_row)
                        tee_row = row.copy()
                        
                        # Set ASA Code and Procedure Code to 93312
                        tee_row['ASA Code'] = '93312'
                        tee_row['Procedure Code'] = '93312'
                        
                        # Set M1 = "26" and M2 = "59" (hardcoded)
                        tee_row['M1'] = '26'
                        tee_row['M2'] = '59'
                        tee_row['M3'] = ''
                        tee_row['M4'] = ''
                        
                        # Populate ICD1-ICD4 from tee_diagnosis codes in order
                        for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                            if icd_col in tee_row:
                                tee_row[icd_col] = ''
                        
                        # Populate ICD codes from diagnosis list (up to 4 codes)
                        if len(tee_diagnosis_codes) >= 1 and 'ICD1' in tee_row:
                            tee_row['ICD1'] = tee_diagnosis_codes[0]
                        if len(tee_diagnosis_codes) >= 2 and 'ICD2' in tee_row:
                            tee_row['ICD2'] = tee_diagnosis_codes[1]
                        if len(tee_diagnosis_codes) >= 3 and 'ICD3' in tee_row:
                            tee_row['ICD3'] = tee_diagnosis_codes[2]
                        if len(tee_diagnosis_codes) >= 4 and 'ICD4' in tee_row:
                            tee_row['ICD4'] = tee_diagnosis_codes[3]
                        
                        # Clear Concurrent Providers
                        if 'Concurrent Providers' in tee_row:
                            tee_row['Concurrent Providers'] = ''
                        
                        # Clear An Start and An Stop columns (keep only in original row)
                        if 'An Start' in tee_row:
                            tee_row['An Start'] = ''
                        if 'An Stop' in tee_row:
                            tee_row['An Stop'] = ''
                        
                        # Clear SRNA field
                        if 'SRNA' in tee_row:
                            tee_row['SRNA'] = ''
                        
                        # Clear Anesthesia Type field
                        if 'Anesthesia Type' in tee_row:
                            tee_row['Anesthesia Type'] = ''
                        
                        # Add the TEE row to results
                        result_rows.append(tee_row)
            
            # Check if we need to create an emergent_case row
            # Conditions:
            # 1. emergent_case column exists
            # 2. emergent_case = TRUE
            # 3. Insurance is NOT Medicare, Medicare replacement, or Medicaid (commercial insurance only)
            if has_emergent_case:
                emergent_case_value = str(row.get('emergent_case', '')).strip().upper()
                
                if emergent_case_value == 'TRUE':
                    # Check if insurance is Medicare or Medicaid
                    # Check both Insurance Name and Insurance Plan fields
                    is_medicare_or_medicaid = False
                    if primary_mednet_code and not insurances_df.empty:
                        # Find the insurance plan by MedNet Code
                        insurance_match = insurances_df[insurances_df['MedNet Code'].astype(str).str.strip() == primary_mednet_code]
                        if not insurance_match.empty:
                            # Check Insurance Plan field
                            insurance_plan = str(insurance_match.iloc[0].get('Insurance Plan', '')).strip().upper()
                            # Check Insurance Name field
                            insurance_name = str(insurance_match.iloc[0].get('Name', '')).strip().upper()
                            # Check if either field contains "medicare" or "medicaid" (case-insensitive)
                            if ('MEDICARE' in insurance_plan or 'MEDICAID' in insurance_plan or
                                'MEDICARE' in insurance_name or 'MEDICAID' in insurance_name):
                                is_medicare_or_medicaid = True
                    
                    # Only add 99140 for commercial insurance (NOT Medicare/Medicaid)
                    if not is_medicare_or_medicaid:
                        # Create a copy of the original input row (not the modified new_row)
                        emergent_row = row.copy()
                    
                        # Set ASA Code and Procedure Code to 99140
                        emergent_row['ASA Code'] = '99140'
                        emergent_row['Procedure Code'] = '99140'
                    
                        # Clear all modifiers (M1-M4)
                        emergent_row['M1'] = ''
                        emergent_row['M2'] = ''
                        emergent_row['M3'] = ''
                        emergent_row['M4'] = ''
                    
                        # Keep ICD1-ICD4 from original row (don't clear them)
                        # ICD codes are already copied from row.copy(), so they're preserved
                    
                        # Clear Concurrent Providers
                        if 'Concurrent Providers' in emergent_row:
                            emergent_row['Concurrent Providers'] = ''
                        
                        # Clear An Start and An Stop columns (keep only in original row)
                        if 'An Start' in emergent_row:
                            emergent_row['An Start'] = ''
                        if 'An Stop' in emergent_row:
                            emergent_row['An Stop'] = ''
                        
                        # Keep SRNA field (don't clear it for emergent_case)
                        
                        # Clear Anesthesia Type field
                        if 'Anesthesia Type' in emergent_row:
                            emergent_row['Anesthesia Type'] = ''
                    
                        # Add the emergent_case row to results
                        result_rows.append(emergent_row)
            
            # Check if we need to create peripheral block rows
            # Conditions depend on peripheral_blocks_mode:
            # - "UNI": Generate ONLY when Anesthesia Type = "General"
            # - "other": Generate when Anesthesia Type is NOT "MAC"
            if has_peripheral_blocks:
                peripheral_blocks_value = row.get('peripheral_blocks', '')
                
                # Check Anesthesia Type only if the column exists
                anesthesia_type_val = ''
                if has_anesthesia_type:
                    anesthesia_type_val = str(row.get('Anesthesia Type', '')).strip().upper()
                
                # Determine if blocks should be generated based on mode
                should_generate_blocks = False
                if peripheral_blocks_mode == "UNI":
                    # UNI mode: Generate ONLY when Anesthesia Type is "General"
                    should_generate_blocks = (anesthesia_type_val == 'GENERAL')
                else:
                    # Other mode: Generate when Anesthesia Type is NOT "MAC"
                    should_generate_blocks = (anesthesia_type_val != 'MAC')
                
                if should_generate_blocks:
                    # Check if peripheral_blocks contains "NONE DONE" - if so, skip block generation
                    peripheral_blocks_str = str(peripheral_blocks_value).strip().upper()
                    if 'NONE DONE' in peripheral_blocks_str or 'NONEDONE' in peripheral_blocks_str:
                        # Skip all block generation when "NONE DONE" is present
                        print(f"ℹ️  Row {idx}: Skipping peripheral blocks - peripheral_blocks contains 'NONE DONE'")
                    else:
                        # Parse the peripheral blocks
                        blocks = parse_peripheral_blocks(peripheral_blocks_value)
                        
                        # Debug logging for peripheral blocks
                        if peripheral_blocks_value and str(peripheral_blocks_value).strip():
                            if not blocks:
                                print(f"⚠️  Row {idx}: peripheral_blocks field has value but parsed to 0 blocks: '{peripheral_blocks_value}'")
                        
                        # Define peripheral nerve block CPT codes (needed for skip check)
                        peripheral_nerve_blocks = [
                            '64445', '64446',  # Sciatic
                            '64415', '64416',  # Interscalene
                            '64417', '64418',  # Axillary (New)
                            '64447', '64448',  # Femoral
                            '64466', '64467', '64468', '64469',  # ESP
                            '64473',           # Fascial Plane (iPACK / PENG) (New)
                            '64488',           # TAP
                            '62322', '62323',  # Neuraxial/Spinal (New)
                        ]
                        
                        # Create a duplicate row for each block
                        for block_idx, block in enumerate(blocks):
                            cpt_code = block['cpt_code']
                            
                            # Skip ultrasound guidance (76937) if it comes after a peripheral nerve block
                            # (It's OK after arterial line 36620, but NOT after peripheral nerve blocks)
                            if cpt_code == '76937':
                                if block_idx > 0:
                                    previous_cpt_code = blocks[block_idx - 1]['cpt_code']
                                    # Skip if previous block was a peripheral nerve block
                                    if previous_cpt_code in peripheral_nerve_blocks:
                                        print(f"ℹ️  Row {idx}: Skipping ultrasound guidance (76937) - comes after peripheral nerve block ({previous_cpt_code})")
                                        continue
                            
                            # Create a copy of the original input row (not the modified new_row)
                            block_row = row.copy()
                            
                            # Set ASA Code and Procedure Code to the CPT code from the block
                            block_row['ASA Code'] = block['cpt_code']
                            block_row['Procedure Code'] = block['cpt_code']
                            
                            # Set provider information from the block
                            if has_md_column:
                                block_row['MD'] = block['md']
                            if has_resident_column:
                                block_row['Resident'] = block['resident']
                            if has_crna_column:
                                block_row['CRNA'] = block['crna']
                            
                            # Handle ICD codes based on CPT code
                            
                            # Special case: ASA Code 01967 or 1967
                            asa_code = str(block_row.get('ASA Code', '')).strip()
                            if asa_code in ['01967', '1967']:
                                # Clear all ICD codes and set ICD1 = "080"
                                for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                    if icd_col in block_row:
                                        block_row[icd_col] = ''
                                block_row['ICD1'] = '080'
                            
                            if cpt_code in peripheral_nerve_blocks:
                                # Peripheral nerve blocks: Clear all ICD codes, set ICD1 = "G89.18"
                                for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                    if icd_col in block_row:
                                        block_row[icd_col] = ''
                                block_row['ICD1'] = 'G89.18'
                            
                            elif cpt_code == '36620':
                                # Arterial line: Keep ICD codes from original input row (already in block_row)
                                pass
                            
                            elif cpt_code == '76937':
                                # Ultrasound guidance: Keep ICD codes ONLY if it comes after 36620
                                if block_idx > 0 and blocks[block_idx - 1]['cpt_code'] == '36620':
                                    # Keep ICD codes from original input row (already in block_row)
                                    pass
                                else:
                                    # Clear all ICD codes
                                    for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                        if icd_col in block_row:
                                            block_row[icd_col] = ''
                            
                            elif cpt_code in ['36556', '93503']:
                                # CVP: Clear all ICD codes
                                for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                    if icd_col in block_row:
                                        block_row[icd_col] = ''
                            
                            else:
                                # Any other CPT code: Clear all ICD codes
                                for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                    if icd_col in block_row:
                                        block_row[icd_col] = ''
                            
                            # Set Type Of Service based on block type
                            if 'Type Of Service' in block_row:
                                if cpt_code in peripheral_nerve_blocks:
                                    # Peripheral nerve blocks: SURGERY
                                    block_row['Type Of Service'] = 'SURGERY'
                                elif cpt_code == '36620':
                                    # Arterial line: SURGERY
                                    block_row['Type Of Service'] = 'SURGERY'
                                elif cpt_code == '76937':
                                    # Ultrasound guidance: DIAGNOSTIC RADIOLOGY
                                    block_row['Type Of Service'] = 'DIAGNOSTIC RADIOLOGY'
                            
                            # Clear Concurrent Providers
                            if 'Concurrent Providers' in block_row:
                                block_row['Concurrent Providers'] = ''
                            
                            # Clear An Start and An Stop columns (keep only in original row)
                            if 'An Start' in block_row:
                                block_row['An Start'] = ''
                            if 'An Stop' in block_row:
                                block_row['An Stop'] = ''
                            
                            # Clear SRNA field
                            if 'SRNA' in block_row:
                                block_row['SRNA'] = ''
                            
                            # Clear Anesthesia Type field
                            if 'Anesthesia Type' in block_row:
                                block_row['Anesthesia Type'] = ''
                            
                            # Set Responsible Provider to MD value from block
                            if 'Responsible Provider' in block_row:
                                block_row['Responsible Provider'] = block['md']
                            
                            # Clear all modifier columns and set M1 and M2 from the block
                            block_row['M1'] = block['m1']
                            block_row['M2'] = block['m2']
                            block_row['M3'] = ''
                            block_row['M4'] = ''
                            
                            # Add the block row to results
                            result_rows.append(block_row)
                else:
                    # Blocks were skipped - log reason based on mode
                    if peripheral_blocks_value and str(peripheral_blocks_value).strip():
                        if peripheral_blocks_mode == "UNI":
                            print(f"⚠️  Row {idx}: Skipping peripheral blocks - Anesthesia Type is '{row.get('Anesthesia Type', '')}' (UNI mode requires 'General')")
                        else:
                            print(f"⚠️  Row {idx}: Skipping peripheral blocks - Anesthesia Type is 'MAC'")
        
        # Create result dataframe
        result_df = pd.DataFrame(result_rows)
        
        # Save to output file(s)
        if output_file is None:
            output_file = input_file.replace('.csv', '_with_modifiers.csv')
        
        # Save in both CSV and XLSX formats
        try:
            # Remove extension to use base path
            base_path = Path(output_file).with_suffix('')
            csv_path, xlsx_path = save_dataframe_dual_format(result_df, base_path)
            print(f"Modifiers generation complete.")
            print(f"CSV output saved to: {csv_path}")
            if xlsx_path:
                print(f"XLSX output saved to: {xlsx_path}")
        except Exception as e:
            # Fallback to CSV only if dual format fails
            print(f"Warning: Could not save XLSX format ({e}), saving CSV only")
            result_df.to_csv(output_file, index=False)
            print(f"Modifiers generation complete. Output saved to: {output_file}")
        
        print(f"Processed {len(df)} input rows, generated {len(result_df)} output rows.")
        print(f"Successfully matched {successful_matches} out of {total_rows} MedNet codes ({successful_matches/total_rows*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for the script.
    """
    if len(sys.argv) < 2:
        print("Usage: python generate_modifiers.py <input_csv_file> [output_csv_file]")
        print("Example: python generate_modifiers.py billing_data.csv")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    print(f"Processing file: {input_file}")
    success = generate_modifiers(input_file, output_file)
    
    if success:
        print("Modifiers generation completed successfully!")
    else:
        print("Modifiers generation failed!")


if __name__ == "__main__":
    main()
