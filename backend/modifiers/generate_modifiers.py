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
5. PT modifier - Added to LAST position (M4) when "Polyps found" = "FOUND" AND "colonoscopy_is_screening" = "TRUE"
   NOTE: PT modifier does NOT require Medicare Modifiers

Special Cases:
- Mednet Code 003 (Blue Cross): 
  * If BOTH MD and CRNA are present: Artificially set Medicare Modifiers = YES and Medical Direction = YES
  * If NOT both MD and CRNA: Artificially set Medicare Modifiers = NO and Medical Direction = NO
  This allows normal modifier generation (including GC) when both providers are present,
  but limits to only P modifier when they are not.

Peripheral Blocks Row Generation:
- When "peripheral_blocks" field is non-empty AND "Anesthesia Type" = "General" (case insensitive)
- Creates duplicate rows for each block in the peripheral_blocks field
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
    - Peripheral nerve blocks (644XX, 64488): Clear all, set ICD1 = "G89.18"
    - Arterial line (36620): Copy ICD1-ICD4 from original input row
    - CVP/Ultrasound (36556, 93503, 76937): Clear all ICD1-ICD4
    - Other codes: Clear all ICD1-ICD4
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
    Load the modifiers definition CSV file.
    Returns a dictionary mapping mednet codes to (medicare_modifiers, medical_direction) tuples.
    """
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        definition_path = script_dir / definition_file
        
        df = pd.read_csv(definition_path)
        
        # Create dictionary mapping MedNet Code to (Medicare Modifiers, Bill Medical Direction)
        modifiers_dict = {}
        for _, row in df.iterrows():
            mednet_code = str(row['MedNet Code']).strip()
            medicare_modifiers = str(row['Medicare Modifiers']).strip().upper() == 'YES'
            medical_direction = str(row['Bill Medical Direction']).strip().upper() == 'YES'
            modifiers_dict[mednet_code] = (medicare_modifiers, medical_direction)
        
        print(f"Loaded {len(modifiers_dict)} modifiers definitions")
        return modifiers_dict
    
    except FileNotFoundError:
        print(f"Warning: {definition_file} not found. No modifiers will be generated.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading {definition_file}: {e}. No modifiers will be generated.")
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


def generate_modifiers(input_file, output_file=None):
    """
    Main function to generate modifiers for medical billing.
    Reads input CSV, processes each row, and generates appropriate modifiers.
    """
    try:
        # Load modifiers definition
        modifiers_dict = load_modifiers_definition()
        
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
            
            # Determine M1 modifier (AA/QK/QZ) based on mednet code
            if primary_mednet_code and primary_mednet_code != '' and primary_mednet_code.lower() != 'nan':
                # Look up the code in modifiers definition
                if primary_mednet_code in modifiers_dict:
                    # Code found in definition - increment successful matches
                    successful_matches += 1
                    
                    # Get the modifiers settings
                    medicare_modifiers, medical_direction = modifiers_dict[primary_mednet_code]
                    
                    # Check if MD and CRNA have values
                    if has_md_column:
                        md_value = row.get('MD', '')
                        if not pd.isna(md_value) and str(md_value).strip() != '':
                            has_md = True
                    
                    if has_crna_column:
                        crna_value = row.get('CRNA', '')
                        if not pd.isna(crna_value) and str(crna_value).strip() != '':
                            has_crna = True
                    
                    # SPECIAL CASE: Mednet code 003 (Blue Cross)
                    # Override modifiers settings based on MD and CRNA presence
                    if primary_mednet_code == '003':
                        if has_md and has_crna:
                            # Both MD and CRNA present: artificially set both to YES
                            medicare_modifiers = True
                            medical_direction = True
                        else:
                            # NOT both present: artificially set both to NO
                            medicare_modifiers = False
                            medical_direction = False
                    
                    # Determine M1 modifier (AA/QK/QZ)
                    m1_modifier = determine_modifier(has_md, has_crna, medicare_modifiers, medical_direction)
            
            # Determine GC modifier based on Resident AND medicare modifiers
            # BUT NOT when m1_modifier is QK (prevent QK + GC combination)
            if has_resident_column and medicare_modifiers and m1_modifier != 'QK':
                resident_value = row.get('Resident', '')
                if not pd.isna(resident_value) and str(resident_value).strip() != '':
                    gc_modifier = 'GC'
            
            # Determine QS modifier based on Anesthesia Type AND medicare modifiers
            if has_anesthesia_type and medicare_modifiers:
                anesthesia_type = str(row.get('Anesthesia Type', '')).strip().upper()
                if anesthesia_type == 'MAC':
                    qs_modifier = 'QS'
            
            # Determine P modifier based on Physical Status
            if has_physical_status:
                physical_status = str(row.get('Physical Status', '')).strip()
                if physical_status and physical_status != '' and physical_status.lower() != 'nan':
                    try:
                        # Convert to integer to validate it's a number
                        status_num = int(float(physical_status))
                        p_modifier = f'P{status_num}'
                    except (ValueError, TypeError):
                        # If Physical Status is not a valid number, skip P modifier
                        pass
            
            # Determine PT modifier based on Polyps found AND colonoscopy_is_screening = TRUE AND Medicare insurance
            # PT modifier requires Medicare insurance
            if has_polyps_found_column and has_colonoscopy_screening_column:
                polyps_value = str(row.get('Polyps found', '')).strip().upper()
                colonoscopy_screening = str(row.get('colonoscopy_is_screening', '')).strip().upper()
                
                # Check if insurance is Medicare
                is_medicare = False
                if primary_mednet_code and not insurances_df.empty:
                    # Find the insurance plan by MedNet Code
                    insurance_match = insurances_df[insurances_df['MedNet Code'].astype(str).str.strip() == primary_mednet_code]
                    if not insurance_match.empty:
                        insurance_plan = str(insurance_match.iloc[0].get('Insurance Plan', '')).strip()
                        if 'Medicare' in insurance_plan or 'MEDICARE' in insurance_plan or 'medicare' in insurance_plan:
                            is_medicare = True
                
                # PT is added only when polyps are found AND it's a screening colonoscopy AND it's Medicare
                if polyps_value == 'FOUND' and colonoscopy_screening == 'TRUE' and is_medicare:
                    pt_modifier = 'PT'
            
            # Apply hierarchy: M1 (AA/QK/QZ/QX) > M2 (GC) > M3 (QS) > M4 (P) > M5 (PT - goes in LAST position)
            # Place modifiers in first available slots (M1, M2, M3, M4)
            # PT modifier always goes in the LAST position (M4) if it exists
            
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
            
            # If PT modifier exists, ensure it goes in LAST position (M4)
            if pt_modifier:
                # Place first modifiers in M1, M2, M3, reserve M4 for PT
                if len(modifiers_list) >= 1:
                    new_row['M1'] = modifiers_list[0]
                if len(modifiers_list) >= 2:
                    new_row['M2'] = modifiers_list[1]
                if len(modifiers_list) >= 3:
                    new_row['M3'] = modifiers_list[2]
                # M4 is reserved for PT
                new_row['M4'] = pt_modifier
            else:
                # No PT modifier, place all modifiers normally
                if len(modifiers_list) >= 1:
                    new_row['M1'] = modifiers_list[0]
                if len(modifiers_list) >= 2:
                    new_row['M2'] = modifiers_list[1]
                if len(modifiers_list) >= 3:
                    new_row['M3'] = modifiers_list[2]
                if len(modifiers_list) >= 4:
                    new_row['M4'] = modifiers_list[3]
            
            result_rows.append(new_row)
            
            # Check if we need to create peripheral block rows
            # Conditions:
            # 1. peripheral_blocks field is non-empty
            # 2. Anesthesia Type = "General" (case insensitive)
            if has_peripheral_blocks and has_anesthesia_type:
                peripheral_blocks_value = row.get('peripheral_blocks', '')
                anesthesia_type_val = str(row.get('Anesthesia Type', '')).strip().upper()
                
                if anesthesia_type_val == 'GENERAL':
                    # Parse the peripheral blocks
                    blocks = parse_peripheral_blocks(peripheral_blocks_value)
                    
                    # Create a duplicate row for each block
                    for block in blocks:
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
                        cpt_code = block['cpt_code']
                        
                        # Define peripheral nerve block CPT codes
                        peripheral_nerve_blocks = [
                            '64445', '64446',  # Sciatic
                            '64415', '64416',  # Interscalene
                            '64447', '64448',  # Femoral
                            '64466', '64467', '64468', '64469',  # ESP
                            '64488'  # TAP
                        ]
                        
                        if cpt_code in peripheral_nerve_blocks:
                            # Peripheral nerve blocks: Clear all ICD codes, set ICD1 = "G89.18"
                            for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                if icd_col in block_row:
                                    block_row[icd_col] = ''
                            block_row['ICD1'] = 'G89.18'
                        
                        elif cpt_code == '36620':
                            # Arterial line: Keep ICD codes from original input row (already in block_row)
                            pass
                        
                        elif cpt_code in ['36556', '93503', '76937']:
                            # CVP and Ultrasound guidance: Clear all ICD codes
                            for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                if icd_col in block_row:
                                    block_row[icd_col] = ''
                        
                        else:
                            # Any other CPT code: Clear all ICD codes
                            for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                                if icd_col in block_row:
                                    block_row[icd_col] = ''
                        
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
