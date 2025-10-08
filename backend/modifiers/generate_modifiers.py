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

- Peripheral Block Row Generation:
  * When "Peripheral block given" = "PERIPHERAL_BLOCK"
  * AND "Anesthesia Type" = "General" (case insensitive)
  * A duplicate row is created with:
    - ASA Code and Procedure Code set to 64447
    - Provider fields copied from Peripheral block MD/Resident/CRNA
    - ICD codes cleared except ICD1 = "G89.18"
    - Concurrent Providers cleared
    - Modifiers set based on "Peripheral block laterality":
      * Left: LT, 59
      * Right: RT, 59
      * Bilateral: 50, 59
"""

import pandas as pd
import sys
from pathlib import Path


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
        
        # Check for Peripheral block columns
        has_peripheral_block_given = 'Peripheral block given' in df.columns
        has_peripheral_block_md = 'Peripheral block MD' in df.columns
        has_peripheral_block_resident = 'Peripheral block Resident' in df.columns
        has_peripheral_block_crna = 'Peripheral block CRNA' in df.columns
        has_peripheral_block_laterality = 'Peripheral block laterality' in df.columns
        
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
            
            # Determine PT modifier based on Polyps found AND colonoscopy_is_screening = TRUE
            # PT modifier does NOT require medicare modifiers
            if has_polyps_found_column and has_colonoscopy_screening_column:
                polyps_value = str(row.get('Polyps found', '')).strip().upper()
                colonoscopy_screening = str(row.get('colonoscopy_is_screening', '')).strip().upper()
                # PT is added only when polyps are found AND it's a screening colonoscopy
                if polyps_value == 'FOUND' and colonoscopy_screening == 'TRUE':
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
            
            # Check if we need to create a peripheral block row
            # Conditions: 
            # 1. "Peripheral block given" = "PERIPHERAL_BLOCK"
            # 2. "Anesthesia Type" = "General" (case insensitive)
            if has_peripheral_block_given and has_anesthesia_type:
                
                peripheral_block_given = str(row.get('Peripheral block given', '')).strip()
                anesthesia_type_val = str(row.get('Anesthesia Type', '')).strip().upper()
                
                if (peripheral_block_given == 'PERIPHERAL_BLOCK' and 
                    anesthesia_type_val == 'GENERAL'):
                    
                    # Create a copy of the original row (not the modified new_row)
                    peripheral_row = row.copy()
                    
                    # Set ASA Code and Procedure Code to 64447
                    peripheral_row['ASA Code'] = '64447'
                    peripheral_row['Procedure Code'] = '64447'
                    
                    # Copy peripheral block provider values
                    if has_peripheral_block_md:
                        peripheral_row['MD'] = row.get('Peripheral block MD', '')
                    if has_peripheral_block_resident:
                        peripheral_row['Resident'] = row.get('Peripheral block Resident', '')
                    if has_peripheral_block_crna:
                        peripheral_row['CRNA'] = row.get('Peripheral block CRNA', '')
                    
                    # Clear ICD codes
                    for icd_col in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                        if icd_col in peripheral_row:
                            peripheral_row[icd_col] = ''
                    
                    # Set ICD1 to G89.18
                    peripheral_row['ICD1'] = 'G89.18'
                    
                    # Clear Concurrent Providers
                    if 'Concurrent Providers' in peripheral_row:
                        peripheral_row['Concurrent Providers'] = ''
                    
                    # Clear all modifier columns
                    peripheral_row['M1'] = ''
                    peripheral_row['M2'] = ''
                    peripheral_row['M3'] = ''
                    peripheral_row['M4'] = ''
                    
                    # Set modifiers based on laterality (case-insensitive)
                    if has_peripheral_block_laterality:
                        laterality = str(row.get('Peripheral block laterality', '')).strip().upper()
                        
                        if laterality == 'LEFT':
                            peripheral_row['M1'] = 'LT'
                            peripheral_row['M2'] = '59'
                        elif laterality == 'RIGHT':
                            peripheral_row['M1'] = 'RT'
                            peripheral_row['M2'] = '59'
                        elif laterality == 'BILATERAL':
                            peripheral_row['M1'] = '50'
                            peripheral_row['M2'] = '59'
                    
                    # Add the peripheral block row to results
                    result_rows.append(peripheral_row)
        
        # Create result dataframe
        result_df = pd.DataFrame(result_rows)
        
        # Save to output file
        if output_file is None:
            output_file = input_file.replace('.csv', '_with_modifiers.csv')
        
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
