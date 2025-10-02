#!/usr/bin/env python3
"""
Script to generate medical modifiers based on mednet codes and provider information.
This script processes CSV data and generates modifiers according to specific business rules.
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
        
        # Ensure M1, M2, M3 columns exist
        if 'M1' not in df.columns:
            df['M1'] = ''
        if 'M2' not in df.columns:
            df['M2'] = ''
        if 'M3' not in df.columns:
            df['M3'] = ''
        
        # Check for Anesthesia Type and Physical Status columns
        has_anesthesia_type = 'Anesthesia Type' in df.columns
        has_physical_status = 'Physical Status' in df.columns
        
        # Process each row
        result_rows = []
        successful_matches = 0
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            new_row = row.copy()
            
            # Reset M1, M2, M3 for this row
            new_row['M1'] = ''
            new_row['M2'] = ''
            new_row['M3'] = ''
            
            primary_mednet_code = str(row.get('Primary Mednet Code', '')).strip()
            
            # Variables to track modifiers
            m1_modifier = ''
            qs_modifier = ''
            p_modifier = ''
            medicare_modifiers = False
            
            # Determine M1 modifier (AA/QK/QZ) based on mednet code
            if primary_mednet_code and primary_mednet_code != '' and primary_mednet_code.lower() != 'nan':
                # Look up the code in modifiers definition
                if primary_mednet_code in modifiers_dict:
                    # Code found in definition - increment successful matches
                    successful_matches += 1
                    
                    # Get the modifiers settings
                    medicare_modifiers, medical_direction = modifiers_dict[primary_mednet_code]
                    
                    # Check if MD and CRNA have values
                    has_md = False
                    has_crna = False
                    
                    if has_md_column:
                        md_value = row.get('MD', '')
                        if not pd.isna(md_value) and str(md_value).strip() != '':
                            has_md = True
                    
                    if has_crna_column:
                        crna_value = row.get('CRNA', '')
                        if not pd.isna(crna_value) and str(crna_value).strip() != '':
                            has_crna = True
                    
                    # Determine M1 modifier (AA/QK/QZ)
                    m1_modifier = determine_modifier(has_md, has_crna, medicare_modifiers, medical_direction)
            
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
            
            # Apply hierarchy: M1 (AA/QK/QZ) > M2 (QS) > M3 (P)
            # Place modifiers in first available slots
            if m1_modifier:
                # M1 has AA/QK/QZ
                new_row['M1'] = m1_modifier
                if qs_modifier:
                    new_row['M2'] = qs_modifier
                    if p_modifier:
                        new_row['M3'] = p_modifier
                else:
                    # No QS, so P goes in M2
                    if p_modifier:
                        new_row['M2'] = p_modifier
            else:
                # No M1 modifier
                if qs_modifier:
                    # QS goes in M2
                    new_row['M2'] = qs_modifier
                    if p_modifier:
                        new_row['M3'] = p_modifier
                else:
                    # No M1, no QS, so P goes in M1
                    if p_modifier:
                        new_row['M1'] = p_modifier
            
            result_rows.append(new_row)
        
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
