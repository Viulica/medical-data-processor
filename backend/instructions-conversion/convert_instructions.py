#!/usr/bin/env python3
"""
Python script for GAP CSV conversion with specific transformations.
This script transforms the GAP instruction format to the required output format.
"""

import pandas as pd
import sys
from pathlib import Path
import os

# Add parent directory to path to import export_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from export_utils import save_dataframe_dual_format


def get_output_columns():
    """
    Returns the exact column order for the output CSV based on GAP_converted_correct.csv
    """
    return [
        "AI Instructions Part",
        "Patient Last Name", 
        "Patient First Name",
        "Patient Middle Name",
        "Patient DOB",
        "Patient MRN",
        "Case #",
        "Anesthesia Type", 
        "Charge Date",
        "Procedure Description",
        "An Start",
        "An Stop",
        "Patient Sex",
        "Patient Marital Status",
        "Patient EmploymentStatus",
        "Patient E-mail Address",
        "Patient CellPhone",
        "Patient HomePhone", 
        "Patient WorkPhone",
        "Patient Street Address",
        "Patient City",
        "Patient State",
        "Patient ZIP Code",
        "Patient Employer",
        "Admit Date",
        "Discharge Date",
        "Surgeon",
        "Referring",
        "Guarantor Last Name",
        "Guarantor First Name", 
        "Guarantor Middle Name",
        "Guarantor DOB",
        "Guarantor Relation",
        "Guarantor Address",
        "Guarantor City",
        "Guarantor State",
        "Guarantor ZIP",
        "Primary Company Name",
        "Primary Cvg Mem Rel to Sub",
        "Primary Company Address 1",
        "Primary Company City",
        "Primary Company State",
        "Primary Company ZIP",
        "Primary Sub Name",
        "Primary Sub ID",
        "Primary Sub DOB",
        "Primary Sub Gender",
        "Primary Sub Group Num",
        "Primary Sub Auth Num",
        "Secondary Company Name",
        "Secondary Sub ID",
        "Secondary Sub Name",
        "Secondary Sub DOB",
        "Secondary Sub Gender",
        "Secondary Cvg Mem Rel to Sub",
        "Secondary Company Address 1",
        "Secondary Company City",
        "Secondary Company State",
        "Secondary Company ZIP",
        "Secondary Sub Group Num",
        "Secondary Auth Num",
        "Tertiary Company Name",
        "Tertiary Sub ID",
        "Tertiary Sub Name",
        "Tertiary Sub DOB",
        "Tertiary Sub Gender",
        "Tertiary Cvg Mem Rel to Sub",
        "Tertiary Company Address 1",
        "Tertiary Company City",
        "Tertiary Company State",
        "Tertiary Company ZIP",
        "Tertiary Sub Group",
        "Tertiary Sub Auth Num",
        # Additional columns from additions.csv
        "Responsible Provider",
        "MD",
        "CRNA",
        "SRNA",
        "Resident",
        "Concurrent Providers",
        "Physical Status",
        "ICD1",
        "ICD2",
        "ICD3",
        "ICD4"
    ]


def get_column_mapping():
    """
    Returns the mapping from input GAP columns to output columns
    """
    return {
        # First column (unnamed in input) maps to "AI Instructions Part"
        "Unnamed: 0": "AI Instructions Part",  # Pandas names the first unnamed column as "AI Instructions Part"
        
        # Basic mappings - direct column renames
        "DOB": "Patient DOB",
        "MRN": "Patient MRN", 
        "CSN": "Case #",
        "Anes Type": "Anesthesia Type",
        "Date": "Charge Date",
        "Procedure": "Procedure Description",
        "An Start": "An Start",
        "An Stop": "An Stop", 
        "Sex": "Patient Sex",
        "Marital Status": "Patient Marital Status",
        "Pt. E-mail Address": "Patient E-mail Address",
        "Cell Phone": "Patient CellPhone",
        "Home Phone": "Patient HomePhone",
        "Street Address": "Patient Street Address",
        "City": "Patient City",
        "ST": "Patient State",
        "ZIP Code": "Patient ZIP Code",
        "Patient Employer": "Patient Employer",
        "Guarantor Name": "",  # Will be split into separate fields
        "Guarantor Relation": "Guarantor Relation",
        "Guarantor Address": "Guarantor Address",
        "Guarantor City": "Guarantor City",
        "Guarantor State": "Guarantor State",
        "Guarantor ZIP": "Guarantor ZIP",
        "Primary Insurance Name": "Primary Company Name",
        "Primary Cvg Mem Rel to Sub": "Primary Cvg Mem Rel to Sub", 
        "Primary Insurance Address": "Primary Company Address 1",
        "Primary CVG Sub Name": "Primary Sub Name",
        "Primary Subsc ID": "Primary Sub ID",
        "Primary CVG Group Num": "Primary Sub Group Num",
        "Secondary Insurance Name": "Secondary Company Name",
        "Secondary Subsc ID": "Secondary Sub ID",
        "Secondary CVG Sub Name": "Secondary Sub Name", 
        "Secondary Cvg Mem Rel to Sub": "Secondary Cvg Mem Rel to Sub",
        "Secondary Insurance Address": "Secondary Company Address 1",
        "Secondary CVG Group Num": "Secondary Sub Group Num",
        
        # Add missing secondary insurance fields
        "Secondary FinClass ID": "",  # Not in output 
        "Secondary HealthPlan ID": "",  # Not in output
    }


def split_patient_name(full_name):
    """
    Split 'Patient Name' into Last, First, Middle components
    Expected format: "LAST, FIRST MIDDLE" or "LAST, FIRST"
    """
    if pd.isna(full_name) or str(full_name).strip() == "":
        return "", "", ""
    
    name_str = str(full_name).strip()
    
    # Split by comma first
    if "," in name_str:
        parts = name_str.split(",", 1)
        last_name = parts[0].strip()
        first_part = parts[1].strip() if len(parts) > 1 else ""
        
        # Split first part by space to get first and middle
        if " " in first_part:
            name_parts = first_part.split()
            first_name = name_parts[0]
            middle_name = " ".join(name_parts[1:])
        else:
            first_name = first_part
            middle_name = ""
    else:
        # No comma, assume it's just first name or handle as best we can
        parts = name_str.split()
        if len(parts) >= 2:
            last_name = parts[-1]  # Last word as last name
            first_name = parts[0]  # First word as first name
            middle_name = " ".join(parts[1:-1]) if len(parts) > 2 else ""
        elif len(parts) == 1:
            last_name = parts[0]
            first_name = ""
            middle_name = ""
        else:
            last_name = ""
            first_name = ""
            middle_name = ""
    
    return last_name, first_name, middle_name


def split_guarantor_name(full_name):
    """
    Split 'Guarantor Name' into Last, First, Middle components
    Expected format: "LAST, FIRST MIDDLE" or "LAST, FIRST"
    """
    if pd.isna(full_name) or str(full_name).strip() == "":
        return "", "", ""
    
    name_str = str(full_name).strip()
    
    # Split by comma first
    if "," in name_str:
        parts = name_str.split(",", 1)
        last_name = parts[0].strip()
        first_part = parts[1].strip() if len(parts) > 1 else ""
        
        # Split first part by space to get first and middle
        if " " in first_part:
            name_parts = first_part.split()
            first_name = name_parts[0]
            middle_name = " ".join(name_parts[1:])
        else:
            first_name = first_part
            middle_name = ""
    else:
        # No comma, assume it's just first name or handle as best we can
        parts = name_str.split()
        if len(parts) >= 2:
            last_name = parts[-1]  # Last word as last name
            first_name = parts[0]  # First word as first name
            middle_name = " ".join(parts[1:-1]) if len(parts) > 2 else ""
        elif len(parts) == 1:
            last_name = parts[0]
            first_name = ""
            middle_name = ""
        else:
            last_name = ""
            first_name = ""
            middle_name = ""
    
    return last_name, first_name, middle_name


def add_manual_descriptions(new_row, row_idx):
    """
    Add manual descriptions that were in the expected output but not in the input
    """
    if row_idx == 0:  # First data row (row 2 in CSV) - ALL TEXT GOES HERE
        new_row["Patient Last Name"] = "patient last name (surname), there might be multiple last names, put them all seperated with one space"
        new_row["Patient First Name"] = "patient first name"
        new_row["Patient Middle Name"] = "patient middle name (just one capital letter like A)"
        new_row["Guarantor Last Name"] = "guarantor last name (surname)"
        new_row["Guarantor First Name"] = "guarantor first name"
        new_row["Guarantor Middle Name"] = "guarantor middle name (just one capital letter like A)"
        new_row["Guarantor DOB"] = "guarantor date of birth in MM/DD/YYYY format"
        new_row["Anesthesia Type"] = "Output enum is these values: GENERAL, MAC, REGIONAL, TIVA - THESE ARE THE ONLY ALLOWED OPTIONS"
        new_row["Procedure Description"] = "extract the procedure description here"
        new_row["An Start"] = "this is the start of anesthesia, it has to be in this format DD/MM/YYYY HH:MM, for example 9/17/2024 15:48"
        new_row["An Stop"] = "this is the stop of anesthesia, it has to be in this format DD/MM/YYYY HH:MM, for example 9/17/2024 15:48"
        new_row["Patient Marital Status"] = "Output enum is these values: Married, Widowed, Single, Divorced, Domestic Partner, Unknown, Other, Legally separated - THESE ARE THE ONLY ALLOWED OPTIONS if its not listed on the pdf then put the unknown option obviously but whenever you write something"
        new_row["Patient EmploymentStatus"] = "Output enum is these values: Employed, Full-time student, Part-time student, None - THESE ARE THE ONLY ALLOWED OPTIONS"
        new_row["Patient CellPhone"] = "output is JUST numbers, nothing else, strip all parentheses and dashes, example output: 546222435 (just numbers)"
        new_row["Patient HomePhone"] = "output is JUST numbers, nothing else, strip all parentheses and dashes, example output: 546222435 (just numbers)"
        new_row["Patient WorkPhone"] = "output is JUST numbers, nothing else, strip all parentheses and dashes, example output: 546222435 (just numbers)"
        new_row["Admit Date"] = "admission date in MM/DD/YYYY format"
        new_row["Discharge Date"] = "discharge date in MM/DD/YYYY format"
        new_row["Surgeon"] = "name of the surgeon who performed the procedure"
        new_row["Referring"] = "name of the referring physician (copy the surgeon name here)"
        new_row["Guarantor State"] = "State of the guarantor, full name all caps"
        # Primary insurance field descriptions (from input file)
        new_row["Primary Company Name"] = "Name of primary insurance"
        new_row["Primary Cvg Mem Rel to Sub"] = "relation of the primary insurance holder to the patient (Self, Child, Spouse or Other) -those are the only 4 options"
        new_row["Primary Company Address 1"] = "address of the primary insurance"
        new_row["Primary Company City"] = "city of the primary insurance"
        new_row["Primary Company State"] = "state of the primary insurance"
        new_row["Primary Company ZIP"] = "zip code of the primary insurance"
        new_row["Primary Sub Name"] = "this is the name of the person who has the policy"
        new_row["Primary Sub ID"] = "this is the policy number of the primary insurance"
        new_row["Primary Sub DOB"] = "date of birth of the primary insurance subscriber in MM/DD/YYYY format"
        new_row["Primary Sub Gender"] = "gender of the primary insurance subscriber (M/F)"
        new_row["Primary Sub Group Num"] = "group number of primary insurance"
        new_row["Primary Sub Auth Num"] = "always leave empty string!!"
        # Secondary insurance field descriptions (copied from primary)
        new_row["Secondary Company Name"] = "Name of secondary insurance"
        new_row["Secondary Sub ID"] = "this is the policy number of the secondary insurance"
        new_row["Secondary Sub Name"] = "this is the name of the person who has the secondary policy"
        new_row["Secondary Sub DOB"] = "date of birth of the secondary insurance subscriber in MM/DD/YYYY format"
        new_row["Secondary Sub Gender"] = "gender of the secondary insurance subscriber (M/F)"
        new_row["Secondary Cvg Mem Rel to Sub"] = "relation of the secondary insurance holder to the patient (Self, Child, Spouse or Other) -those are the only 4 options"
        new_row["Secondary Company Address 1"] = "address of the secondary insurance"
        new_row["Secondary Company City"] = "city of the secondary insurance"
        new_row["Secondary Company State"] = "state of the secondary insurance"
        new_row["Secondary Company ZIP"] = "zip code of the secondary insurance"
        new_row["Secondary Sub Group Num"] = "group number of secondary insurance"
        new_row["Secondary Auth Num"] = "always leave empty string!!"
        # Tertiary insurance field descriptions (copied from primary)
        new_row["Tertiary Company Name"] = "Name of tertiary insurance"
        new_row["Tertiary Sub ID"] = "this is the policy number of the tertiary insurance"
        new_row["Tertiary Sub Name"] = "this is the name of the person who has the tertiary policy"
        new_row["Tertiary Sub DOB"] = "date of birth of the tertiary insurance subscriber in MM/DD/YYYY format"
        new_row["Tertiary Sub Gender"] = "gender of the tertiary insurance subscriber (M/F)"
        new_row["Tertiary Cvg Mem Rel to Sub"] = "relation of the tertiary insurance holder to the patient (Self, Child, Spouse or Other) -those are the only 4 options"
        new_row["Tertiary Company Address 1"] = "address of the tertiary insurance"
        new_row["Tertiary Company City"] = "city of the tertiary insurance"
        new_row["Tertiary Company State"] = "state of the tertiary insurance"
        new_row["Tertiary Company ZIP"] = "zip code of the tertiary insurance"
        new_row["Tertiary Sub Group"] = "group number of tertiary insurance"
        new_row["Tertiary Sub Auth Num"] = "always leave empty string!!"
        # Clear Patient WorkPhone and other fields for row 2
        new_row["Patient WorkPhone"] = ""
        
    elif row_idx == 1:  # Second data row (row 3 in CSV) - CLEAR ALL TEXT
        new_row["Patient Last Name"] = ""
        new_row["Patient First Name"] = ""
        new_row["Patient Middle Name"] = ""
        new_row["Guarantor Last Name"] = ""
        new_row["Guarantor First Name"] = ""
        new_row["Guarantor Middle Name"] = ""
        new_row["Guarantor DOB"] = ""
        new_row["Anesthesia Type"] = ""
        new_row["Procedure Description"] = ""
        new_row["An Start"] = ""
        new_row["An Stop"] = ""
        new_row["Patient Marital Status"] = ""
        new_row["Patient EmploymentStatus"] = ""
        new_row["Patient CellPhone"] = ""
        new_row["Patient HomePhone"] = ""
        new_row["Patient WorkPhone"] = ""
        new_row["Admit Date"] = ""
        new_row["Discharge Date"] = ""
        new_row["Surgeon"] = ""
        new_row["Referring"] = ""
        new_row["Guarantor State"] = ""
        new_row["Primary Company City"] = ""
        new_row["Primary Company State"] = ""
        new_row["Primary Company ZIP"] = ""
        new_row["Primary Sub DOB"] = ""
        new_row["Primary Sub Gender"] = ""
        new_row["Primary Sub Auth Num"] = ""
        new_row["Secondary Company Name"] = ""
        new_row["Secondary Sub ID"] = ""
        new_row["Secondary Sub Name"] = ""
        new_row["Secondary Sub DOB"] = ""
        new_row["Secondary Sub Gender"] = ""
        new_row["Secondary Cvg Mem Rel to Sub"] = ""
        new_row["Secondary Company Address 1"] = ""
        new_row["Secondary Company City"] = ""
        new_row["Secondary Company State"] = ""
        new_row["Secondary Company ZIP"] = ""
        new_row["Secondary Sub Group Num"] = ""
        new_row["Secondary Auth Num"] = ""
        new_row["Tertiary Company Name"] = ""
        new_row["Tertiary Sub ID"] = ""
        new_row["Tertiary Sub Name"] = ""
        new_row["Tertiary Sub DOB"] = ""
        new_row["Tertiary Sub Gender"] = ""
        new_row["Tertiary Cvg Mem Rel to Sub"] = ""
        new_row["Tertiary Company Address 1"] = ""
        new_row["Tertiary Company City"] = ""
        new_row["Tertiary Company State"] = ""
        new_row["Tertiary Company ZIP"] = ""
        new_row["Tertiary Sub Group"] = ""
        new_row["Tertiary Sub Auth Num"] = ""
        new_row["Patient WorkPhone"] = ""
        
    elif row_idx == 2:  # Third data row (row 4 in CSV) - CLEAR ALL TEXT
        new_row["Patient Last Name"] = ""
        new_row["Patient First Name"] = ""
        new_row["Patient Middle Name"] = ""
        new_row["Guarantor Last Name"] = ""
        new_row["Guarantor First Name"] = ""
        new_row["Guarantor Middle Name"] = ""
        new_row["Guarantor DOB"] = ""
        new_row["Anesthesia Type"] = ""
        new_row["Procedure Description"] = ""
        new_row["An Start"] = ""
        new_row["An Stop"] = ""
        new_row["Patient Marital Status"] = ""
        new_row["Patient EmploymentStatus"] = ""
        new_row["Patient CellPhone"] = ""
        new_row["Patient HomePhone"] = ""
        new_row["Patient WorkPhone"] = ""
        new_row["Admit Date"] = ""
        new_row["Discharge Date"] = ""
        new_row["Surgeon"] = ""
        new_row["Referring"] = ""
        new_row["Guarantor State"] = ""
        new_row["Primary Company City"] = ""
        new_row["Primary Company State"] = ""
        new_row["Primary Company ZIP"] = ""
        new_row["Primary Sub DOB"] = ""
        new_row["Primary Sub Gender"] = ""
        new_row["Primary Sub Auth Num"] = ""
        new_row["Secondary Company Name"] = ""
        new_row["Secondary Sub ID"] = ""
        new_row["Secondary Sub Name"] = ""
        new_row["Secondary Sub DOB"] = ""
        new_row["Secondary Sub Gender"] = ""
        new_row["Secondary Cvg Mem Rel to Sub"] = ""
        new_row["Secondary Company Address 1"] = ""
        new_row["Secondary Company City"] = ""
        new_row["Secondary Company State"] = ""
        new_row["Secondary Company ZIP"] = ""
        new_row["Secondary Sub Group Num"] = ""
        new_row["Secondary Auth Num"] = ""
        new_row["Tertiary Company Name"] = ""
        new_row["Tertiary Sub ID"] = ""
        new_row["Tertiary Sub Name"] = ""
        new_row["Tertiary Sub DOB"] = ""
        new_row["Tertiary Sub Gender"] = ""
        new_row["Tertiary Cvg Mem Rel to Sub"] = ""
        new_row["Tertiary Company Address 1"] = ""
        new_row["Tertiary Company City"] = ""
        new_row["Tertiary Company State"] = ""
        new_row["Tertiary Company ZIP"] = ""
        new_row["Tertiary Sub Group"] = ""
        new_row["Tertiary Sub Auth Num"] = ""
        new_row["Patient WorkPhone"] = ""


def add_additions_data(new_row, row_idx, additions_df):
    """
    Add data from additions.csv to the current row
    """
    # List of additional columns from additions.csv
    additional_columns = [
        "Responsible Provider", "MD", "CRNA", "SRNA", "Resident", 
        "Concurrent Providers", "Physical Status", "ICD1", "ICD2", "ICD3", "ICD4"
    ]
    
    if additions_df is not None and row_idx < len(additions_df):
        # Copy data from additions.csv for this row
        for col in additional_columns:
            if col in additions_df.columns:
                value = additions_df.iloc[row_idx][col]
                new_row[col] = value if not pd.isna(value) else ""
            else:
                new_row[col] = ""
    else:
        # If no additions file or row doesn't exist, set empty values
        for col in additional_columns:
            new_row[col] = ""


def convert_data(input_file, output_file=None):
    """
    Main function to convert GAP CSV with specific transformations
    """
    try:
        # Read the main CSV file with dtype=str to preserve leading zeros in codes
        # Try multiple encodings to handle different file formats
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        last_error = None
        
        for encoding in encodings_to_try:
            try:
                print(f"Attempting to read file with encoding: {encoding}")
                df = pd.read_csv(input_file, dtype=str, encoding=encoding)
                print(f"Successfully read file with encoding: {encoding}")
                break
            except UnicodeDecodeError as e:
                print(f"Failed to read with {encoding}: {e}")
                last_error = e
                continue
        
        if df is None:
            print(f"Error: Could not read file with any standard encoding. Last error: {last_error}")
            return False
        
        if len(df) < 1:
            print("Error: CSV file must have at least 1 row (headers)")
            return False
        
        # Read the additions.csv file
        additions_file = Path(input_file).parent / "additions.csv"
        additions_df = None
        if additions_file.exists():
            print(f"Found additions file: {additions_file}")
            # Try multiple encodings for additions file too
            for encoding in encodings_to_try:
                try:
                    additions_df = pd.read_csv(additions_file, encoding=encoding)
                    print(f"Successfully read additions file with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            if additions_df is None:
                print(f"Warning: Could not read additions.csv with any encoding")
        else:
            print(f"Warning: additions.csv not found at {additions_file}")
            print("Will create empty additional columns")
        
        # Get the column mapping and output structure
        column_mapping = get_column_mapping()
        output_columns = get_output_columns()
        
        # Create new dataframe with the exact output structure
        result_data = []
        
        # Process each data row
        for row_idx in range(len(df)):
            new_row = {}
            
            # Initialize all output columns with empty values
            for col in output_columns:
                new_row[col] = ""
            
            # Handle Patient Name splitting (only for non-description rows)
            if "Patient Name" in df.columns and row_idx not in [0, 1, 2]:
                patient_name = df.iloc[row_idx]["Patient Name"]
                last_name, first_name, middle_name = split_patient_name(patient_name)
                new_row["Patient Last Name"] = last_name
                new_row["Patient First Name"] = first_name  
                new_row["Patient Middle Name"] = middle_name
            
            # Handle Guarantor Name splitting (only for non-description rows)
            if "Guarantor Name" in df.columns and row_idx not in [0, 1, 2]:
                guarantor_name = df.iloc[row_idx]["Guarantor Name"]
                last_name, first_name, middle_name = split_guarantor_name(guarantor_name)
                new_row["Guarantor Last Name"] = last_name
                new_row["Guarantor First Name"] = first_name  
                new_row["Guarantor Middle Name"] = middle_name
            
            # Map all columns using the mapping
            for old_col in df.columns:
                if old_col == "Patient Name" or old_col == "Guarantor Name":
                    continue  # Already handled above
                    
                clean_old_col = old_col.strip('"')
                
                if clean_old_col in column_mapping:
                    new_header = column_mapping[clean_old_col]
                    if new_header in new_row:  # Only if it's in our output columns
                        value = df.iloc[row_idx][old_col]
                        new_row[new_header] = value if not pd.isna(value) else ""
            
            # Add manual descriptions for specific rows
            add_manual_descriptions(new_row, row_idx)
            
            # Add data from additions.csv
            add_additions_data(new_row, row_idx, additions_df)
            
            result_data.append(new_row)
        
        # Create result dataframe with exact column order
        result_df = pd.DataFrame(result_data, columns=output_columns)
        
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
