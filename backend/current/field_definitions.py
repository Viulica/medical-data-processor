# Field definitions for medical document extraction
# Supports loading from different Excel files for different hospitals

import pandas as pd

# System fields that are always added
SYSTEM_FIELDS = [
    {
        'name': 'source_file',
        'description': 'Source file name (automatically added by system)',
        'location': 'System generated',
        'output_format': 'String'
    },
    {
        'name': 'page_number', 
        'description': 'Page number (automatically added by system)',
        'location': 'System generated',
        'output_format': 'Integer'
    }
]

def load_field_definitions_from_excel(excel_file_path):
    """Load field definitions from Excel file with transposed structure."""
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path, header=0)
        
        # Get field names from column headers (first row)
        field_names = df.columns.tolist()
        
        # Read the description row (row index 0 after header)
        description_row = df.iloc[0] if len(df) > 0 else None
        
        # Read the location row (row index 1 after header) 
        location_row = df.iloc[1] if len(df) > 1 else None
        
        # Read the output format row (row index 2 after header)
        output_format_row = df.iloc[2] if len(df) > 2 else None
        
        field_definitions = []
        
        for i, field_name in enumerate(field_names):
            # Skip empty or unnamed columns
            if pd.isna(field_name) or str(field_name).strip() == '' or str(field_name).startswith('Unnamed'):
                continue
                
            field_def = {
                'name': str(field_name).strip(),
                'description': str(description_row.iloc[i]).strip() if description_row is not None and not pd.isna(description_row.iloc[i]) else '',
                'location': str(location_row.iloc[i]).strip() if location_row is not None and not pd.isna(location_row.iloc[i]) else '',
                'output_format': str(output_format_row.iloc[i]).strip() if output_format_row is not None and not pd.isna(output_format_row.iloc[i]) else ''
            }
            
            field_definitions.append(field_def)
        
        return field_definitions
        
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return []

def get_field_definitions(excel_file_path):
    """Get all field definitions (system + Excel) for a specific hospital."""
    # Load from Excel
    excel_fields = load_field_definitions_from_excel(excel_file_path)
    
    # Combine system fields with Excel fields
    all_fields = SYSTEM_FIELDS + excel_fields
    
    return all_fields

def get_fieldnames(excel_file_path):
    """Return list of field names for CSV headers."""
    field_definitions = get_field_definitions(excel_file_path)
    return [field['name'] for field in field_definitions]

def generate_extraction_prompt(excel_file_path):
    """Generate the extraction prompt from field definitions."""
    field_definitions = get_field_definitions(excel_file_path)
    
    prompt_header = """You are an expert in extracting structured data from medical documents. For each patient record provided in the following text, extract the information specified below.
If a field is not present or cannot be determined from the provided text, output null for that field.

BE VERY CAREFUL to not confuse zero and the letter O. If the sign is completely round then it is an O (letter O), if it is a little bit oval then it is an O.
Extraction Instructions per Patient Record:
"""
    
    field_instructions = []
    for field in field_definitions:
        # Skip the metadata fields that are automatically added
        if field['name'] not in ['source_file', 'page_number']:

            # Create instruction combining description, location, and output format
            instruction_parts = []
            
            if field.get('description'):
                instruction_parts.append(field['description'])
            
            if field.get('location'):
                instruction_parts.append(f"Location: {field['location']}")
            
            if field.get('output_format'):
                instruction_parts.append(f"Format: {field['output_format']}")
            
            instruction = ' | '.join(instruction_parts) if instruction_parts else 'Extract if available'
            field_instructions.append(f"{field['name']}: {instruction}")


    field_specific_instructions = """


     1- For address zip and city fields, do not extract the comma sign, just the text.

    2. When it comes to guarantor relation, the names DO NOT have to match exatctly to be self relation.

    3. Also when it comes to guarantor relation , if the relation is listed as "parent" in the pdf, your output is "Child" because the only three valid options for this field are "Self", "Child", and "Other".

    4. If the date of birth of patient and guarantor is the same AND the names are roughly the same, then output "Self" for guarantor relation.

    5. IMPORTANT: if you see any random ID looking numbers in a different color and font then the original pdf (they look copy pasted) then ignore them (do not put them in any of the extracted fields!!!)

    6. ALSO IMPORTANT: never DROP leading zeros from any number. Always write all the strings and numbers as they are.

    7. ALSO IMPORTANT: if for a certain field I said you should put an empty string, always put an empty string make sure the final output is an empty string.


    8. OTHER RULES (make sure to double check you are following these rules):
    9. FOR THE CHARGE DATE FIELD: if there is a date of service, put that, if there is no date of service then put the admission date, if there is no clear others, then take whatever makes most sense.

    """

    prompt_footer = """


    ALSO: be very careful to not confuse the number 1 and the letter l , double check you did not confuse them.

    FINAL INSTRUCTION: make sure to fill out all the fields that have data contained in the pdf. double check you did not leave empty something that should be filled out, or vice versa that you hallunicated. something.

    Your entire response must be a single JSON object representing the one patient record found on this page. Do not include any other text or commentary."""
    
    # save the prompt to a file
    with open('extraction_prompt.txt', 'w') as f:
        f.write(prompt_header + '\n'.join(field_instructions) + prompt_footer)

    return prompt_header + '\n'.join(field_instructions) + prompt_footer 