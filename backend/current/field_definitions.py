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
        
        # Read the priority row (row index 3 after header)
        priority_row = df.iloc[3] if len(df) > 3 else None
        
        field_definitions = []
        
        for i, field_name in enumerate(field_names):
            # Skip empty or unnamed columns
            if pd.isna(field_name) or str(field_name).strip() == '' or str(field_name).startswith('Unnamed'):
                continue
                
            # Check if priority is set to YES (case insensitive)
            priority_value = str(priority_row.iloc[i]).strip().upper() if priority_row is not None and not pd.isna(priority_row.iloc[i]) else ''
            is_priority = priority_value == 'YES'
                
            field_def = {
                'name': str(field_name).strip(),
                'description': str(description_row.iloc[i]).strip() if description_row is not None and not pd.isna(description_row.iloc[i]) else '',
                'location': str(location_row.iloc[i]).strip() if location_row is not None and not pd.isna(location_row.iloc[i]) else '',
                'output_format': str(output_format_row.iloc[i]).strip() if output_format_row is not None and not pd.isna(output_format_row.iloc[i]) else '',
                'priority': is_priority
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

def get_priority_fields(excel_file_path):
    """Return list of field definitions that are marked as priority."""
    field_definitions = get_field_definitions(excel_file_path)
    return [field for field in field_definitions if field.get('priority', False) and field['name'] not in ['source_file', 'page_number']]

def get_normal_fields(excel_file_path):
    """Return list of field definitions that are NOT marked as priority."""
    field_definitions = get_field_definitions(excel_file_path)
    return [field for field in field_definitions if not field.get('priority', False) and field['name'] not in ['source_file', 'page_number']]

def generate_extraction_prompt(excel_file_path, fields_to_include=None):
    """Generate the extraction prompt from field definitions.
    
    Args:
        excel_file_path: Path to Excel file with field definitions
        fields_to_include: Optional list of field definitions to include. If None, includes all non-priority fields.
    """
    if fields_to_include is None:
        # Get only normal (non-priority) fields
        field_definitions = get_normal_fields(excel_file_path)
    else:
        field_definitions = fields_to_include
    
    prompt_header = """You are an expert in extracting structured data from medical documents. For each patient record provided in the following text, extract the information specified below.
If a field is not present or cannot be determined from the provided text, output null for that field.

BE VERY CAREFUL to not confuse zero and the letter O. If the sign is completely round then it is an O (letter O), if it is a little bit oval then it is an O.
Extraction Instructions per Patient Record:
"""
    
    field_instructions = []
    for field in field_definitions:
        # Build a clear, structured instruction for each field
        field_instruction = f"""
=== {field['name']} ===
"""
        
        if field.get('description'):
            field_instruction += f"Description: {field['description']}\n"
        
        if field.get('location'):
            field_instruction += f"Where to find: {field['location']}\n"
        
        if field.get('output_format'):
            field_instruction += f"Output format: {field['output_format']}\n"
        
        field_instruction += f"Extract: {field['name']}\n"
        field_instructions.append(field_instruction)


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

def generate_priority_field_prompt(field_definition):
    """Generate a focused extraction prompt for a single priority field.
    
    Args:
        field_definition: Dictionary containing field definition (name, description, location, output_format)
    
    Returns:
        String containing the extraction prompt focused on this single field
    """
    prompt_header = f"""You are an expert in extracting structured data from medical documents. 

IMPORTANT: Focus ONLY on extracting the following specific field. Pay close attention to the details provided.

BE VERY CAREFUL to not confuse zero and the letter O. If the sign is completely round then it is an O (letter O), if it is a little bit oval then it is an O.

=== {field_definition['name']} ===
"""
    
    if field_definition.get('description'):
        prompt_header += f"Description: {field_definition['description']}\n"
    
    if field_definition.get('location'):
        prompt_header += f"Where to find: {field_definition['location']}\n"
    
    if field_definition.get('output_format'):
        prompt_header += f"Output format: {field_definition['output_format']}\n"
    
    prompt_footer = """

IMPORTANT RULES:
1. NEVER DROP leading zeros from any number. Always write all the strings and numbers as they are.
2. If the field is not present or cannot be determined from the provided text, output null for that field.
3. Be very careful to not confuse the number 1 and the letter l, double check you did not confuse them.
4. If you see any random ID looking numbers in a different color and font than the original pdf (they look copy pasted), ignore them.

Your entire response must be a single JSON object with ONE key: "{field_definition['name']}" and its extracted value. Do not include any other text or commentary.

Example response format:
{{"field_name": "extracted_value"}}
"""
    
    return prompt_header + prompt_footer.replace('{field_definition[\'name\']}', field_definition['name']) 