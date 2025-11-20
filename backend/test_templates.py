#!/usr/bin/env python3
"""
Test script for Instruction Templates feature.
This demonstrates how to use the template management functions.
"""

import sys
import os

# Add current directory to path to import db_utils
sys.path.insert(0, os.path.dirname(__file__))

from db_utils import (
    init_database,
    get_all_templates,
    get_template,
    create_template,
    update_template,
    delete_template
)

def test_templates():
    """Test the template management functions"""
    
    print("=" * 60)
    print("Testing Instruction Templates Feature")
    print("=" * 60)
    
    # 1. Initialize database
    print("\n1. Initializing database...")
    if init_database():
        print("   ✅ Database initialized successfully")
    else:
        print("   ❌ Failed to initialize database")
        return False
    
    # 2. Create a sample template
    print("\n2. Creating a sample template...")
    sample_fields = [
        {
            "name": "Patient Name",
            "description": "Full name of the patient",
            "location": "Top of first page in header section",
            "output_format": "String (First Last)",
            "priority": False
        },
        {
            "name": "Date of Birth",
            "description": "Patient's date of birth",
            "location": "Header section, labeled as DOB",
            "output_format": "MM/DD/YYYY",
            "priority": False
        },
        {
            "name": "Date of Service",
            "description": "Date when the service was provided",
            "location": "Service details section",
            "output_format": "MM/DD/YYYY",
            "priority": True
        },
        {
            "name": "Primary Insurance",
            "description": "Name of primary insurance carrier",
            "location": "Insurance section, labeled as Primary",
            "output_format": "String",
            "priority": False
        }
    ]
    
    template_data = {"fields": sample_fields}
    template_id = create_template(
        name="Test Template - General Surgery",
        description="Sample template for testing the templates feature",
        template_data=template_data
    )
    
    if template_id:
        print(f"   ✅ Template created with ID: {template_id}")
    else:
        print("   ❌ Failed to create template")
        return False
    
    # 3. Get the template by ID
    print("\n3. Retrieving template by ID...")
    template = get_template(template_id=template_id)
    if template:
        print(f"   ✅ Retrieved template: {template['name']}")
        print(f"      Description: {template['description']}")
        print(f"      Fields: {len(template['template_data']['fields'])} fields")
        print(f"      Created: {template['created_at']}")
    else:
        print("   ❌ Failed to retrieve template")
        return False
    
    # 4. Get template by name
    print("\n4. Retrieving template by name...")
    template_by_name = get_template(template_name="Test Template - General Surgery")
    if template_by_name:
        print(f"   ✅ Retrieved template by name: {template_by_name['name']}")
    else:
        print("   ❌ Failed to retrieve template by name")
        return False
    
    # 5. List all templates
    print("\n5. Listing all templates...")
    result = get_all_templates(page=1, page_size=10)
    print(f"   ✅ Found {result['total']} template(s)")
    for tmpl in result['templates']:
        print(f"      - ID: {tmpl['id']}, Name: {tmpl['name']}")
    
    # 6. Update the template
    print("\n6. Updating template...")
    success = update_template(
        template_id=template_id,
        description="Updated description for testing"
    )
    if success:
        updated_template = get_template(template_id=template_id)
        print(f"   ✅ Template updated successfully")
        print(f"      New description: {updated_template['description']}")
    else:
        print("   ❌ Failed to update template")
        return False
    
    # 7. Search for templates
    print("\n7. Searching for templates...")
    search_result = get_all_templates(page=1, page_size=10, search="surgery")
    print(f"   ✅ Search found {search_result['total']} template(s) matching 'surgery'")
    
    # 8. Display field details
    print("\n8. Template field details:")
    template = get_template(template_id=template_id)
    fields = template['template_data']['fields']
    print(f"   Template has {len(fields)} fields:")
    for i, field in enumerate(fields, 1):
        print(f"\n   Field {i}: {field['name']}")
        print(f"      Description: {field['description']}")
        print(f"      Location: {field['location']}")
        print(f"      Format: {field['output_format']}")
        print(f"      Priority: {'YES' if field['priority'] else 'NO'}")
    
    # 9. Clean up - delete the test template
    print("\n9. Cleaning up test template...")
    confirm = input("   Do you want to delete the test template? (y/n): ").strip().lower()
    if confirm == 'y':
        if delete_template(template_id):
            print(f"   ✅ Template {template_id} deleted successfully")
        else:
            print(f"   ❌ Failed to delete template {template_id}")
    else:
        print(f"   ℹ️  Test template kept in database (ID: {template_id})")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_templates()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

