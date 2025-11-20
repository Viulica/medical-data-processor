# Instruction Templates Feature

This feature allows you to save Excel instruction files as templates in the PostgreSQL database, making it easy to reuse extraction configurations without uploading the same Excel file multiple times.

## Overview

The Instruction Templates feature provides:

- **Upload and Save**: Upload an Excel file and save it as a named template
- **List Templates**: View all saved templates with pagination and search
- **View Template Details**: Get full details of a specific template including field definitions
- **Edit Templates**: Update template name, description, or field definitions
- **Delete Templates**: Remove templates you no longer need
- **Export Templates**: Download a template back to Excel format
- **Use in Extraction**: Apply saved templates directly in PDF extraction workflows

## Database Schema

The templates are stored in a PostgreSQL table:

```sql
CREATE TABLE instruction_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    template_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Endpoints

### 1. List All Templates

Get a paginated list of all templates with optional search.

**Endpoint:** `GET /api/templates`

**Query Parameters:**

- `page` (int, optional, default=1): Page number
- `page_size` (int, optional, default=50): Number of results per page
- `search` (string, optional): Search term for name or description

**Response:**

```json
{
  "templates": [
    {
      "id": 1,
      "name": "Hospital A - General Surgery",
      "description": "Standard extraction template for Hospital A surgical cases",
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 10,
  "page": 1,
  "page_size": 50,
  "total_pages": 1
}
```

**Example Usage:**

```bash
curl http://localhost:8000/api/templates?page=1&page_size=20&search=surgery
```

---

### 2. Get Template by ID

Get full details of a specific template including field definitions.

**Endpoint:** `GET /api/templates/{template_id}`

**Response:**

```json
{
  "id": 1,
  "name": "Hospital A - General Surgery",
  "description": "Standard extraction template for Hospital A surgical cases",
  "template_data": {
    "fields": [
      {
        "name": "Patient Name",
        "description": "Full name of patient",
        "location": "Top of first page",
        "output_format": "String",
        "priority": false
      },
      {
        "name": "Date of Service",
        "description": "Date of the procedure",
        "location": "Service details section",
        "output_format": "MM/DD/YYYY",
        "priority": true
      }
    ]
  },
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

**Example Usage:**

```bash
curl http://localhost:8000/api/templates/1
```

---

### 3. Get Template by Name

Get a template by its unique name.

**Endpoint:** `GET /api/templates/by-name/{template_name}`

**Response:** Same as "Get Template by ID"

**Example Usage:**

```bash
curl http://localhost:8000/api/templates/by-name/Hospital%20A%20-%20General%20Surgery
```

---

### 4. Upload Template

Upload an Excel file and save it as a template.

**Endpoint:** `POST /api/templates/upload`

**Form Data:**

- `name` (string, required): Unique name for the template
- `description` (string, optional): Description of the template
- `excel_file` (file, required): Excel file (.xlsx or .xls)

**Excel File Format:**
The Excel file should have the same structure as your existing instruction files:

- **Row 1 (Header)**: Field names (e.g., "Patient Name", "DOB", "MRN")
- **Row 2**: Field descriptions
- **Row 3**: Location hints (where to find the data)
- **Row 4**: Output format specifications
- **Row 5**: Priority field markers ("YES" or "NO")

**Response:**

```json
{
  "message": "Template uploaded successfully",
  "template_id": 5,
  "name": "Hospital B - Cardiology",
  "fields_count": 25
}
```

**Example Usage:**

```bash
curl -X POST http://localhost:8000/api/templates/upload \
  -F "name=Hospital B - Cardiology" \
  -F "description=Cardiology procedures for Hospital B" \
  -F "excel_file=@/path/to/instructions.xlsx"
```

---

### 5. Update Template

Update an existing template's name, description, or field definitions.

**Endpoint:** `PUT /api/templates/{template_id}`

**Form Data:**

- `name` (string, optional): New name for the template
- `description` (string, optional): New description
- `excel_file` (file, optional): New Excel file to replace field definitions

**Response:**

```json
{
  "message": "Template updated successfully",
  "template": {
    "id": 5,
    "name": "Hospital B - Cardiology Updated",
    "description": "Updated description",
    "template_data": { ... },
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-02T10:30:00"
  }
}
```

**Example Usage:**

```bash
# Update only the description
curl -X PUT http://localhost:8000/api/templates/5 \
  -F "description=New description for cardiology template"

# Update with a new Excel file
curl -X PUT http://localhost:8000/api/templates/5 \
  -F "excel_file=@/path/to/updated_instructions.xlsx"

# Update everything
curl -X PUT http://localhost:8000/api/templates/5 \
  -F "name=Hospital B - Cardiology 2024" \
  -F "description=Updated for 2024" \
  -F "excel_file=@/path/to/new_instructions.xlsx"
```

---

### 6. Delete Template

Delete a template from the database.

**Endpoint:** `DELETE /api/templates/{template_id}`

**Response:**

```json
{
  "message": "Template 5 deleted successfully"
}
```

**Example Usage:**

```bash
curl -X DELETE http://localhost:8000/api/templates/5
```

---

### 7. Export Template as Excel

Download a template back to Excel format.

**Endpoint:** `POST /api/templates/{template_id}/export`

**Response:** Excel file download

**Example Usage:**

```bash
curl -X POST http://localhost:8000/api/templates/1/export \
  --output downloaded_template.xlsx
```

---

## Using Templates in Your Application

### Option 1: Frontend Integration

1. **List available templates** when user starts extraction
2. **Let user select** from dropdown or choose to upload new Excel
3. **Fetch template data** and use it to generate extraction prompts
4. **Process PDFs** using the template field definitions

### Option 2: Programmatic Usage (Python)

```python
import requests

# Get a template
response = requests.get('http://localhost:8000/api/templates/1')
template = response.json()

# Use template data for extraction
field_definitions = template['template_data']['fields']

# Generate extraction prompt from template
from field_definitions import generate_extraction_prompt

# You can create an in-memory Excel file or use the field definitions directly
# ... your extraction logic here
```

### Option 3: Integration with Existing Upload Endpoint

You could modify your existing `/upload` endpoint to accept an optional `template_id` parameter:

```python
@app.post("/upload")
async def upload_files(
    template_id: int = Form(None),
    excel_file: UploadFile = File(None),
    # ... other parameters
):
    if template_id:
        # Use template from database
        from db_utils import get_template
        template = get_template(template_id=template_id)
        field_definitions = template['template_data']['fields']
        # Use field_definitions for extraction
    elif excel_file:
        # Use uploaded Excel file
        # ... existing logic
    else:
        raise HTTPException(400, "Either template_id or excel_file must be provided")
```

---

## Common Workflows

### Workflow 1: Save a New Template

1. Upload your Excel instruction file via web interface or API
2. Give it a descriptive name (e.g., "Hospital A - Emergency")
3. Add optional description
4. Template is saved and ready to use

### Workflow 2: Use an Existing Template

1. List available templates
2. Select the appropriate template by name or ID
3. Use template data for PDF extraction
4. No need to upload Excel file again

### Workflow 3: Update a Template

1. Get the template you want to update
2. Upload a new Excel file or change name/description
3. Template is updated for future use
4. All historical data remains unchanged

### Workflow 4: Manage Templates

1. Search templates by name or description
2. View template details to see field definitions
3. Export template as Excel for backup or sharing
4. Delete outdated templates

---

## Benefits

1. **Efficiency**: No need to upload the same Excel file repeatedly
2. **Version Control**: Track when templates were created and updated
3. **Reusability**: Share templates across different extraction jobs
4. **Organization**: Group templates by hospital, specialty, or use case
5. **Searchability**: Find templates quickly by name or description
6. **Flexibility**: Edit templates without losing historical data

---

## Excel File Format Reference

Your Excel instruction files should follow this structure:

```
| Patient Name | DOB        | MRN     | Date of Service | ... |
|--------------|------------|---------|-----------------|-----|
| Full name    | Birth date | Medical | Service date    | ... |  (Description row)
| Top of page  | Header     | Header  | Service section | ... |  (Location row)
| String       | MM/DD/YYYY | String  | MM/DD/YYYY      | ... |  (Format row)
| NO           | NO         | NO      | YES             | ... |  (Priority row)
```

- **Row 1**: Column headers (field names)
- **Row 2**: Field descriptions
- **Row 3**: Location hints
- **Row 4**: Output format
- **Row 5**: Priority ("YES" for high-priority fields, "NO" for normal)

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200**: Success
- **400**: Bad request (invalid file format, missing parameters)
- **404**: Template not found
- **500**: Server error

Error response format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Database Connection

The feature uses the existing PostgreSQL connection configured in `db_utils.py`:

```python
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://postgres:password@host:port/database'
)
```

Make sure your `DATABASE_URL` environment variable is set correctly.

---

## Next Steps

1. **Test the API**: Use the provided curl examples or Postman
2. **Integrate with Frontend**: Add template selection UI to your Vue.js frontend
3. **Migrate Existing Files**: Upload your existing Excel files as templates
4. **Update Extraction Logic**: Modify PDF extraction to use templates

---

## Technical Notes

- Templates are stored as JSONB in PostgreSQL for efficient querying
- Field definitions preserve all metadata (description, location, format, priority)
- The system is backward compatible - existing Excel upload workflow still works
- Template names must be unique across the database
- Templates are isolated per database (perfect for multi-tenant setups)

---

## Support

For issues or questions about the Instruction Templates feature:

1. Check database connection in `db_utils.py`
2. Verify PostgreSQL is running and accessible
3. Ensure the `instruction_templates` table was created (check logs on startup)
4. Test endpoints using provided curl examples
