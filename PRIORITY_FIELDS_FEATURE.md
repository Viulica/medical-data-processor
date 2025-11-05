# Priority Fields Feature

## Overview

The priority fields feature allows you to mark specific fields for separate, focused API calls during data extraction. This improves accuracy for fields that are difficult to extract or particularly important.

## How It Works

### Excel File Structure

Your Excel field definitions file should now have **4 rows** (instead of 3):

1. **Row 1 (Header)**: Field names (column headers)
2. **Row 2**: Description - What the field represents
3. **Row 3**: Location - Where to find this information in the PDF
4. **Row 4**: Output Format - Expected format of the extracted data
5. **Row 5 (NEW)**: Priority - Set to "YES" (case insensitive) to mark as priority field

### Example Excel Layout

```
| Patient Name | MRN      | Date of Birth | Primary Subsc ID |
|--------------|----------|---------------|------------------|
| Full name    | Medical  | Patient's DOB | Primary insurance|
| Top of form  | Upper    | Personal info | Insurance section|
| String       | String   | MM/DD/YYYY    | String           |
| NO           | YES      | NO            | YES              |
```

In this example:

- `MRN` and `Primary Subsc ID` are marked as priority fields (YES)
- `Patient Name` and `Date of Birth` are normal fields (NO or blank)

## Processing Flow

### Without Priority Fields (Original):

1. One API call per PDF extracts ALL fields together
2. Total API calls = 1 per PDF

### With Priority Fields (New):

1. One API call extracts all NORMAL (non-priority) fields
2. Separate API calls extract each PRIORITY field individually
3. Results are merged together automatically
4. Total API calls = 1 + (number of priority fields) per PDF

## Benefits

1. **Better Accuracy**: Priority fields get dedicated model attention
2. **Focused Extraction**: Model instructions are specific to one field at a time
3. **Transparent**: Final output CSV/XLSX looks exactly the same
4. **Flexible**: Choose which fields need extra accuracy

## Output

The final CSV/XLSX files remain unchanged:

- Same column structure
- Same data format
- No indication of which fields were processed as priority
- Users see a seamless, unified result

## Example Use Cases

Mark these types of fields as priority:

- **Critical IDs**: MRN, Account Number, Insurance IDs
- **Complex Dates**: Fields with multiple date formats
- **Amounts**: Charges, payments that must be accurate
- **Challenging Fields**: Fields the model often gets wrong

## Performance Considerations

- Each priority field adds one API call per PDF
- Example: 100 PDFs with 2 priority fields = 300 API calls (100 normal + 200 priority)
- Recommended: Limit priority fields to 2-3 most critical fields
- Processing time increases proportionally with priority fields

## Logs

During processing, you'll see messages like:

```
🎯 Priority fields (separate API calls): MRN, Primary Subsc ID
📊 Normal fields (single API call): 25 fields
🎯 Processing 2 priority field(s) for patient_001.pdf
✅ Merged priority field 'MRN' for patient_001.pdf
✅ Merged priority field 'Primary Subsc ID' for patient_001.pdf
```

## Implementation Details

### Modified Files

1. **field_definitions.py**:

   - Reads Priority row (row 4) from Excel
   - Separates fields into priority vs normal
   - Generates focused prompts for priority fields

2. **2-extract_info.py**:
   - Extracts normal fields with one API call
   - Extracts each priority field separately
   - Merges all results into unified output

### Backward Compatibility

- If Priority row is missing or empty, all fields treated as normal
- Existing Excel files without Priority row work unchanged
- No breaking changes to API or output format
