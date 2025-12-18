# Modifier Exclusion Feature - QS Modifier Control

## Overview

This feature adds the ability to enable or disable specific modifiers (starting with QS) on a per-insurance basis. By default, all modifiers are enabled, but you can selectively disable them for specific insurance codes.

## What's New

### Database Changes

A new column `enable_qs` has been added to the `modifiers_config` table:

```sql
ALTER TABLE modifiers_config 
ADD COLUMN IF NOT EXISTS enable_qs BOOLEAN NOT NULL DEFAULT TRUE;
```

**Default Behavior:** All insurances have QS modifier enabled by default (TRUE).

### Modified Files

1. **`backend/db_utils.py`**
   - Updated table schema to include `enable_qs` column
   - Modified `get_modifiers_dict()` to return `enable_qs` setting
   - Updated `upsert_modifier()` to accept `enable_qs` parameter

2. **`backend/modifiers/generate_modifiers.py`**
   - Updated modifier generation logic to check `enable_qs` before adding QS modifier
   - QS modifier is now only added when:
     - `medicare_modifiers = TRUE`
     - `Anesthesia Type = 'MAC'`
     - `enable_qs = TRUE` (NEW!)

3. **`backend/main.py`**
   - Updated API endpoints to handle `enable_qs` parameter:
     - `POST /api/modifiers`
     - `PUT /api/modifiers/{mednet_code}`

4. **`backend/migrate_modifiers_to_db.py`**
   - Updated to preserve existing `enable_qs` settings during migration

5. **`backend/add_enable_qs_column.py`** (NEW)
   - Migration script to add the column to existing databases

## How to Use

### Step 1: Run the Database Migration

If you have an existing database, run the migration script to add the new column:

```bash
cd backend
python add_enable_qs_column.py
```

This will add the `enable_qs` column with a default value of `TRUE` for all existing records.

### Step 2: Disable QS for a Specific Insurance

#### Option A: Using the API

**Update existing insurance:**

```bash
curl -X PUT "http://localhost:5001/api/modifiers/YOUR_MEDNET_CODE" \
  -F "medicare_modifiers=true" \
  -F "bill_medical_direction=true" \
  -F "enable_qs=false"
```

**Create new insurance with QS disabled:**

```bash
curl -X POST "http://localhost:5001/api/modifiers" \
  -F "mednet_code=YOUR_MEDNET_CODE" \
  -F "medicare_modifiers=true" \
  -F "bill_medical_direction=true" \
  -F "enable_qs=false"
```

#### Option B: Direct Database Update

```sql
UPDATE modifiers_config 
SET enable_qs = FALSE 
WHERE mednet_code = 'YOUR_MEDNET_CODE';
```

### Step 3: Generate Modifiers

When you run the modifier generation process, the system will automatically respect the `enable_qs` setting:

```bash
cd backend/modifiers
python generate_modifiers.py input.csv output.csv
```

## Examples

### Example 1: Disable QS for a Single Insurance

**Scenario:** You want to disable QS modifier for MedNet Code "003D"

```sql
UPDATE modifiers_config 
SET enable_qs = FALSE 
WHERE mednet_code = '003D';
```

**Result:** 
- For insurance 003D: QS modifier will NOT be added even when Anesthesia Type = MAC
- For all other insurances: QS modifier works normally (when Medicare Modifiers = YES and Anesthesia Type = MAC)

### Example 2: Disable QS for Multiple Insurances

```sql
UPDATE modifiers_config 
SET enable_qs = FALSE 
WHERE mednet_code IN ('003D', '0546', '14601A');
```

### Example 3: Re-enable QS

```sql
UPDATE modifiers_config 
SET enable_qs = TRUE 
WHERE mednet_code = '003D';
```

## Verification

### Check Current Settings

```sql
SELECT mednet_code, medicare_modifiers, bill_medical_direction, enable_qs
FROM modifiers_config
WHERE mednet_code = 'YOUR_MEDNET_CODE';
```

### View All Insurances with QS Disabled

```sql
SELECT mednet_code, medicare_modifiers, bill_medical_direction, enable_qs
FROM modifiers_config
WHERE enable_qs = FALSE
ORDER BY mednet_code;
```

## Debug Output

When processing files, the system will log QS modifier decisions:

```
🔍 DEBUG Row 1 - MedNet Code: 003D
   From DB: medicare_modifiers=True, medical_direction=True, enable_qs=False
   QS modifier DISABLED for this insurance (Anesthesia Type: MAC, enable_qs: False)
```

Or when enabled:

```
🔍 DEBUG Row 1 - MedNet Code: 003D
   From DB: medicare_modifiers=True, medical_direction=True, enable_qs=True
   QS modifier added (Anesthesia Type: MAC, enable_qs: True)
```

## Future Enhancements

This feature lays the groundwork for additional modifier controls:

- `enable_gc` - Control GC modifier
- `enable_p_modifiers` - Control P1-P6 modifiers
- `enable_pt` - Control PT modifier
- `enable_m1_modifiers` - Control AA/QK/QZ/QX modifiers

The same pattern can be extended to add columns for each modifier type as needed.

## Technical Details

### Data Flow

1. **Configuration Load:**
   ```python
   modifiers_dict = get_modifiers_dict()
   # Returns: {mednet_code: (medicare_modifiers, medical_direction, enable_qs)}
   ```

2. **Modifier Decision:**
   ```python
   medicare_modifiers, medical_direction, enable_qs = modifiers_dict[mednet_code]
   
   if has_anesthesia_type and medicare_modifiers and enable_qs:
       if anesthesia_type == 'MAC':
           qs_modifier = 'QS'
   ```

3. **Database Storage:**
   - Column: `enable_qs BOOLEAN NOT NULL DEFAULT TRUE`
   - Default ensures backward compatibility

### Backward Compatibility

- **CSV Fallback:** If database is unavailable, CSV fallback defaults to `enable_qs = True`
- **API Default:** API endpoints default to `enable_qs = True` if not provided
- **Existing Records:** Migration sets `enable_qs = TRUE` for all existing records

## Troubleshooting

### Problem: QS modifier still appearing after disabling

**Check:**
1. Verify the database update was successful:
   ```sql
   SELECT enable_qs FROM modifiers_config WHERE mednet_code = 'YOUR_CODE';
   ```

2. Restart the application to reload configuration

3. Check that the Medicare Modifiers setting is still correct

### Problem: Migration script fails

**Solution:**
- Ensure database connection is working
- Check if column already exists (script is safe to run multiple times)
- Verify database user has ALTER TABLE permissions

## Summary

This feature provides fine-grained control over modifier generation, allowing you to:

✅ Enable/disable QS modifier per insurance  
✅ Maintain default behavior (all enabled)  
✅ Update settings via API or database  
✅ View debug output during generation  
✅ Extend to other modifiers in the future  

The implementation is backward-compatible and safe to deploy without affecting existing functionality.

