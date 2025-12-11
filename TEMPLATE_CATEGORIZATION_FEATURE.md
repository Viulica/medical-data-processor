# Template Categorization Feature

## Overview

This feature adds the ability to categorize instruction templates into three distinct categories:

- **DEMO** - For demographic/basic information templates
- **DEMO + CPT + ICD** - For comprehensive templates including CPT and ICD codes
- **CHARGE** - For charge-related templates

## Implementation Summary

### 1. Database Changes (`backend/db_utils.py`)

#### Schema Updates

- Added `category` column (VARCHAR(50)) to the `instruction_templates` table
- Added index on the `category` column for efficient filtering
- Implemented automatic migration that adds the column to existing databases without data loss

#### Function Updates

- **`get_all_templates()`**: Added `category` parameter for filtering templates by category
- **`get_template()`**: Updated to return the category field
- **`create_template()`**: Added `category` parameter to support categorization during template creation
- **`update_template()`**: Added `category` parameter to allow updating template categories

### 2. Backend API Changes (`backend/main.py`)

#### Endpoint Updates

**GET `/api/templates`**

- Added optional `category` query parameter for filtering templates
- Example: `/api/templates?category=DEMO`

**POST `/api/templates/upload`**

- Added optional `category` form field
- Validates category against allowed values
- Returns category in response

**PUT `/api/templates/{template_id}`**

- Added optional `category` form field for updating template category
- Validates category against allowed values
- Returns updated category in response

### 3. Frontend Changes (`frontend/src/App.vue`)

#### Data Model

- Added `category` field to `currentTemplate` object
- Added `templateCategories` array with the three allowed categories
- Added `templateCategoryFilter` for filtering templates by category

#### UI Components

**Add Template Modal**

- Added category dropdown selector with three options
- Category selection is optional

**Edit Template Modal**

- Added category dropdown selector
- Pre-populates with existing category when editing
- Allows changing the category

**Templates Grid**

- Added visual category badges to template cards
- Three color-coded badge styles:
  - **DEMO**: Purple gradient badge
  - **DEMO + CPT + ICD**: Pink/red gradient badge
  - **CHARGE**: Blue/cyan gradient badge

**Templates Tab Header**

- Added category filter dropdown
- Allows filtering templates by category or viewing all categories

#### Methods Added

- **`onTemplateCategoryFilterChange()`**: Handles category filter changes and reloads templates
- **`getCategoryBadgeClass()`**: Returns appropriate CSS class for category badge styling

#### API Integration

- Updated `saveTemplate()` to include category in FormData
- Updated `updateTemplate()` to include category in FormData
- Updated `editTemplate()` to populate category when editing
- Updated `loadTemplates()` to pass category filter parameter
- Updated `closeTemplateModals()` to reset category field

### 4. CSS Styling (`frontend/src/App.vue`)

Added beautiful gradient-based category badge styles:

```css
.category-badge-demo      /* Purple gradient */
/* Purple gradient */
.category-badge-full      /* Pink/red gradient */
.category-badge-charge; /* Blue/cyan gradient */
```

## Usage

### Creating a Template with Category

1. Go to Settings → Field Templates
2. Click "➕ Upload New Template"
3. Fill in template name and description
4. Select a category from the dropdown (optional)
5. Upload the Excel file
6. Click "Upload Template"

### Editing Template Category

1. Go to Settings → Field Templates
2. Click the edit (pencil) icon on any template
3. Change the category in the dropdown
4. Click "Save Changes"

### Filtering Templates by Category

1. Go to Settings → Field Templates
2. Use the category dropdown next to the search box
3. Select a category to filter, or "All Categories" to see everything

### Viewing Template Categories

- Templates with assigned categories display a colored badge below the template name
- The badge color indicates the category type
- Templates without a category have no badge

## Database Migration

The feature includes automatic database migration that:

- Checks if the `category` column exists
- Adds the column and index if missing
- Does not affect existing template data
- Runs automatically when the backend server starts

## API Examples

### Get all DEMO templates

```bash
GET /api/templates?category=DEMO
```

### Create template with category

```bash
POST /api/templates/upload
Content-Type: multipart/form-data

name: "My Template"
description: "Template description"
category: "DEMO + CPT + ICD"
excel_file: [file]
```

### Update template category

```bash
PUT /api/templates/123
Content-Type: multipart/form-data

category: "CHARGE"
```

## Technical Notes

- Category is optional - templates can exist without a category
- Category values are validated on the backend
- The three allowed categories are enforced by validation
- Database migration is idempotent and safe to run multiple times
- Category filtering is performed at the database level for efficiency
- Category badges use modern CSS gradients for visual appeal

## Benefits

1. **Organization**: Easily organize templates by their purpose
2. **Quick Filtering**: Rapidly find templates of a specific type
3. **Visual Identification**: Instantly recognize template types with color-coded badges
4. **Flexibility**: Category assignment is optional and can be changed at any time
5. **Scalability**: Database indexes ensure efficient filtering even with many templates
