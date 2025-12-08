# Gemini 2.5 Flash PDF Splitting Feature

## Overview

This feature provides a new, faster, and more efficient method for splitting PDFs using Google's Gemini 2.5 Flash model instead of OCR-based text detection.

## Key Benefits

1. **⚡ Much Faster**: Gemini analyzes PDF pages directly without requiring resource-intensive OCR
2. **🚀 Won't Clog Server**: Processes pages in batches without spawning heavy OCR processes
3. **🔥 Parallel Processing**: Uses 3 threads to process multiple batches simultaneously
4. **🎯 More Accurate**: Gemini can understand document structure and text context better than OCR
5. **💰 Cost Efficient**: Batch processing reduces API calls (30 pages per call by default)

## Architecture

### Backend Components

1. **Script**: `/backend/current/1-split_pdf_gemini.py`

   - Main splitting logic using Gemini 2.5 Flash
   - **Parallel Processing**: Uses ThreadPoolExecutor with 3 workers by default
   - Processes PDF in configurable batches (default: 30 pages per API call)
   - Handles JSON parsing with 3 retry attempts
   - Prevents off-by-one indexing errors with careful page number conversion
   - Sanitizes LLM output (removes markdown code blocks, validates JSON structure)
   - Results properly ordered to prevent mixing up batch results

2. **API Endpoint**: `/split-pdf-gemini` in `backend/main.py`

   - Accepts: PDF file, filter string, batch size (optional, default: 30), model name (optional), max_workers (optional, default: 3)
   - Returns: Job ID for status polling
   - Background processing with proper timeout (30 minutes)
   - Memory cleanup after processing
   - Supports parallel processing for faster results

3. **Legacy Endpoint**: `/split-pdf` (OCR-based, kept for backward compatibility)

### Frontend Components

1. **UI**: Split PDF tab in `frontend/src/App.vue`
   - Toggle between Gemini and OCR methods (Gemini is default/recommended)
   - Batch size configuration (10-100 pages, default: 30)
   - Uses 3 parallel threads for faster processing
   - Method selection clearly labeled with benefits
   - Same UX pattern as other tabs (upload, process, download)

## How It Works

### Process Flow

1. **User uploads PDF** and specifies filter text (e.g., "Patient Address")
2. **Frontend sends request** to `/split-pdf-gemini` endpoint
3. **Backend processes in parallel**:
   - PDF split into batches (30 pages each by default)
   - **3 batches processed simultaneously** using ThreadPoolExecutor
   - Each batch sent to Gemini with structured prompt
   - Gemini returns JSON with matching page numbers
   - Results collected and **sorted by original order** to prevent mixing
   - Pages converted from batch-relative to document-absolute indexing
   - All matching pages collected in correct sequence
   - PDF split into sections based on matching pages
4. **Results packaged** as ZIP file
5. **User downloads** split PDFs

### Prompt Engineering

The prompt explicitly:

- Requests JSON output only
- Specifies exact JSON structure: `{"matching_pages": [1, 3, 5]}`
- Clarifies page numbering (1-based within batch)
- Handles edge cases (no matches returns empty array)
- Prevents hallucinations by requesting no additional text

### Error Handling

- **JSON Parsing**: 3 retry attempts with exponential backoff
- **Invalid Page Numbers**: Out-of-range pages are skipped with warning
- **API Failures**: Retry with backoff, proper error messages
- **Empty Results**: Clear user message if no matches found
- **Timeouts**: 30-minute timeout (much longer than needed due to efficiency)

## Configuration

### Backend Environment Variables

- `GOOGLE_API_KEY`: Required for Gemini API access
- Standard OpenBLAS thread limits for memory management

### Frontend Settings

- **Method Toggle**: Checkbox to enable/disable Gemini (default: enabled)
- **Batch Size**: Number slider (10-100 pages, default: 30)
  - Lower: More API calls, faster individual calls
  - Higher: Fewer API calls, slower individual calls
  - Optimized: 30 pages with 3 parallel threads
- **Parallel Threads**: Fixed at 3 threads (not exposed in UI for simplicity)
  - Processes 3 batches simultaneously
  - Results properly ordered to prevent mixing

## Usage Example

### Command Line (Direct Script)

```bash
cd backend/current
python 1-split_pdf_gemini.py input.pdf output_folder "Patient Address" 30 gemini-2.5-flash 3
```

Args:

1. Input PDF path
2. Output folder path
3. Filter string (comma-separated for multiple: "Text1,Text2")
4. Batch size (optional, default: 30)
5. Model name (optional, default: gemini-2.5-flash)
6. Max workers (optional, default: 3)

### Web Interface

1. Navigate to "✂️ Split PDF" tab
2. Check "⚡ Use Gemini 2.5 Flash" (should be checked by default)
3. Optionally adjust batch size (30 is recommended, processes with 3 parallel threads)
4. Upload PDF file
5. Enter filter text (e.g., "Patient Address")
6. Click "Split PDF"
7. Wait for processing (status updates in real-time) - much faster with parallel processing!
8. Download ZIP file with split PDFs

## Technical Details

### Page Number Conversion

Critical to avoid off-by-one errors:

```
User sees: Page 1, 2, 3 (1-based)
PDF Reader: Page 0, 1, 2 (0-based)
Gemini sees: Batch pages 1, 2, 3 (1-based within batch)

Conversion:
1. Gemini returns batch-relative 1-based: [2, 4]
2. Convert to batch 0-based: [1, 3]
3. Convert to document 0-based: [batch_start + 1, batch_start + 3]
```

### JSON Sanitization

LLM might return:

````json
```json
{"matching_pages": [1, 3]}
````

`````

Sanitization:
1. Strip whitespace
2. Remove leading ````json` or ```` ```
3. Remove trailing ` ``` `
4. Parse JSON
5. Validate structure
6. Retry on failure

### Parallel Processing & Order Preservation

To prevent mixing up results when processing batches in parallel:

1. Each batch is assigned an **order index** (0, 1, 2, ...)
2. Batches are submitted to ThreadPoolExecutor (3 workers)
3. Results are collected with their order index: `(order_idx, matching_pages)`
4. After all batches complete, results are **sorted by order index**
5. Pages are extracted in correct sequence
6. Final list is sorted by page number

This ensures that even though batches process in parallel (and may complete in any order), the final results are always in the correct document order.

## Comparison: Gemini vs OCR

| Feature | Gemini 2.5 Flash | OCR Method |
|---------|------------------|------------|
| Speed | Very Fast (seconds) | Slow (minutes) |
| Parallel Processing | Yes (3 threads) | No |
| Server Load | Low | High |
| Accuracy | High | Medium |
| Memory Usage | Low | High |
| Resource Intensive | No | Yes |
| Concurrent Jobs | Many | Few |
| Timeout Risk | Very Low | High |
| Cost per Page | Low (30 pages/batch) | Medium |
| Pages per API Call | 30 | N/A (OCR) |

## Troubleshooting

### "No matching pages found"

- Check that filter text exactly matches what's in the PDF
- Try simpler/shorter filter text
- Ensure PDF is readable (not scanned as images)

### "Gemini splitting timed out"

- Reduce batch size (try 5-8 pages)
- Check API key is valid
- Verify internet connectivity

### "Failed to parse JSON"

- This is automatically retried 3 times
- If still failing, the PDF might be corrupted
- Try OCR method as fallback

## Future Improvements

Possible enhancements:
- Support for multiple filter strings with AND/OR logic
- Preview mode (show which pages match before splitting)
- Custom section naming patterns
- Merge consecutive short sections
- Support for regex patterns in filter text

## API Reference

### POST /split-pdf-gemini

**Request:**
- `pdf_file`: File (PDF)
- `filter_string`: String (text to search for)
- `batch_size`: Integer (optional, 10-100, default: 30)
- `model`: String (optional, default: "gemini-2.5-flash")
- `max_workers`: Integer (optional, 1-10, default: 3)

**Response:**
```json
{
  "job_id": "uuid",
  "message": "PDF uploaded and Gemini splitting started (parallel processing)"
}
`````

**Job Status: GET /job-status/{job_id}**

```json
{
  "status": "completed",
  "progress": 100,
  "message": "Gemini splitting completed successfully! Created 5 sections."
}
```

**Download: GET /download/{job_id}**

Returns ZIP file with split PDFs.
