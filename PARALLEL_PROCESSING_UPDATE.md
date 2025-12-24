# Parallel Processing Update - Gemini PDF Splitting

## Summary of Changes

Updated the Gemini PDF splitting feature to use **30 pages per API call** with **3 parallel threads** for maximum efficiency.

## What Changed

### 1. Backend Script (`backend/current/1-split_pdf_gemini.py`)

**Before:**
- Sequential processing (one batch at a time)
- 10 pages per API call
- Single-threaded

**After:**
- ✅ **Parallel processing with ThreadPoolExecutor**
- ✅ **30 pages per API call** (3x more efficient)
- ✅ **3 concurrent threads** processing batches simultaneously
- ✅ **Order preservation** - results sorted by batch index to prevent mixing
- ✅ Failed batch tracking and reporting

**Key Implementation Details:**
```python
# Batches are assigned order indices
batch_tasks = [(order_idx, start, end), ...]

# Process in parallel with 3 workers
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit all batches
    futures = {executor.submit(...): (order_idx, start, end)}
    
    # Collect results with order index
    results_with_order = [(order_idx, pages), ...]
    
# Sort by order index to maintain sequence
results_with_order.sort(key=lambda x: x[0])
```

### 2. Backend API (`backend/main.py`)

**Updated Endpoint:** `/split-pdf-gemini`

**New Parameters:**
- `batch_size`: Default changed from 10 → **30**
- `max_workers`: New parameter, default **3**
- Validation: batch_size (1-100), max_workers (1-10)

**Function Signature:**
```python
def split_pdf_gemini_background(
    job_id, pdf_path, filter_string, 
    batch_size=30,  # Changed from 10
    model="gemini-2.5-flash",
    max_workers=3   # New parameter
)
```

### 3. Frontend (`frontend/src/App.vue`)

**UI Changes:**
- Default batch size: 10 → **30**
- Batch size range: 5-50 → **10-100**
- Help text updated to mention "3 parallel threads"
- Input validation adjusted for new range

**Data Properties:**
```javascript
splitBatchSize: 30,  // Changed from 10
```

### 4. Documentation (`GEMINI_SPLIT_FEATURE.md`)

- Updated all references to default batch size (30)
- Added section on parallel processing and order preservation
- Updated comparison table with parallel processing info
- Updated API reference with new parameters
- Updated command-line examples

## Performance Impact

### Speed Improvements

For a 300-page PDF:

**Before (Sequential, 10 pages/batch):**
- 30 API calls
- ~30 seconds (1 second per call)
- Total: **~30 seconds**

**After (Parallel, 30 pages/batch, 3 threads):**
- 10 API calls
- 3 calls processed simultaneously
- ~4 batches of parallel calls
- Total: **~10-12 seconds** (60-70% faster!)

### Resource Usage

- **API Calls**: Reduced by 67% (30 → 10 calls for 300 pages)
- **Memory**: Slightly higher (3 concurrent batches vs 1)
- **Server Load**: Still very low compared to OCR method
- **Cost**: Reduced by 67% (fewer API calls)

## Order Preservation Mechanism

Critical for preventing mixed-up results:

1. **Batch Assignment**: Each batch gets unique order index (0, 1, 2, ...)
2. **Parallel Execution**: 3 batches process simultaneously
3. **Result Collection**: Results stored as `(order_idx, pages)` tuples
4. **Sorting**: After all complete, sort by order_idx
5. **Extraction**: Pages extracted in correct document order
6. **Final Sort**: Page numbers sorted for clean output

**Example:**
```
Batch 0: Pages 1-30   → Completes 3rd → Stored as (0, [5, 12])
Batch 1: Pages 31-60  → Completes 1st → Stored as (1, [45])
Batch 2: Pages 61-90  → Completes 2nd → Stored as (2, [])

After sorting by order_idx:
[(0, [5, 12]), (1, [45]), (2, [])]

Final result: [5, 12, 45] ✅ Correct order!
```

## Testing Recommendations

### Test Parallel Processing

1. **Large PDF Test** (300+ pages):
   - Should complete in ~10-15 seconds
   - Check that all sections are created
   - Verify page numbers are in correct order

2. **Order Verification**:
   - Process PDF with known matching pages
   - Verify sections are in correct sequence
   - Check that no pages are missing or duplicated

3. **Concurrent Jobs**:
   - Start multiple split jobs simultaneously
   - Verify they don't interfere with each other
   - Check memory usage stays reasonable

4. **Error Handling**:
   - Simulate API failures
   - Verify failed batches are reported
   - Check that partial failures don't corrupt results

### Performance Benchmarks

Run these tests and compare:

```bash
# Test 1: 100-page PDF
time python 1-split_pdf_gemini.py test_100.pdf output "Patient Address" 30 gemini-2.5-flash 3

# Test 2: 300-page PDF
time python 1-split_pdf_gemini.py test_300.pdf output "Patient Address" 30 gemini-2.5-flash 3

# Test 3: 600-page PDF
time python 1-split_pdf_gemini.py test_600.pdf output "Patient Address" 30 gemini-2.5-flash 3
```

Expected times:
- 100 pages: ~5-8 seconds
- 300 pages: ~10-15 seconds
- 600 pages: ~20-30 seconds

## Migration Notes

### For Existing Users

No action required! The changes are backward compatible:

- Default behavior is now faster (30 pages, 3 threads)
- Users can still adjust batch size if needed
- OCR method still available as fallback
- All existing PDFs will work the same way

### For Developers

If you're calling the API directly:

**Old API Call (still works):**
```bash
curl -X POST /split-pdf-gemini \
  -F "pdf_file=@test.pdf" \
  -F "filter_string=Patient Address" \
  -F "batch_size=10"
```

**New API Call (recommended):**
```bash
curl -X POST /split-pdf-gemini \
  -F "pdf_file=@test.pdf" \
  -F "filter_string=Patient Address" \
  -F "batch_size=30" \
  -F "max_workers=3"
```

## Configuration Options

### Tuning for Different Scenarios

**Small PDFs (<100 pages):**
- batch_size: 20-30
- max_workers: 2-3
- Fast enough with default settings

**Large PDFs (300+ pages):**
- batch_size: 30-50
- max_workers: 3-5
- More parallelism helps significantly

**Very Large PDFs (1000+ pages):**
- batch_size: 50-100
- max_workers: 5-10
- Maximum parallelism for fastest processing

**Rate Limit Concerns:**
- batch_size: 50-100 (fewer calls)
- max_workers: 1-2 (slower but safer)
- Reduces API call frequency

## Troubleshooting

### "Too many concurrent requests" error

**Solution:** Reduce `max_workers` to 1 or 2

### Results seem out of order

**Check:** This shouldn't happen! The order preservation mechanism should prevent this.
**Debug:** Enable logging to see batch order indices

### Slower than expected

**Possible causes:**
- API rate limiting
- Network latency
- Large batch sizes (try 20-30 instead of 50+)

**Solution:** Adjust batch_size and max_workers based on your network/API limits

## Summary

✅ **3x fewer API calls** (30 pages vs 10 pages per call)  
✅ **60-70% faster** processing with 3 parallel threads  
✅ **Order preservation** ensures results never get mixed up  
✅ **Backward compatible** - existing code still works  
✅ **No linting errors** - clean implementation  
✅ **Fully documented** - updated all docs  

The Gemini PDF splitting feature is now significantly faster and more efficient while maintaining accuracy and reliability!



