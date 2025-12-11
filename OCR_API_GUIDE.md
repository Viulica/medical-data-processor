# Simple OCR API Solutions for PDF Splitting

This guide shows you the **simplest** OCR API options to replace your current Gemini and local OCR solutions.

## 🏆 Recommended Options (Easiest to Hardest)

### 1. OCR.space ⭐ **RECOMMENDED - Literally 5 lines of code**

**Why this one:**

- Free tier: 25,000 requests/month
- No complex setup - just get an API key
- Direct PDF support
- Fast and reliable

**Setup:**

```bash
# 1. Get free API key (2 minutes)
# Go to: https://ocr.space/ocrapi
# Sign up and copy your API key

# 2. Set environment variable
export OCRSPACE_API_KEY="your_api_key_here"

# 3. Install (if needed)
pip install requests

# 4. Run it!
python backend/current/1-split_pdf_ocrspace.py input.pdf output "Patient Address"
```

**Pros:**

- ✅ Super simple - like "resend" for emails
- ✅ Free tier is generous
- ✅ No Google Cloud setup needed
- ✅ Works with PDF directly

**Cons:**

- ❌ Rate limits on free tier
- ❌ Less accurate than Google Vision for complex layouts

---

### 2. LlamaOCR (LlamaParse) 🦙 **BEST FOR DOCUMENTS**

**Why this one:**

- Free tier: 1,000 pages/day
- Specifically designed for documents
- Best for medical documents with tables/forms
- Very accurate

**Setup:**

```bash
# 1. Get free API key (2 minutes)
# Go to: https://cloud.llamaindex.ai/api-key
# Sign up and copy your API key

# 2. Install
pip install llama-parse

# 3. Set environment variable
export LLAMA_CLOUD_API_KEY="your_api_key_here"

# 4. Run it!
python backend/current/1-split_pdf_llamaocr.py input.pdf output "Patient Address"
```

**Pros:**

- ✅ Built specifically for documents
- ✅ Excellent with medical forms
- ✅ Handles tables and complex layouts well
- ✅ 1,000 pages/day is enough for most use cases

**Cons:**

- ❌ Slightly slower than OCR.space (but more accurate)
- ❌ Lower rate limit than OCR.space

---

### 3. Google Cloud Vision API (If you want maximum accuracy)

**Why this one:**

- Most accurate OCR available
- You already have Google setup for Gemini
- Can reuse same project

**Setup:**

```bash
# 1. Use same Google Cloud project as Gemini
# 2. Enable Cloud Vision API in console
# 3. Use same GOOGLE_API_KEY or service account

# 4. Install
pip install google-cloud-vision

# 5. Run (I can create this file if you want)
```

**Pros:**

- ✅ Best accuracy
- ✅ Same Google ecosystem as Gemini
- ✅ Good free tier (1,000 pages/month)

**Cons:**

- ❌ More complex setup
- ❌ Need Google Cloud project
- ❌ Not as "simple" as OCR.space

---

## 📊 Quick Comparison

| Feature             | OCR.space       | LlamaOCR        | Google Vision        |
| ------------------- | --------------- | --------------- | -------------------- |
| **Ease of setup**   | 🟢 5 mins       | 🟢 5 mins       | 🟡 15 mins           |
| **Code simplicity** | 🟢 Super simple | 🟢 Super simple | 🟡 Moderate          |
| **Accuracy**        | 🟡 Good         | 🟢 Excellent    | 🟢 Excellent         |
| **Free tier**       | 25,000/month    | 1,000/day       | 1,000/month          |
| **Speed**           | 🟢 Fast         | 🟡 Moderate     | 🟢 Fast              |
| **Medical docs**    | 🟡 OK           | 🟢 Great        | 🟢 Great             |
| **Cost**            | Free → $6/mo    | Free → $150/mo  | Free → Pay-as-you-go |

---

## 🚀 My Recommendation

**Start with OCR.space** because:

1. ✅ Literally 5 lines of code (like resend for emails)
2. ✅ No complex setup
3. ✅ 25,000 requests/month is plenty
4. ✅ If you need more accuracy later, switch to LlamaOCR

**Switch to LlamaOCR if:**

- Your medical documents have complex layouts/tables
- You need better accuracy
- 1,000 pages/day is enough for you

---

## 📝 Usage Examples

### OCR.space

```bash
# Basic usage
python backend/current/1-split_pdf_ocrspace.py input.pdf output "Patient Address"

# Multiple filter strings
python backend/current/1-split_pdf_ocrspace.py input.pdf output "Patient Address,Date of Birth"

# More workers (faster, but watch rate limits)
python backend/current/1-split_pdf_ocrspace.py input.pdf output "Patient Address" 10
```

### LlamaOCR

```bash
# Basic usage (same as above)
python backend/current/1-split_pdf_llamaocr.py input.pdf output "Patient Address"

# Keep workers low (3-5) to respect API limits
python backend/current/1-split_pdf_llamaocr.py input.pdf output "Patient Address" 3
```

---

## 🔧 Integrating into main.py

If you want me to integrate either solution into your `main.py` Flask/FastAPI app, let me know which one you prefer!

**Integration is simple:**

```python
# Import the function
from current.split_pdf_ocrspace import split_pdf_with_ocrspace

# Use it in your endpoint
result = split_pdf_with_ocrspace(
    input_pdf_path=uploaded_file,
    output_folder=output_dir,
    filter_strings=["Patient Address"],
    max_workers=5
)
```

---

## ❓ Which one should I use?

**TL;DR:** Start with **OCR.space** - it's the simplest and most like "resend" for emails.

If you need better accuracy for complex medical forms → **LlamaOCR**

If you need absolute best accuracy and already use Google → **Google Vision**
