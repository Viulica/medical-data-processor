# Local Testing Guide

## Overview

This guide will help you run the Medical Data Processor application locally for testing.

## Architecture

- **Backend**: FastAPI server (Python) - Port 8000
- **Frontend**: Vue.js app - Port 8080
- **API URL**: Frontend is configured to use `http://localhost:8000` by default

## Prerequisites

### 1. Python Requirements

```bash
# Python 3.8 or higher
python3 --version

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Node.js Requirements

```bash
# Node.js 14 or higher
node --version

# Install frontend dependencies
cd frontend
npm install
```

## Running the Application Locally

### Step 1: Start the Backend Server

```bash
# Navigate to backend directory
cd /Users/leon/Documents/MD/medical-data-processor/backend

# Run the FastAPI server
python main.py
```

**Expected output:**

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
🚀 Medical Data Processor API starting up...
📡 Port: 8000
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Backend will be available at:** `http://localhost:8000`

**Test the backend:**

```bash
# Open in browser or use curl
curl http://localhost:8000

# Should return:
# {"status":"healthy","message":"Medical Data Processor API is running","port":8000,"environment":"development"}
```

### Step 2: Start the Frontend Development Server

Open a **new terminal window** (keep backend running):

```bash
# Navigate to frontend directory
cd /Users/leon/Documents/MD/medical-data-processor/frontend

# Start Vue development server
npm run serve
```

**Expected output:**

```
  App running at:
  - Local:   http://localhost:8080/
  - Network: http://192.168.x.x:8080/

  Note that the development build is not optimized.
  To create a production build, run npm run build.
```

**Frontend will be available at:** `http://localhost:8080`

### Step 3: Access the Application

Open your browser and navigate to:

```
http://localhost:8080
```

The frontend will automatically connect to the backend at `http://localhost:8000`.

## Testing the UNI Conversion with AI Reordering

### Quick Test

1. **Prepare a test CSV file** with these columns:

   - `Procedure`
   - `POST-OP DIAGNOSIS`
   - `Post-op Diagnosis - Coded`
   - Any other UNI format columns

2. **Use the frontend:**

   - Go to `http://localhost:8080`
   - Find the "UNI Conversion" section
   - Upload your CSV file
   - Wait for processing (you'll see progress)
   - Download the converted file

3. **Or use curl to test directly:**

   ```bash
   # Upload CSV for conversion
   curl -X POST http://localhost:8000/convert-uni \
     -F "csv_file=@/path/to/your/test.csv"

   # Returns: {"job_id":"xxx-xxx-xxx","message":"UNI CSV uploaded and conversion started"}

   # Check status
   curl http://localhost:8000/status/YOUR_JOB_ID

   # Download result when complete
   curl http://localhost:8000/download/YOUR_JOB_ID?format=csv -o result.csv
   ```

### Testing the Multi-threaded ICD Reordering

The AI reordering happens automatically during UNI conversion. You'll see:

```
🔍 Phase 1: Extracting ICD codes from all rows...
✓ Extracted ICD codes from 150 rows

🤖 Phase 2: AI reordering 120 rows with 2+ ICD codes (10 workers)...
    📊 Progress: 12/120 (10.0%)
    📊 Progress: 24/120 (20.0%)
    ...
✓ AI reordering complete for 120 rows

📝 Phase 3: Processing 150 rows with field mapping...
```

Watch the backend terminal to see the progress!

## Configuration

### Frontend API URL

The frontend is configured to use the backend URL from environment variable or default to localhost:

**File:** `frontend/src/App.vue` (line 1777)

```javascript
const API_URL = process.env.VUE_APP_API_URL || "http://localhost:8000";
```

### To change the backend URL:

**Option 1: Environment Variable**

```bash
# Create .env file in frontend directory
echo "VUE_APP_API_URL=http://your-backend-url:8000" > frontend/.env

# Restart frontend
npm run serve
```

**Option 2: Edit App.vue directly**

```javascript
// Change line 1777 in frontend/src/App.vue
const API_URL = "http://your-custom-url:8000";
```

### Backend Port

**File:** `backend/main.py` (line 74)

```python
PORT = int(os.environ.get('PORT', 8000))
```

To change port:

```bash
# Set environment variable
export PORT=9000

# Or run with inline env var
PORT=9000 python main.py
```

## Troubleshooting

### Backend Issues

**1. Port already in use:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
PORT=8001 python main.py
```

**2. Missing dependencies:**

```bash
cd backend
pip install -r requirements.txt
```

**3. Google AI API errors:**

```
⚠️  Could not initialize AI client
```

- Check if API key is valid in `backend/uni-conversion/convert.py`
- Check internet connection

### Frontend Issues

**1. Port already in use:**

```bash
# Kill process on port 8080
lsof -i :8080
kill -9 <PID>

# Or Vue will automatically use next available port (8081, 8082, etc.)
```

**2. Cannot connect to backend:**

```
Error: Network Error
```

- Make sure backend is running on port 8000
- Check `http://localhost:8000` in browser
- Check frontend API_URL configuration

**3. Missing dependencies:**

```bash
cd frontend
rm -rf node_modules
npm install
```

## Testing Different Features

### 1. PDF Processing

```bash
# Upload ZIP of PDFs + Excel instructions
curl -X POST http://localhost:8000/upload \
  -F "zip_file=@pdfs.zip" \
  -F "excel_file=@instructions.xlsx" \
  -F "n_pages=5" \
  -F "model=gemini-2.5-flash-lite"
```

### 2. CPT Code Prediction

```bash
curl -X POST http://localhost:8000/predict-cpt \
  -F "csv_file=@data.csv" \
  -F "client=uni"
```

### 3. UNI Conversion (with AI ICD reordering)

```bash
curl -X POST http://localhost:8000/convert-uni \
  -F "csv_file=@uni_data.csv"
```

### 4. Generate Modifiers

```bash
curl -X POST http://localhost:8000/generate-modifiers \
  -F "csv_file=@data.csv"
```

### 5. Insurance Code Prediction

```bash
curl -X POST http://localhost:8000/predict-insurance-codes \
  -F "data_csv=@data.csv" \
  -F "enable_ai=true"
```

## Monitoring

### Check Backend Health

```bash
curl http://localhost:8000/health
```

### Check Memory Usage

```bash
curl http://localhost:8000/memory
```

### View All Jobs

```bash
curl http://localhost:8000/status/YOUR_JOB_ID
```

## Development Workflow

### Making Changes

**Backend changes:**

1. Edit Python files in `backend/`
2. Stop backend (Ctrl+C)
3. Restart: `python main.py`

**Frontend changes:**

1. Edit Vue files in `frontend/src/`
2. Changes auto-reload (hot module replacement)
3. No restart needed!

**UNI Conversion changes:**

1. Edit `backend/uni-conversion/convert.py`
2. Restart backend
3. Test with new CSV upload

### Viewing Logs

**Backend logs:**

- Shown in terminal where you ran `python main.py`
- Look for:
  - `🔍 Phase 1: Extracting ICD codes...`
  - `🤖 Phase 2: AI reordering...`
  - `📝 Phase 3: Processing...`

**Frontend logs:**

- Open browser DevTools (F12)
- Check Console tab for errors
- Check Network tab for API calls

## Production Deployment

The app is configured for Railway deployment:

- Backend uses `railway.json` and `Procfile`
- Frontend is built and deployed separately
- Environment variables set in Railway dashboard

**For local testing, you don't need to worry about production config!**

## Quick Start Summary

```bash
# Terminal 1: Backend
cd backend
python main.py

# Terminal 2: Frontend
cd frontend
npm run serve

# Browser
# Open http://localhost:8080
```

That's it! 🚀

## Need Help?

- Backend not starting? Check `requirements.txt` is installed
- Frontend not connecting? Check backend is running on port 8000
- AI reordering not working? Check API key in `convert.py`
- Port conflicts? Use different ports with environment variables
