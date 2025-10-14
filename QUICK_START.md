# Quick Start - Local Testing

## The Easiest Way 🚀

```bash
# From the project root directory
./start-local.sh
```

This will:

- ✅ Start backend on `http://localhost:8000`
- ✅ Start frontend on `http://localhost:8080`
- ✅ Show you the logs
- ✅ Clean up everything when you press Ctrl+C

Then open your browser to: **http://localhost:8080**

---

## Manual Way (if you prefer)

### Terminal 1 - Backend

```bash
cd backend
python3 main.py
```

### Terminal 2 - Frontend

```bash
cd frontend
npm run serve
```

Then open: **http://localhost:8080**

---

## First Time Setup

### Install Backend Dependencies

```bash
cd backend
pip3 install -r requirements.txt
```

### Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## Testing UNI Conversion with AI ICD Reordering

1. Go to `http://localhost:8080`
2. Find "UNI Conversion" section
3. Upload your CSV file
4. Watch the backend terminal for progress:
   ```
   🔍 Phase 1: Extracting ICD codes...
   🤖 Phase 2: AI reordering (10 workers)...
   📝 Phase 3: Processing...
   ```
5. Download the result!

---

## Configuration

### Frontend connects to backend via:

**File:** `frontend/src/App.vue` (line 1777)

```javascript
const API_URL = process.env.VUE_APP_API_URL || "http://localhost:8000";
```

### To use a different backend URL:

```bash
# Create frontend/.env
echo "VUE_APP_API_URL=http://your-url:8000" > frontend/.env
```

---

## Troubleshooting

### Port 8000 already in use?

```bash
# Find and kill the process
lsof -i :8000
kill -9 <PID>

# Or use a different port
PORT=8001 python3 main.py
```

### Port 8080 already in use?

Vue will automatically use the next available port (8081, 8082, etc.)

### Backend not connecting?

1. Check backend is running: `curl http://localhost:8000`
2. Should return: `{"status":"healthy",...}`

---

## View Logs

```bash
# Backend logs (if using start-local.sh)
tail -f /tmp/medical-backend.log

# Frontend logs (if using start-local.sh)
tail -f /tmp/medical-frontend.log
```

---

## Stop Everything

If using `start-local.sh`:

- Press **Ctrl+C** (stops both servers)

If running manually:

- Press **Ctrl+C** in each terminal

---

## Full Documentation

See [LOCAL_TESTING_GUIDE.md](LOCAL_TESTING_GUIDE.md) for complete documentation.

---

## Quick Test

```bash
# Test backend health
curl http://localhost:8000

# Should return:
# {"status":"healthy","message":"Medical Data Processor API is running",...}
```

---

**That's it! Happy testing! 🎉**
