from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import zipfile
import tempfile
import shutil
import uuid
from pathlib import Path
import sys
import subprocess
import json
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Processing API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for job status
job_status = {}

class ProcessingJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"
        self.progress = 0
        self.message = "Job created"
        self.result_file = None
        self.error = None

def process_pdfs_background(job_id: str, zip_path: str, excel_path: str, n_pages: int, excel_filename: str):
    """Background task to process PDFs"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Unzipping files..."
        job.progress = 10
        
        # Create temporary directory for processing
        temp_dir = Path(f"/tmp/processing_{job_id}")
        temp_dir.mkdir(exist_ok=True)
        
        # Unzip files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir / "input")
        
        job.message = "Files unzipped, starting processing..."
        job.progress = 20
        
        # Copy Excel file to temp directory
        excel_dest = temp_dir / "instructions" / excel_filename
        excel_dest.parent.mkdir(exist_ok=True)
        shutil.copy2(excel_path, excel_dest)
        
        # Set up environment for the processing script
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent / "current")
        
        # Run the processing script
        script_path = Path(__file__).parent / "current" / "2-extract_info.py"
        
        job.message = f"Processing PDFs (extracting first {n_pages} pages per patient)..."
        job.progress = 30
        
        # Run the script with subprocess
        result = subprocess.run([
            sys.executable, str(script_path),
            str(temp_dir / "input"),
            str(excel_dest),
            str(n_pages),  # n_pages parameter
            "3"   # max_workers
        ], capture_output=True, text=True, cwd=temp_dir, env=env)
        
        if result.returncode != 0:
            raise Exception(f"Processing failed: {result.stderr}")
        
        job.message = "Processing complete, preparing results..."
        job.progress = 80
        
        # Find the generated CSV file
        extracted_dir = temp_dir / "extracted"
        csv_files = list(extracted_dir.glob("*.csv"))
        
        if not csv_files:
            raise Exception("No CSV file generated")
        
        # Copy the CSV to a permanent location
        result_file = Path(f"/tmp/results/{job_id}.csv")
        result_file.parent.mkdir(exist_ok=True)
        shutil.copy2(csv_files[0], result_file)
        
        job.result_file = str(result_file)
        job.status = "completed"
        job.progress = 100
        job.message = "Processing completed successfully!"
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        os.unlink(zip_path)
        os.unlink(excel_path)
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Processing failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {str(e)}")

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    excel_file: UploadFile = File(...),
    n_pages: int = Form(..., ge=1, le=50)  # Validate page count between 1-50
):
    """Upload ZIP file, Excel instructions file, and page count"""
    
    # Validate file types
    if not zip_file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="First file must be a ZIP archive")
    
    if not excel_file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Second file must be an Excel file")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    job = ProcessingJob(job_id)
    job_status[job_id] = job
    
    # Save uploaded files
    zip_path = f"/tmp/{job_id}_archive.zip"
    excel_path = f"/tmp/{job_id}_instructions.xlsx"
    
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(zip_file.file, f)
    
    with open(excel_path, "wb") as f:
        shutil.copyfileobj(excel_file.file, f)
    
    # Start background processing
    background_tasks.add_task(process_pdfs_background, job_id, zip_path, excel_path, n_pages, excel_file.filename)
    
    return {"job_id": job_id, "message": "Files uploaded and processing started"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error
    }

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the processed CSV file"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if not job.result_file or not os.path.exists(job.result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        job.result_file,
        media_type="text/csv",
        filename=f"processed_data_{job_id}.csv"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 