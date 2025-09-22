import os
import sys
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    logger.info("✅ FastAPI imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import FastAPI: {e}")
    raise e

try:
    import zipfile
    import tempfile
    import shutil
    import uuid
    from pathlib import Path
    import subprocess
    import json
    from typing import List
    logger.info("✅ Standard library modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import standard library module: {e}")
    raise e



# Global port configuration - this ensures consistency
PORT = int(os.environ.get('PORT', 8000))

app = FastAPI(title="PDF Processing API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Log when the application starts"""
    logger.info("🚀 Medical Data Processor API starting up...")
    logger.info(f"📡 Port: {PORT}")
    logger.info(f"🔑 Google API Key: {'Set' if os.environ.get('GOOGLE_API_KEY') else 'Not set'}")
    logger.info(f"🌍 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'development')}")
    logger.info(f"🏗️ Railway Project: {os.environ.get('RAILWAY_PROJECT_ID', 'unknown')}")
    logger.info(f"🚂 Railway Service: {os.environ.get('RAILWAY_SERVICE_NAME', 'unknown')}")
    logger.info(f"🔧 PORT env var: {os.environ.get('PORT', 'not set')}")
    logger.info(f"🔧 GLOBAL PORT: {PORT}")

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

def process_pdfs_background(job_id: str, zip_path: str, excel_path: str, n_pages: int, excel_filename: str, model: str = "gemini-2.5-pro"):
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
            "7",  # max_workers
            model  # model parameter
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

def split_pdf_background(job_id: str, pdf_path: str, filter_string: str):
    """Background task to split PDF using the existing detection script"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting PDF splitting..."
        job.progress = 10
        
        # Use existing input and output folders
        current_dir = Path(__file__).parent / "current"
        input_dir = current_dir / "input"
        output_dir = current_dir / "output"
        
        # Create input and output folders if they don't exist
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Clear output folder first
        if output_dir.exists():
            for file in output_dir.glob("*.pdf"):
                file.unlink()
        
        # Copy PDF to input folder
        pdf_filename = Path(pdf_path).name
        input_pdf_path = input_dir / pdf_filename
        shutil.copy2(pdf_path, input_pdf_path)
        
        # Verify the file was copied
        if not input_pdf_path.exists():
            raise Exception(f"Failed to copy PDF to input folder: {input_pdf_path}")
        
        logger.info(f"PDF copied to input folder: {input_pdf_path}")
        logger.info(f"Input folder now contains: {[f.name for f in input_dir.glob('*.pdf')]}")
        
        job.message = "PDF copied, running split script..."
        job.progress = 30
        
        # Set up environment for the splitting script
        env = os.environ.copy()
        env['PYTHONPATH'] = str(current_dir)
        
        # Original script path
        script_path = current_dir / "1-split_pdf_by_detections.py"
        
        # Create a temporary script with custom filter string
        temp_script_path = current_dir / f"temp_split_{job_id}.py"
        
        # Read the original script
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Replace the filter strings with the custom one
        custom_filter_lines = f'FILTER_STRINGS = ["{filter_string}"]\n'
        
        # Find and replace the FILTER_STRINGS line
        lines = script_content.split('\n')
        new_lines = []
        filter_replaced = False
        
        for line in lines:
            if line.strip().startswith('FILTER_STRINGS = ') and not filter_replaced:
                new_lines.append(custom_filter_lines)
                filter_replaced = True
            else:
                new_lines.append(line)
        
        # Write the modified script
        with open(temp_script_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        job.message = f"Splitting PDF into sections using filter: '{filter_string}'..."
        job.progress = 50
        
        # Run the modified script
        logger.info(f"Running split script with custom filter: {filter_string}")
        logger.info(f"Working directory: {current_dir}")
        logger.info(f"Input folder contents: {list(input_dir.glob('*.pdf'))}")
        
        result = subprocess.run([
            sys.executable, str(temp_script_path)
        ], capture_output=True, text=True, cwd=current_dir, env=env)
        
        logger.info(f"Script stdout: {result.stdout}")
        logger.info(f"Script stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise Exception(f"Splitting failed: {result.stderr}")
        
        job.message = "Creating ZIP archive of split PDFs..."
        job.progress = 80
        
        # Create ZIP file with all split PDFs
        zip_path = Path(f"/tmp/results/{job_id}_split_pdfs.zip")
        zip_path.parent.mkdir(exist_ok=True)
        
        # Find all PDF files created by the script
        pdf_files = list(output_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise Exception("No PDF files were created by the splitting script")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for pdf_file in pdf_files:
                zipf.write(pdf_file, pdf_file.name)
        
        job.result_file = str(zip_path)
        job.status = "completed"
        job.progress = 100
        job.message = f"PDF splitting completed successfully! Created {len(pdf_files)} sections."
        
        # Clean up uploaded file, input folder, and temp script
        os.unlink(pdf_path)
        if input_pdf_path.exists():
            input_pdf_path.unlink()
            logger.info(f"Cleaned up input file: {input_pdf_path}")
        
        # Clean up temp script
        if temp_script_path.exists():
            temp_script_path.unlink()
            logger.info(f"Cleaned up temp script: {temp_script_path}")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"PDF splitting failed: {str(e)}"
        logger.error(f"Split job {job_id} failed: {str(e)}")
        
        # Clean up on error
        try:
            if 'input_pdf_path' in locals() and input_pdf_path.exists():
                input_pdf_path.unlink()
                logger.info(f"Cleaned up input file on error: {input_pdf_path}")
            if 'temp_script_path' in locals() and temp_script_path.exists():
                temp_script_path.unlink()
                logger.info(f"Cleaned up temp script on error: {temp_script_path}")
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    excel_file: UploadFile = File(...),
    n_pages: int = Form(..., ge=1, le=50),  # Validate page count between 1-50
    model: str = Form(default="gemini-2.5-pro")  # Model parameter with default
):
    """Upload ZIP file, Excel instructions file, and page count"""
    
    try:
        logger.info(f"Received upload request - zip: {zip_file.filename}, excel: {excel_file.filename}, pages: {n_pages}, model: {model}")
        
        # Validate file types
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="First file must be a ZIP archive")
        
        if not excel_file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Second file must be an Excel file")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created job {job_id}")
        
        # Save uploaded files
        zip_path = f"/tmp/{job_id}_archive.zip"
        excel_path = f"/tmp/{job_id}_instructions.xlsx"
        
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        
        with open(excel_path, "wb") as f:
            shutil.copyfileobj(excel_file.file, f)
        
        logger.info(f"Files saved - zip: {zip_path}, excel: {excel_path}")
        
        # Start background processing
        background_tasks.add_task(process_pdfs_background, job_id, zip_path, excel_path, n_pages, excel_file.filename, model)
        
        logger.info(f"Background task started for job {job_id}")
        
        return {"job_id": job_id, "message": "Files uploaded and processing started"}
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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

@app.post("/split-pdf")
async def split_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    filter_string: str = Form(..., description="Text to search for in PDF pages for splitting")
):
    """Upload a single PDF file to split into sections"""
    
    try:
        logger.info(f"Received PDF split request - pdf: {pdf_file.filename}")
        
        # Validate file type
        if not pdf_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created split job {job_id}")
        
        # Save uploaded PDF
        pdf_path = f"/tmp/{job_id}_input.pdf"
        
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
        
        logger.info(f"PDF saved - path: {pdf_path}")
        
        # Start background processing
        background_tasks.add_task(split_pdf_background, job_id, pdf_path, filter_string)
        
        logger.info(f"Background split task started for job {job_id}")
        
        return {"job_id": job_id, "message": "PDF uploaded and splitting started"}
        
    except Exception as e:
        logger.error(f"PDF split upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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
    
    # Determine file type and filename based on job type
    if job.result_file.endswith('.zip'):
        filename = f"split_pdfs_{job_id}.zip"
        media_type = "application/zip"
    else:
        filename = f"processed_data_{job_id}.csv"
        media_type = "text/csv"
    
    return FileResponse(
        job.result_file,
        media_type=media_type,
        filename=filename
    )

@app.get("/")
async def root():
    """Root endpoint for Railway health check"""
    try:
        return {
            "status": "healthy", 
            "message": "Medical Data Processor API is running", 
            "port": PORT,
            "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'development')
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/debug/folders")
async def debug_folders():
    """Debug endpoint to check folder contents"""
    try:
        current_dir = Path(__file__).parent / "current"
        input_dir = current_dir / "input"
        output_dir = current_dir / "output"
        
        return {
            "current_dir": str(current_dir),
            "current_dir_exists": current_dir.exists(),
            "input_dir": str(input_dir),
            "input_dir_exists": input_dir.exists(),
            "input_files": [f.name for f in input_dir.glob("*")] if input_dir.exists() else [],
            "output_dir": str(output_dir),
            "output_dir_exists": output_dir.exists(),
            "output_files": [f.name for f in output_dir.glob("*")] if output_dir.exists() else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if input and output folders exist
        current_dir = Path(__file__).parent / "current"
        input_dir = current_dir / "input"
        output_dir = current_dir / "output"
        
        input_files = list(input_dir.glob("*.pdf")) if input_dir.exists() else []
        output_files = list(output_dir.glob("*.pdf")) if output_dir.exists() else []
        
        return {
            "status": "healthy", 
            "timestamp": "2025-08-15",
            "port": PORT,
            "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'development'),
            "input_folder": str(input_dir),
            "input_files": [f.name for f in input_files],
            "output_folder": str(output_dir),
            "output_files": [f.name for f in output_files]
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # This is for local development only
    # Railway will use uvicorn directly via railway.json
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info") 