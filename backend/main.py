import os
import sys
import logging
import gc
import signal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables to limit threading and prevent resource exhaustion
os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import export utilities
from export_utils import save_dataframe_dual_format, convert_csv_to_xlsx

def ensure_csv_file(input_file_path: str, output_file_path: str = None) -> str:
    """
    Convert XLSX/XLS file to CSV if needed, or return the CSV path as-is.
    
    Args:
        input_file_path: Path to input file (CSV, XLSX, or XLS)
        output_file_path: Optional output path. If not provided, uses input path with .csv extension
    
    Returns:
        str: Path to CSV file (either original or converted)
    """
    import pandas as pd
    
    input_path = Path(input_file_path)
    
    # If already CSV, return as-is
    if input_path.suffix.lower() == '.csv':
        return str(input_path)
    
    # If XLSX or XLS, convert to CSV
    if input_path.suffix.lower() in ('.xlsx', '.xls'):
        if output_file_path is None:
            output_file_path = str(input_path.with_suffix('.csv'))
        
        try:
            # Read Excel file
            df = pd.read_excel(input_file_path, dtype=str)
            # Save as CSV
            df.to_csv(output_file_path, index=False)
            logger.info(f"Converted {input_file_path} to {output_file_path}")
            return output_file_path
        except Exception as e:
            raise Exception(f"Failed to convert Excel file to CSV: {str(e)}")
    
    # Unknown format
    raise Exception(f"Unsupported file format: {input_path.suffix}")

def kill_process_tree(process):
    """Kill a process and all its children"""
    try:
        import psutil
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                child.kill()
                logger.info(f"Killed child process {child.pid}")
            except psutil.NoSuchProcess:
                pass
        
        # Kill parent
        process.kill()
        process.wait()
        logger.info(f"Killed parent process {process.pid}")
    except Exception as e:
        logger.error(f"Error killing process tree: {e}")
        # Fallback to simple kill
        try:
            process.kill()
            process.wait()
        except:
            pass

# Try to import required modules
try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Body
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
    from typing import List, Optional
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
    logger.info(f"🔑 OCR.space API Key: {'Set' if os.environ.get('OCRSPACE_API_KEY') else 'Not set'}")
    logger.info(f"🔑 OpenRouter API Key: {'Set' if os.environ.get('OPENROUTER_API_KEY') else 'Not set'}")
    logger.info(f"🔑 OpenAI API Key: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not set'} (fallback for OpenRouter)")
    logger.info(f"🌍 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'development')}")
    logger.info(f"🏗️ Railway Project: {os.environ.get('RAILWAY_PROJECT_ID', 'unknown')}")
    logger.info(f"🚂 Railway Service: {os.environ.get('RAILWAY_SERVICE_NAME', 'unknown')}")
    logger.info(f"🔧 PORT env var: {os.environ.get('PORT', 'not set')}")
    logger.info(f"🔧 GLOBAL PORT: {PORT}")
    
    # Initialize database schema
    try:
        from db_utils import init_database
        init_database()
        logger.info("✅ Database schema initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")

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
        self.result_file_xlsx = None  # Store XLSX version
        self.error = None
        self.metadata = {}  # Store additional data like reasoning

def process_pdfs_background(job_id: str, zip_path: str, excel_path: str, n_pages: int, excel_filename: str, model: str = "gemini-2.5-flash", worktracker_group: str = None, worktracker_batch: str = None, extract_csn: bool = False):
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
        # Limit OpenBLAS threads to prevent resource exhaustion
        env['OPENBLAS_NUM_THREADS'] = '12'
        env['OMP_NUM_THREADS'] = '12'
        env['MKL_NUM_THREADS'] = '12'
        
        # Run the processing script
        script_path = Path(__file__).parent / "current" / "2-extract_info.py"
        
        job.message = f"Processing PDFs (extracting first {n_pages} pages per patient)..."
        job.progress = 30
        
        # Run the script with subprocess (with timeout)
        process = None
        try:
            cmd = [
                sys.executable, str(script_path),
                str(temp_dir / "input"),
                str(excel_dest),
                str(n_pages),  # n_pages parameter
                "7",  # max_workers
                model  # model parameter
            ]
            
            # Add worktracker fields if provided
            if worktracker_group:
                cmd.append(worktracker_group)
            else:
                cmd.append("")  # Empty string if not provided
                
            if worktracker_batch:
                cmd.append(worktracker_batch)
            else:
                cmd.append("")  # Empty string if not provided
            
            # Add extract_csn flag
            if extract_csn:
                cmd.append("true")
            else:
                cmd.append("false")
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=temp_dir, env=env)
            
            # Wait for process with timeout (30 minutes max)
            stdout, stderr = process.communicate(timeout=1800)
            
            # Log stdout and stderr for debugging
            logger.info(f"Script stdout: {stdout}")
            if stderr:
                logger.info(f"Script stderr: {stderr}")
            
            if process.returncode != 0:
                raise Exception(f"Processing failed: {stderr}")
        except subprocess.TimeoutExpired:
            # Kill the process and all its children if it times out
            if process:
                logger.warning(f"Process timed out, killing process tree for PID {process.pid}")
                kill_process_tree(process)
                gc.collect()  # Force cleanup after killing process
            raise Exception("Processing timed out after 30 minutes")
        except Exception as e:
            # Ensure process is terminated on any error
            if process and process.poll() is None:
                logger.warning(f"Process failed, killing process tree for PID {process.pid}")
                kill_process_tree(process)
                gc.collect()  # Force cleanup after killing process
            raise e
        
        job.message = "Processing complete, preparing results..."
        job.progress = 80
        
        # Find the generated CSV file
        extracted_dir = temp_dir / "extracted"
        csv_files = list(extracted_dir.glob("*.csv"))
        
        if not csv_files:
            # Provide detailed diagnostic information
            input_dir = temp_dir / "input"
            input_exists = input_dir.exists()
            input_files = list(input_dir.rglob("*")) if input_exists else []
            extracted_exists = extracted_dir.exists()
            
            error_msg = f"No CSV file generated. Diagnostics:\n"
            error_msg += f"- Input dir exists: {input_exists}\n"
            error_msg += f"- Input files: {[f.name for f in input_files[:10]]}\n"  # Show first 10 files
            error_msg += f"- Extracted dir exists: {extracted_exists}\n"
            error_msg += f"- Script stdout (last 500 chars): {stdout[-500:] if stdout else 'empty'}\n"
            error_msg += f"- Script stderr (last 500 chars): {stderr[-500:] if stderr else 'empty'}"
            logger.error(error_msg)
            raise Exception("No CSV file generated - check logs for details")
        
        # Copy the CSV to a permanent location and create XLSX version
        result_base = Path(f"/tmp/results/{job_id}")
        result_base.parent.mkdir(exist_ok=True)
        
        # Copy CSV
        result_file = result_base.with_suffix('.csv')
        shutil.copy2(csv_files[0], result_file)
        
        # Create XLSX version
        try:
            import pandas as pd
            df = pd.read_csv(result_file, dtype=str)
            result_file_xlsx = result_base.with_suffix('.xlsx')
            df.to_excel(result_file_xlsx, index=False, engine='openpyxl')
            job.result_file_xlsx = str(result_file_xlsx)
            logger.info(f"Created XLSX version: {result_file_xlsx}")
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(result_file)
        job.status = "completed"
        job.progress = 100
        job.message = "Processing completed successfully!"
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        os.unlink(zip_path)
        os.unlink(excel_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after processing")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Processing failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

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
        # Limit OpenBLAS threads to prevent resource exhaustion
        env['OPENBLAS_NUM_THREADS'] = '12'
        env['OMP_NUM_THREADS'] = '12'
        env['MKL_NUM_THREADS'] = '12'
        
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
        
        process = None
        try:
            process = subprocess.Popen([
                sys.executable, str(temp_script_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=current_dir, env=env)
            
            # Wait for process with timeout (20 minutes max for splitting)
            stdout, stderr = process.communicate(timeout=1200)
            
            logger.info(f"Script stdout: {stdout}")
            logger.info(f"Script stderr: {stderr}")
            
            if process.returncode != 0:
                raise Exception(f"Splitting failed: {stderr}")
        except subprocess.TimeoutExpired:
            # Kill the process and all its children if it times out
            if process:
                logger.warning(f"PDF splitting timed out, killing process tree for PID {process.pid}")
                kill_process_tree(process)
                gc.collect()  # Force cleanup after killing process
            raise Exception("PDF splitting timed out after 20 minutes")
        except Exception as e:
            # Ensure process is terminated on any error
            if process and process.poll() is None:
                logger.warning(f"PDF splitting failed, killing process tree for PID {process.pid}")
                kill_process_tree(process)
                gc.collect()  # Force cleanup after killing process
            raise e
        
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
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after PDF splitting")
        
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
        
        # Clean up memory even on failure
        gc.collect()

def split_pdf_gemini_background(job_id: str, pdf_path: str, filter_string: str, batch_size: int = 5, model: str = "gemini-flash-latest", max_workers: int = 12):
    """Background task to split PDF using new splitting method"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting PDF splitting..."
        job.progress = 10
        
        # Use existing output folder
        current_dir = Path(__file__).parent / "current"
        output_dir = current_dir / "output"
        
        # Create output folder if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Clear output folder first
        if output_dir.exists():
            for file in output_dir.glob("*.pdf"):
                file.unlink()
        
        job.message = f"Analyzing PDF with Gemini (filter: '{filter_string}')..."
        job.progress = 30
        
        # Set up environment for the script
        env = os.environ.copy()
        env['PYTHONPATH'] = str(current_dir)
        # Copy API keys
        if 'GOOGLE_API_KEY' in os.environ:
            env['GOOGLE_API_KEY'] = os.environ['GOOGLE_API_KEY']
        
        # Limit OpenBLAS threads to prevent resource exhaustion
        env['OPENBLAS_NUM_THREADS'] = '12'
        env['OMP_NUM_THREADS'] = '12'
        env['MKL_NUM_THREADS'] = '12'
        
        # Path to Gemini splitting script
        script_path = current_dir / "1-split_pdf_gemini.py"
        
        if not script_path.exists():
            raise Exception(f"Gemini splitting script not found: {script_path}")
        
        job.message = f"Analyzing PDF pages..."
        job.progress = 40
        
        # Import and call the function directly
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("split_pdf_gemini", script_path)
        split_module = importlib.util.module_from_spec(spec)
        sys.modules["split_pdf_gemini"] = split_module
        spec.loader.exec_module(split_module)
        split_pdf_with_gemini = split_module.split_pdf_with_gemini
        
        logger.info(f"Running Gemini split function with filter: {filter_string}")
        logger.info(f"Input PDF: {pdf_path}")
        logger.info(f"Output folder: {output_dir}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Model: {model}")
        logger.info(f"Max workers: {max_workers}")
        
        job.message = "Gemini processing PDF pages..."
        job.progress = 60
        
        # Call the function directly
        filter_strings = [filter_string]
        created_count = split_pdf_with_gemini(
            pdf_path, str(output_dir), filter_strings, batch_size, model, max_workers
        )
        
        if created_count is None:
            raise Exception("Gemini splitting failed - no sections created")
        
        job.message = "Creating ZIP archive of split PDFs..."
        job.progress = 85
        
        # Create ZIP file with all split PDFs
        zip_path = Path(f"/tmp/results/{job_id}_split_pdfs_gemini.zip")
        zip_path.parent.mkdir(exist_ok=True)
        
        # Find all PDF files created by the script
        pdf_files = list(output_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise Exception("No PDF files were created by the splitting script. This could mean no matching pages were found.")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for pdf_file in pdf_files:
                zipf.write(pdf_file, pdf_file.name)
        
        job.result_file = str(zip_path)
        job.status = "completed"
        job.progress = 100
        job.message = f"Splitting completed successfully! Created {len(pdf_files)} sections."
        
        # Clean up uploaded file
        os.unlink(pdf_path)
        logger.info(f"Cleaned up input file: {pdf_path}")
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after Gemini PDF splitting")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"PDF splitting failed: {str(e)}"
        logger.error(f"Split job {job_id} failed: {str(e)}")
        
        # Clean up on error
        try:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
                logger.info(f"Cleaned up input file on error: {pdf_path}")
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
        
        # Clean up memory even on failure
        gc.collect()

def split_pdf_ocrspace_background(job_id: str, pdf_path: str, filter_string: str, max_workers: int = 15):
    """Background task to split PDF using OCR.space API"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting PDF splitting with OCR.space..."
        job.progress = 10
        
        # Use job-specific output folder (prevents conflicts with concurrent users)
        current_dir = Path(__file__).parent / "current"
        output_dir = current_dir / "output" / job_id  # Job-specific folder
        
        # Create output folder if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        job.message = f"Analyzing PDF with OCR.space API (filter: '{filter_string}')..."
        job.progress = 30
        
        # Use job-specific output folder (prevents conflicts with concurrent users)
        output_dir = current_dir / "output" / job_id  # Job-specific folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up environment for the script
        env = os.environ.copy()
        env['PYTHONPATH'] = str(current_dir)
        
        # Limit OpenBLAS threads to prevent resource exhaustion
        env['OPENBLAS_NUM_THREADS'] = '12'
        env['OMP_NUM_THREADS'] = '12'
        env['MKL_NUM_THREADS'] = '12'
        
        # Path to OCR.space splitting script
        script_path = current_dir / "1-split_pdf_ocrspace.py"
        
        if not script_path.exists():
            raise Exception(f"OCR.space splitting script not found: {script_path}")
        
        job.message = f"OCR processing PDF pages..."
        job.progress = 40
        
        # Run the OCR.space splitting script
        # Args: input_pdf, output_folder, filter_strings, max_workers
        logger.info(f"Running OCR.space split script with filter: {filter_string}")
        logger.info(f"Input PDF: {pdf_path}")
        logger.info(f"Output folder: {output_dir} (job-specific)")
        logger.info(f"Max workers: {max_workers}")
        
        process = None
        try:
            process = subprocess.Popen([
                sys.executable, 
                str(script_path),
                str(pdf_path),  # input PDF
                str(output_dir),  # output folder (job-specific)
                filter_string,  # filter string
                str(max_workers)  # parallel workers
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=current_dir, env=env)
            
            job.message = "OCR.space processing PDF pages..."
            job.progress = 60
            
            # Wait for process with timeout (20 minutes max)
            stdout, stderr = process.communicate(timeout=1200)
            
            logger.info(f"OCR.space script stdout: {stdout}")
            logger.info(f"OCR.space script stderr: {stderr}")
            
            if process.returncode != 0:
                raise Exception(f"OCR.space splitting failed: {stderr}")
            
        except subprocess.TimeoutExpired:
            # Kill the process and all its children if it times out
            if process:
                logger.warning(f"OCR.space PDF splitting timed out, killing process tree for PID {process.pid}")
                kill_process_tree(process)
                gc.collect()  # Force cleanup after killing process
            raise Exception("OCR.space PDF splitting timed out after 20 minutes")
        except Exception as e:
            # Ensure process is terminated on any error
            if process and process.poll() is None:
                logger.warning(f"OCR.space PDF splitting failed, killing process tree for PID {process.pid}")
                kill_process_tree(process)
                gc.collect()  # Force cleanup after killing process
            raise e
        
        job.message = "Creating ZIP archive of split PDFs..."
        job.progress = 85
        
        # Create ZIP file with all split PDFs
        zip_path = Path(f"/tmp/results/{job_id}_split_pdfs_ocrspace.zip")
        zip_path.parent.mkdir(exist_ok=True)
        
        # Find all PDF files created by the script
        pdf_files = list(output_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise Exception("No PDF files were created by the splitting script. This could mean no matching pages were found.")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for pdf_file in pdf_files:
                zipf.write(pdf_file, pdf_file.name)
        
        job.result_file = str(zip_path)
        job.status = "completed"
        job.progress = 100
        job.message = f"Splitting completed successfully! Created {len(pdf_files)} sections."
        
        # Clean up uploaded file
        os.unlink(pdf_path)
        logger.info(f"Cleaned up input file: {pdf_path}")
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after OCR.space PDF splitting")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"PDF splitting failed: {str(e)}"
        logger.error(f"Split job {job_id} failed: {str(e)}")
        
        # Clean up on error
        try:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
                logger.info(f"Cleaned up input file on error: {pdf_path}")
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
        
        # Clean up memory even on failure
        gc.collect()

def predict_cpt_background(job_id: str, csv_path: str, client: str = "uni"):
    """Background task to predict CPT codes from CSV procedures"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting CPT code prediction..."
        job.progress = 10
        
        # Import required modules for CPT prediction
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from google import genai
        from google.genai import types
        
        # Setup clients
        # Use JSON string from environment variable (Option 1)
        import json
        import tempfile
        credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if credentials_json:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(credentials_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            
            genai_client = genai.Client(
                vertexai=True,
                project="835764687231",
                location="us-central1",
            )
        else:
            # Fallback to API key method
            genai_client = genai.Client(vertexai=True, api_key="AQ.Ab8RN6LnO1TE5YbcCw1PLVGe2qxhL7TuOVtVm3GnhXndEM0nsw")
        
        fallback_client = genai.Client(vertexai=True, api_key="AQ.Ab8RN6LnO1TE5YbcCw1PLVGe2qxhL7TuOVtVm3GnhXndEM0nsw")
        
        # Client model mapping
        client_models = {
            "uni": "projects/835764687231/locations/us-central1/endpoints/4576355411292061696",
            "sio-stl": "projects/835764687231/locations/us-central1/endpoints/6830407024790994944",
            "gap-fin": "projects/835764687231/locations/us-central1/endpoints/8077904121572622336",
            "apo-utp": "projects/835764687231/locations/us-central1/endpoints/8135325016821596160"
        }
        
        custom_model = client_models.get(client, client_models["uni"])
        fallback_model = "gemini-3-pro-preview"
        
        logger.info(f"Using client model for {client}: {custom_model}")
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=1,
            seed=0,
            max_output_tokens=50,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        
        def format_asa_code(code):
            """Format ASA code with leading zeros based on length"""
            if not code or not isinstance(code, str):
                return code
            
            # Clean the code - remove any non-numeric characters except leading zeros
            cleaned_code = code.strip()
            
            # Extract only numeric characters
            numeric_code = ''.join(filter(str.isdigit, cleaned_code))
            
            if not numeric_code:
                return code  # Return original if no numbers found
            
            # Apply formatting rules
            if len(numeric_code) == 4:
                # 4 numbers: add 1 leading zero
                return f"0{numeric_code}"
            elif len(numeric_code) == 3:
                # 3 numbers: add 2 leading zeros
                return f"00{numeric_code}"
            else:
                # Return as is for other lengths
                return numeric_code
        
        def apply_colonoscopy_correction(row, predicted_code, insurances_df):
            """
            Apply colonoscopy-specific correction rules based on procedure type,
            insurance plan, and polyp findings.
            
            Args:
                row: DataFrame row containing procedure information
                predicted_code: The AI-predicted CPT code
                insurances_df: DataFrame containing insurance information
            
            Returns:
                Corrected CPT code or original predicted code if no correction applies
            """
            try:
                # Get the required fields from the row
                is_colonoscopy = str(row.get('is_colonoscopy', '')).strip().upper() == 'TRUE'
                colonoscopy_is_screening = str(row.get('colonoscopy_is_screening', '')).strip().upper() == 'TRUE'
                is_upper_endonoscopy = str(row.get('is_upper_endonoscopy', '')).strip().upper() == 'TRUE'
                polyps_found = str(row.get('Polyps found', '')).strip().upper() == 'FOUND'
                primary_mednet_code = str(row.get('Primary Mednet Code', '')).strip()
                
                # Priority rule: if both upper endoscopy and colonoscopy, always return 00813
                if is_upper_endonoscopy and is_colonoscopy:
                    logger.info(f"Colonoscopy correction: Both upper endoscopy and colonoscopy detected -> 00813")
                    return "00813"
                
                # Only proceed if it's a colonoscopy
                if not is_colonoscopy:
                    return predicted_code
                
                # Check if insurance is Medicare
                is_medicare = False
                if primary_mednet_code and not insurances_df.empty:
                    # Find the insurance plan by MedNet Code
                    insurance_match = insurances_df[insurances_df['MedNet Code'].astype(str).str.strip() == primary_mednet_code]
                    if not insurance_match.empty:
                        insurance_plan = str(insurance_match.iloc[0].get('Insurance Plan', '')).strip()
                        if 'Medicare' in insurance_plan or 'MEDICARE' in insurance_plan or 'medicare' in insurance_plan:
                            is_medicare = True
                            logger.info(f"Colonoscopy correction: Medicare insurance detected ({insurance_plan})")
                
                # Apply correction rules
                if is_medicare:
                    # MEDICARE rules
                    if colonoscopy_is_screening and not polyps_found:
                        logger.info(f"Colonoscopy correction: Medicare + screening + no polyps -> 00812")
                        return "00812"
                    elif colonoscopy_is_screening and polyps_found:
                        logger.info(f"Colonoscopy correction: Medicare + screening + polyps found -> 00811")
                        return "00811"
                    else:  # not screening (polyps don't matter)
                        logger.info(f"Colonoscopy correction: Medicare + not screening -> 00811")
                        return "00811"
                else:
                    # NOT MEDICARE rules
                    if colonoscopy_is_screening and not polyps_found:
                        logger.info(f"Colonoscopy correction: Non-Medicare + screening + no polyps -> 00812")
                        return "00812"
                    elif colonoscopy_is_screening and polyps_found:
                        logger.info(f"Colonoscopy correction: Non-Medicare + screening + polyps found -> 00812")
                        return "00812"
                    else:  # not screening (polyps don't matter)
                        logger.info(f"Colonoscopy correction: Non-Medicare + not screening -> 00811")
                        return "00811"
                
            except Exception as e:
                logger.warning(f"Colonoscopy correction failed: {str(e)}, returning original prediction")
                return predicted_code

        def get_prediction_and_review(procedure, retries=5):
            """Two-stage prediction: 1) Custom model predicts, 2) Gemini Flash reviews"""
            
            # Check if input is invalid (None, empty, or less than 3 characters)
            procedure_str = str(procedure) if procedure is not None else ""
            procedure_str = procedure_str.strip()
            
            if not procedure_str or len(procedure_str) < 3:
                # Skip custom model and go directly to fallback for invalid input
                initial_prediction = None
                model_source = None
                failure_reason = f"Input too short or empty (length: {len(procedure_str)})"
            else:
                # Stage 1: Get initial prediction from custom model
                initial_prediction = None
                model_source = None
                failure_reason = None
            
            prompt = f'For this procedure: "{procedure_str}" give me the most appropriate anesthesia CPT code'
            last_error = "Unknown error"

            # Try custom model first (only if input is valid)
            if initial_prediction is None and procedure_str and len(procedure_str) >= 3:
                for attempt in range(retries):
                    try:
                        response = genai_client.models.generate_content(
                            model=custom_model,
                            contents=[types.Content(role="user", parts=[{"text": prompt}])],
                            config=generate_content_config
                        )
                        result = response.text
                        if result and result.strip().startswith("0"):
                            initial_prediction = format_asa_code(result.strip())
                            model_source = "base_model"
                            break
                        else:
                            failure_reason = f"Base model returned invalid format: '{result.strip()}'"
                    except Exception as e:
                        last_error = str(e)
                        failure_reason = f"Base model error (attempt {attempt + 1}/{retries}): {str(e)}"

            # Fallback to production model if custom model failed
            if not initial_prediction:
                try:
                    # Add web search to fallback model
                    fallback_tools = [
                        types.Tool(googleSearch=types.GoogleSearch()),
                    ]
                    fallback_config = types.GenerateContentConfig(
                        tools=fallback_tools
                    )
                    
                    response = fallback_client.models.generate_content(
                        model=fallback_model,
                        contents=[types.Content(role="user", parts=[{"text": prompt + "Only answer with the code, absolutely nothing else, no other text."}])],
                        config=fallback_config
                    )
                    fallback_result = response.text
                    if fallback_result:
                        initial_prediction = format_asa_code(fallback_result.strip())
                        model_source = "fallback"
                    else:
                        return f"Prediction failed: empty response", "error", f"Fallback model returned empty response. Base model failure: {failure_reason}"
                except Exception as e:
                    return f"Prediction failed: {str(e)}", "error", f"Fallback model error: {str(e)}. Base model failure: {failure_reason}"

            # Stage 2: Review with custom instructions (if available)
            try:
                # Load custom instructions
                instructions_file = Path("data/cpt_instructions.json")
                custom_instructions = ""
                if instructions_file.exists():
                    with open(instructions_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        custom_instructions = data.get("instructions", "").strip()

                # If no custom instructions, return initial prediction with model source
                if not custom_instructions:
                    return initial_prediction, model_source, failure_reason

                # Create review prompt
                review_prompt = f"""You are a medical coding reviewer. Your job is to REVIEW and potentially CORRECT a CPT code prediction, but ONLY if the custom medical coder instructions specifically apply to this case.

PROCEDURE DESCRIPTION: "{procedure}"
PREDICTED CPT CODE: "{initial_prediction}"

CUSTOM MEDICAL CODER INSTRUCTIONS:
{custom_instructions}

IMPORTANT RULES:
1. Your job is NOT to predict a new code - it's to REVIEW and potentially CORRECT the existing prediction
2. ONLY change the code if the custom instructions specifically mention this type of procedure OR this specific code
3. If the custom instructions don't apply to this specific procedure or code, return the original predicted code unchanged
4. If you do make a correction, return only the corrected code with proper formatting (leading zeros)
5. Return ONLY the CPT code, nothing else

What is your final code decision?


answer ONLY with the code, nothing else"""

                # Use Gemini Flash for review with thinking enabled
                review_config = types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more consistent reviews
                    top_p=0.9,
                    max_output_tokens=50,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
                
                review_response = fallback_client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=[types.Content(role="user", parts=[{"text": review_prompt}])],
                    config=review_config
                )
                
                reviewed_result = review_response.text.strip()
                
                # Format the reviewed result
                if reviewed_result and reviewed_result.replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "") == "":
                    # It's a numeric code, format it
                    final_result = format_asa_code(reviewed_result)
                    # If review changed the code, mark as reviewed
                    if final_result != initial_prediction:
                        return final_result, "reviewed", failure_reason
                    else:
                        return final_result, model_source, failure_reason
                else:
                    # Not a clean numeric code, return original prediction
                    return initial_prediction, model_source, failure_reason

            except Exception as e:
                logger.warning(f"Review stage failed: {str(e)}, returning initial prediction")
                return initial_prediction, model_source, failure_reason
        
        job.message = "Reading CSV file..."
        job.progress = 20
        
        # Read CSV file with dtype=str to preserve leading zeros in MedNet codes
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', dtype=str)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='latin-1', dtype=str)
            except Exception as e:
                raise Exception(f"Could not read CSV with utf-8 or latin-1 encoding: {e}")

        if "Procedure Description" not in df.columns:
            raise Exception("CSV file missing 'Procedure Description' column")
        
        # Load insurances.csv for colonoscopy correction
        job.message = "Loading insurance data for colonoscopy corrections..."
        insurances_df = pd.DataFrame()
        try:
            insurances_path = Path(__file__).parent / "insurances.csv"
            if insurances_path.exists():
                insurances_df = pd.read_csv(insurances_path, dtype=str)
                logger.info(f"Loaded {len(insurances_df)} insurance records for colonoscopy correction")
            else:
                logger.warning(f"insurances.csv not found at {insurances_path}, colonoscopy corrections may not work properly")
        except Exception as e:
            logger.warning(f"Failed to load insurances.csv: {str(e)}, colonoscopy corrections may not work properly")

        job.message = f"Processing {len(df)} procedures..."
        job.progress = 30
        
        # Process predictions with threading
        predictions = [None] * len(df)
        model_sources = [None] * len(df)
        failure_reasons = [None] * len(df)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(get_prediction_and_review, proc): i for i, proc in enumerate(df["Procedure Description"])}
            
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if isinstance(result, tuple) and len(result) == 3:
                    predictions[idx], model_sources[idx], failure_reasons[idx] = result
                elif isinstance(result, tuple) and len(result) == 2:
                    # Handle old format for backward compatibility
                    predictions[idx], model_sources[idx] = result
                    failure_reasons[idx] = None
                else:
                    # Handle single return value
                    predictions[idx] = result
                    model_sources[idx] = "unknown"
                    failure_reasons[idx] = None
                completed += 1
                job.progress = 30 + int((completed / len(df)) * 50)
                job.message = f"Processed {completed}/{len(df)} procedures..."

        job.message = "Applying colonoscopy corrections..."
        job.progress = 85
        
        # Apply colonoscopy corrections to predictions (only for uni client)
        corrected_predictions = []
        correction_applied = []
        for idx, row in df.iterrows():
            original_prediction = predictions[idx]
            # Only apply colonoscopy correction for uni client
            if client == "uni":
                corrected_code = apply_colonoscopy_correction(row, original_prediction, insurances_df)
            else:
                corrected_code = original_prediction
            corrected_predictions.append(corrected_code)
            # Track if correction was applied
            correction_applied.append("Yes" if corrected_code != original_prediction else "No")
        
        job.message = "Adding predictions to CSV..."
        job.progress = 90
        
        # Insert predictions, model sources, and failure reasons into dataframe
        insert_index = df.columns.get_loc("Procedure Description") + 1
        df.insert(insert_index, "ASA Code", corrected_predictions)
        df.insert(insert_index + 1, "Procedure Code", corrected_predictions)
        df.insert(insert_index + 2, "Model Source", model_sources)
        df.insert(insert_index + 3, "Colonoscopy Correction Applied", correction_applied)
        df.insert(insert_index + 4, "Base Model Failure Reason", failure_reasons)
        
        # Save result in both formats
        result_base = Path(f"/tmp/results/{job_id}_with_codes")
        result_base.parent.mkdir(exist_ok=True)
        
        result_csv_path, result_xlsx_path = save_dataframe_dual_format(df, result_base)
        
        job.result_file = result_csv_path
        job.result_file_xlsx = result_xlsx_path
        job.status = "completed"
        job.progress = 100
        job.message = f"CPT prediction completed! Processed {len(df)} procedures."
        
        # Clean up input file
        os.unlink(csv_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after CPT prediction")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"CPT prediction failed: {str(e)}"
        logger.error(f"CPT prediction job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

def predict_cpt_custom_background(job_id: str, csv_path: str, confidence_threshold: float = 0.5):
    """Background task to predict CPT codes using custom TAN-ESC model"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting custom model CPT prediction..."
        job.progress = 10
        
        # Import the custom prediction function
        import sys
        custom_coding_path = Path(__file__).parent / "custom-coding"
        sys.path.insert(0, str(custom_coding_path))
        
        from predict import predict_codes_api
        
        job.message = f"Loading TAN-ESC model and processing {csv_path}..."
        job.progress = 30
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_with_codes")
        result_base.parent.mkdir(exist_ok=True)
        
        # Run the prediction using the custom model
        # Model files should be in backend/custom-coding/ directory
        model_dir = custom_coding_path
        
        job.message = "Making predictions with TAN-ESC model..."
        job.progress = 50
        
        # Save to CSV first
        result_file_csv = result_base.with_suffix('.csv')
        success = predict_codes_api(
            input_file=csv_path,
            output_file=str(result_file_csv),
            model_dir=str(model_dir),
            confidence_threshold=confidence_threshold
        )
        
        if not success:
            raise Exception("Custom model prediction failed")
        
        job.message = "Prediction complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not result_file_csv.exists():
            raise Exception("Output file was not created")
        
        # Create XLSX version
        try:
            result_file_xlsx = convert_csv_to_xlsx(result_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = result_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(result_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = f"TAN-ESC prediction completed successfully!"
        
        # Clean up input file
        os.unlink(csv_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after custom model prediction")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Custom model prediction failed: {str(e)}"
        logger.error(f"Custom model prediction job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

def predict_cpt_general_background(job_id: str, csv_path: str, model: str = "gpt5", max_workers: int = 5, custom_instructions: str = None):
    """Background task to predict CPT codes using OpenAI general model"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting OpenAI general model CPT prediction..."
        job.progress = 10
        
        # Import the general prediction function
        import sys
        general_coding_path = Path(__file__).parent / "general-coding"
        sys.path.insert(0, str(general_coding_path))
        
        from predict_general import predict_codes_general_api
        
        job.message = f"Processing with OpenAI {model} model..."
        job.progress = 30
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_with_codes")
        result_base.parent.mkdir(exist_ok=True)
        
        # Progress callback to update job status
        def progress_callback(completed, total, message):
            job.progress = 30 + int((completed / total) * 60)
            job.message = message
        
        # Save to CSV first
        result_file_csv = result_base.with_suffix('.csv')
        success = predict_codes_general_api(
            input_file=csv_path,
            output_file=str(result_file_csv),
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=max_workers,
            progress_callback=progress_callback
        )
        
        if not success:
            raise Exception("OpenAI general model prediction failed")
        
        job.message = "Prediction complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not result_file_csv.exists():
            raise Exception("Output file was not created")
        
        # Create XLSX version
        try:
            result_file_xlsx = convert_csv_to_xlsx(result_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = result_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(result_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = f"OpenAI general model prediction completed successfully!"
        
        # Clean up input file
        os.unlink(csv_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after general model prediction")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"OpenAI general model prediction failed: {str(e)}"
        logger.error(f"General model prediction job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

def predict_cpt_from_pdfs_background(job_id: str, zip_path: str, n_pages: int = 1, model: str = "openai/gpt-5", max_workers: int = 5, custom_instructions: str = None):
    """Background task to predict CPT codes from PDF images using OpenAI vision model"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Extracting PDFs from ZIP archive..."
        job.progress = 10
        
        # Create temporary directory for processing
        temp_dir = Path(f"/tmp/processing_{job_id}")
        temp_dir.mkdir(exist_ok=True)
        
        # Unzip files
        pdfs_dir = temp_dir / "pdfs"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pdfs_dir)
        
        # Log what was extracted for debugging
        extracted_files = list(pdfs_dir.rglob("*"))
        pdf_files_extracted = [f for f in extracted_files if f.is_file() and f.suffix.lower() == '.pdf']
        logger.info(f"Extracted {len(extracted_files)} total items, {len(pdf_files_extracted)} PDF files")
        if pdf_files_extracted:
            logger.info(f"PDF files found: {[f.name for f in pdf_files_extracted]}")
        
        job.message = f"Starting image-based CPT prediction (analyzing {n_pages} page(s) per PDF)..."
        job.progress = 20
        
        # Import the general prediction function
        import sys
        general_coding_path = Path(__file__).parent / "general-coding"
        sys.path.insert(0, str(general_coding_path))
        
        from predict_general import predict_codes_from_pdfs_api
        
        job.message = f"Processing PDFs with OpenRouter {model} vision model..."
        job.progress = 30
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_with_codes")
        result_base.parent.mkdir(exist_ok=True)
        
        # Progress callback to update job status
        def progress_callback(completed, total, message):
            job.progress = 30 + int((completed / total) * 60)
            job.message = message
        
        # Save to CSV first
        result_file_csv = result_base.with_suffix('.csv')
        # Use OPENROUTER_API_KEY if available, fallback to OPENAI_API_KEY
        api_key = os.getenv("OPENROUTER_API_KEY") 
        success = predict_codes_from_pdfs_api(
            pdf_folder=str(temp_dir / "pdfs"),
            output_file=str(result_file_csv),
            n_pages=n_pages,
            model=model,
            api_key=api_key,
            max_workers=max_workers,
            progress_callback=progress_callback,
            custom_instructions=custom_instructions
        )
        
        if not success:
            raise Exception("OpenRouter vision model prediction failed")
        
        job.message = "Prediction complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not result_file_csv.exists():
            raise Exception("Output file was not created")
        
        # Create XLSX version
        try:
            result_file_xlsx = convert_csv_to_xlsx(result_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = result_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(result_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = f"Vision-based CPT prediction completed successfully!"
        
        # Clean up input files
        os.unlink(zip_path)
        shutil.rmtree(temp_dir)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after vision-based prediction")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Vision-based prediction failed: {str(e)}"
        logger.error(f"Vision-based prediction job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

def predict_icd_from_pdfs_background(job_id: str, zip_path: str, n_pages: int = 1, model: str = "openai/gpt-5", max_workers: int = 5, custom_instructions: str = None):
    """Background task to predict ICD codes from PDF images using OpenAI vision model"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Extracting PDFs from ZIP archive..."
        job.progress = 10
        
        # Create temporary directory for processing
        temp_dir = Path(f"/tmp/processing_{job_id}")
        temp_dir.mkdir(exist_ok=True)
        
        # Unzip files
        pdfs_dir = temp_dir / "pdfs"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pdfs_dir)
        
        # Log what was extracted for debugging
        extracted_files = list(pdfs_dir.rglob("*"))
        pdf_files_extracted = [f for f in extracted_files if f.is_file() and f.suffix.lower() == '.pdf']
        logger.info(f"Extracted {len(extracted_files)} total items, {len(pdf_files_extracted)} PDF files")
        if pdf_files_extracted:
            logger.info(f"PDF files found: {[f.name for f in pdf_files_extracted]}")
        
        job.message = f"Starting image-based ICD prediction (analyzing {n_pages} page(s) per PDF)..."
        job.progress = 20
        
        # Import the general prediction function
        import sys
        general_coding_path = Path(__file__).parent / "general-coding"
        sys.path.insert(0, str(general_coding_path))
        
        from predict_general import predict_icd_codes_from_pdfs_api
        
        job.message = f"Processing PDFs with OpenRouter {model} vision model..."
        job.progress = 30
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_with_icd_codes")
        result_base.parent.mkdir(exist_ok=True)
        
        # Progress callback to update job status
        def progress_callback(completed, total, message):
            job.progress = 30 + int((completed / total) * 60)
            job.message = message
        
        # Save to CSV first
        result_file_csv = result_base.with_suffix('.csv')
        # Use OPENROUTER_API_KEY if available, fallback to OPENAI_API_KEY
        api_key = os.getenv("OPENROUTER_API_KEY") 
        success = predict_icd_codes_from_pdfs_api(
            pdf_folder=str(temp_dir / "pdfs"),
            output_file=str(result_file_csv),
            n_pages=n_pages,
            model=model,
            api_key=api_key,
            max_workers=max_workers,
            progress_callback=progress_callback,
            custom_instructions=custom_instructions
        )
        
        if not success:
            raise Exception("OpenRouter vision model ICD prediction failed")
        
        job.message = "Prediction complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not result_file_csv.exists():
            raise Exception("Output file was not created")
        
        # Create XLSX version
        try:
            result_file_xlsx = convert_csv_to_xlsx(result_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = result_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(result_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = f"Vision-based ICD prediction completed successfully!"
        
        # Clean up input files
        os.unlink(zip_path)
        shutil.rmtree(temp_dir)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after vision-based ICD prediction")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Vision-based ICD prediction failed: {str(e)}"
        logger.error(f"Vision-based ICD prediction job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    excel_file: UploadFile = File(None),
    template_id: int = Form(None),
    n_pages: int = Form(..., ge=1, le=50),  # Validate page count between 1-50
    model: str = Form(default="gemini-2.5-flash"),  # Model parameter with default
    worktracker_group: str = Form(None),  # Optional worktracker group field
    worktracker_batch: str = Form(None),  # Optional worktracker batch field
    extract_csn: str = Form(None)  # Optional extract CSN flag
):
    """Upload ZIP file and either Excel instructions file or template ID"""
    
    try:
        # Validate that either excel_file or template_id is provided
        if not excel_file and not template_id:
            raise HTTPException(status_code=400, detail="Either excel_file or template_id must be provided")
        
        if excel_file and template_id:
            raise HTTPException(status_code=400, detail="Provide either excel_file or template_id, not both")
        
        excel_filename = None
        if excel_file:
            excel_filename = excel_file.filename
            logger.info(f"Received upload request - zip: {zip_file.filename}, excel: {excel_file.filename}, pages: {n_pages}, model: {model}")
        else:
            logger.info(f"Received upload request - zip: {zip_file.filename}, template_id: {template_id}, pages: {n_pages}, model: {model}")
        
        # Validate file types
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="First file must be a ZIP archive")
        
        if excel_file and not excel_file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Excel file must be .xlsx or .xls")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created job {job_id}")
        
        # Save ZIP file
        zip_path = f"/tmp/{job_id}_archive.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        
        # Handle Excel file or template
        excel_path = f"/tmp/{job_id}_instructions.xlsx"
        
        if excel_file:
            # Save uploaded Excel file
            with open(excel_path, "wb") as f:
                shutil.copyfileobj(excel_file.file, f)
            logger.info(f"Files saved - zip: {zip_path}, excel: {excel_path}")
        else:
            # Fetch template from database and create Excel file
            from db_utils import get_template
            template = get_template(template_id=template_id)
            
            if not template:
                raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
            
            # Extract field definitions from template
            fields = template['template_data'].get('fields', [])
            
            if not fields:
                raise HTTPException(status_code=400, detail="Template has no field definitions")
            
            # Create DataFrame in the expected format
            data = {}
            for field in fields:
                data[field['name']] = [
                    field.get('description', ''),
                    field.get('location', ''),
                    field.get('output_format', ''),
                    'YES' if field.get('priority', False) else 'NO'
                ]
            
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_excel(excel_path, index=False, engine='openpyxl')
            
            excel_filename = f"{template['name']}.xlsx"
            logger.info(f"Created Excel from template - zip: {zip_path}, template: {template['name']}, excel: {excel_path}")
        
        # Parse extract_csn flag
        extract_csn_flag = extract_csn and extract_csn.lower() == "true"
        
        # Start background processing
        background_tasks.add_task(
            process_pdfs_background, 
            job_id, 
            zip_path, 
            excel_path, 
            n_pages, 
            excel_filename, 
            model,
            worktracker_group,
            worktracker_batch,
            extract_csn_flag
        )
        
        logger.info(f"Background task started for job {job_id}")
        
        return {"job_id": job_id, "message": "Files uploaded and processing started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    response = {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error
    }
    # Include metadata if present (e.g., reasoning for PDF splitting)
    if hasattr(job, 'metadata') and job.metadata:
        response["metadata"] = job.metadata
    return response

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up a completed job and its files"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    # Clean up CSV result file if it exists
    if job.result_file and os.path.exists(job.result_file):
        try:
            os.unlink(job.result_file)
            logger.info(f"Cleaned up CSV result file: {job.result_file}")
        except Exception as e:
            logger.error(f"Failed to clean up CSV result file: {e}")
    
    # Clean up XLSX result file if it exists
    if job.result_file_xlsx and os.path.exists(job.result_file_xlsx):
        try:
            os.unlink(job.result_file_xlsx)
            logger.info(f"Cleaned up XLSX result file: {job.result_file_xlsx}")
        except Exception as e:
            logger.error(f"Failed to clean up XLSX result file: {e}")
    
    # Remove job from memory
    del job_status[job_id]
    logger.info(f"Cleaned up job {job_id} from memory")
    
    return {"message": f"Job {job_id} cleaned up successfully"}

@app.delete("/cleanup-all")
async def cleanup_all_jobs():
    """Clean up all completed jobs and their files"""
    cleaned_count = 0
    total_size_freed = 0
    
    for job_id, job in list(job_status.items()):
        if job.status in ["completed", "failed"]:
            # Clean up CSV result file if it exists
            if job.result_file and os.path.exists(job.result_file):
                try:
                    file_size = os.path.getsize(job.result_file)
                    os.unlink(job.result_file)
                    total_size_freed += file_size
                    logger.info(f"Cleaned up CSV result file: {job.result_file}")
                except Exception as e:
                    logger.error(f"Failed to clean up CSV result file: {e}")
            
            # Clean up XLSX result file if it exists
            if job.result_file_xlsx and os.path.exists(job.result_file_xlsx):
                try:
                    file_size = os.path.getsize(job.result_file_xlsx)
                    os.unlink(job.result_file_xlsx)
                    total_size_freed += file_size
                    logger.info(f"Cleaned up XLSX result file: {job.result_file_xlsx}")
                except Exception as e:
                    logger.error(f"Failed to clean up XLSX result file: {e}")
            
            # Remove job from memory
            del job_status[job_id]
            cleaned_count += 1
    
    logger.info(f"Cleaned up {cleaned_count} jobs, freed {total_size_freed} bytes")
    return {
        "message": f"Cleaned up {cleaned_count} jobs",
        "jobs_cleaned": cleaned_count,
        "bytes_freed": total_size_freed
    }

@app.post("/save-instructions")
async def save_instructions(request: dict):
    """Save custom coding instructions to JSON file"""
    try:
        instructions = request.get("instructions", "")
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save instructions to JSON file
        instructions_file = data_dir / "cpt_instructions.json"
        with open(instructions_file, "w", encoding="utf-8") as f:
            json.dump({"instructions": instructions}, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved custom instructions: {len(instructions)} characters")
        return {"message": "Instructions saved successfully"}
        
    except Exception as e:
        logger.error(f"Failed to save instructions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save instructions: {str(e)}")

@app.get("/load-instructions")
async def load_instructions():
    """Load custom coding instructions from JSON file"""
    try:
        instructions_file = Path("data/cpt_instructions.json")
        
        if not instructions_file.exists():
            return {"instructions": ""}
        
        with open(instructions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instructions = data.get("instructions", "")
        logger.info(f"Loaded custom instructions: {len(instructions)} characters")
        return {"instructions": instructions}
        
    except Exception as e:
        logger.error(f"Failed to load instructions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load instructions: {str(e)}")

@app.post("/predict-cpt")
async def predict_cpt(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    client: str = Form(default="uni")
):
    """Upload a CSV or XLSX file to predict CPT codes for procedures"""
    
    try:
        logger.info(f"Received CPT prediction request - file: {csv_file.filename}, client: {client}")
        
        # Validate file type
        if not csv_file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created CPT prediction job {job_id}")
        
        # Save uploaded file
        input_path = f"/tmp/{job_id}_input{Path(csv_file.filename).suffix}"
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"File saved - path: {input_path}")
        
        # Convert to CSV if needed
        csv_path = ensure_csv_file(input_path, f"/tmp/{job_id}_input.csv")
        
        # Clean up original file if it was converted
        if csv_path != input_path and os.path.exists(input_path):
            os.unlink(input_path)
        
        logger.info(f"CSV ready - path: {csv_path}")
        
        # Start background processing
        background_tasks.add_task(predict_cpt_background, job_id, csv_path, client)
        
        logger.info(f"Background CPT prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": "File uploaded and CPT prediction started"}
        
    except Exception as e:
        logger.error(f"CPT prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/predict-cpt-custom")
async def predict_cpt_custom(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5)
):
    """Upload a CSV or XLSX file to predict CPT codes using custom TAN-ESC model"""
    
    try:
        logger.info(f"Received custom model CPT prediction request - file: {csv_file.filename}, threshold: {confidence_threshold}")
        
        # Validate file type
        if not csv_file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created custom model CPT prediction job {job_id}")
        
        # Save uploaded file
        input_path = f"/tmp/{job_id}_custom_input{Path(csv_file.filename).suffix}"
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"File saved - path: {input_path}")
        
        # Convert to CSV if needed
        csv_path = ensure_csv_file(input_path, f"/tmp/{job_id}_custom_input.csv")
        
        # Clean up original file if it was converted
        if csv_path != input_path and os.path.exists(input_path):
            os.unlink(input_path)
        
        logger.info(f"CSV ready - path: {csv_path}")
        
        # Start background processing with custom model
        background_tasks.add_task(predict_cpt_custom_background, job_id, csv_path, confidence_threshold)
        
        logger.info(f"Background custom model CPT prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": "File uploaded and TAN-ESC prediction started"}
        
    except Exception as e:
        logger.error(f"Custom model CPT prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/predict-cpt-general")
async def predict_cpt_general(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    model: str = Form(default="gpt5"),
    max_workers: int = Form(default=5),
    custom_instructions: Optional[str] = Form(default=None),
    instruction_template_id: Optional[int] = Form(default=None)
):
    """Upload a CSV or XLSX file to predict CPT codes using OpenAI general model"""
    
    try:
        logger.info(f"Received general model CPT prediction request - file: {csv_file.filename}, model: {model}, workers: {max_workers}")
        
        # Fetch instruction template if provided
        if instruction_template_id:
            from db_utils import get_prediction_instruction
            template = get_prediction_instruction(instruction_id=instruction_template_id)
            if template:
                custom_instructions = template['instructions_text']
                logger.info(f"Using instruction template '{template['name']}' for general CPT prediction")
            else:
                logger.warning(f"Instruction template {instruction_template_id} not found, proceeding without custom instructions")
        
        # Validate file type
        if not csv_file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created general model CPT prediction job {job_id}")
        
        # Save uploaded file
        input_path = f"/tmp/{job_id}_general_input{Path(csv_file.filename).suffix}"
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"File saved - path: {input_path}")
        
        # Convert to CSV if needed
        csv_path = ensure_csv_file(input_path, f"/tmp/{job_id}_general_input.csv")
        
        # Clean up original file if it was converted
        if csv_path != input_path and os.path.exists(input_path):
            os.unlink(input_path)
        
        logger.info(f"CSV ready - path: {csv_path}")
        
        # Start background processing with general model
        background_tasks.add_task(predict_cpt_general_background, job_id, csv_path, model, max_workers, custom_instructions)
        
        logger.info(f"Background general model CPT prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": f"File uploaded and OpenAI {model} prediction started"}
        
    except Exception as e:
        logger.error(f"General model CPT prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/predict-cpt-from-pdfs")
async def predict_cpt_from_pdfs(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    n_pages: int = Form(default=1, ge=1, le=50),
    model: str = Form(default="openai/gpt-5"),
    max_workers: int = Form(default=5),
    custom_instructions: Optional[str] = Form(default=None),
    instruction_template_id: Optional[int] = Form(default=None)
):
    """Upload a ZIP file containing PDFs to predict CPT codes using OpenAI vision model"""
    
    try:
        logger.info(f"Received vision-based CPT prediction request - zip: {zip_file.filename}, pages: {n_pages}, model: {model}, workers: {max_workers}")
        
        # Validate file type
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        # Fetch instruction template if provided
        if instruction_template_id:
            from db_utils import get_prediction_instruction
            template = get_prediction_instruction(instruction_id=instruction_template_id)
            if template:
                custom_instructions = template['instructions_text']
                logger.info(f"Using instruction template '{template['name']}' for CPT prediction")
            else:
                logger.warning(f"Instruction template {instruction_template_id} not found, proceeding without custom instructions")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created vision-based CPT prediction job {job_id}")
        
        # Save uploaded ZIP
        zip_path = f"/tmp/{job_id}_pdfs.zip"
        
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        
        logger.info(f"ZIP saved - path: {zip_path}")
        
        # Start background processing with vision model
        background_tasks.add_task(predict_cpt_from_pdfs_background, job_id, zip_path, n_pages, model, max_workers, custom_instructions)
        
        logger.info(f"Background vision-based CPT prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": f"ZIP uploaded and OpenAI {model} vision prediction started"}
        
    except Exception as e:
        logger.error(f"Vision-based CPT prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/predict-icd-from-pdfs")
async def predict_icd_from_pdfs(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    n_pages: int = Form(default=1, ge=1, le=50),
    model: str = Form(default="openai/gpt-5"),
    max_workers: int = Form(default=5),
    custom_instructions: Optional[str] = Form(default=None),
    instruction_template_id: Optional[int] = Form(default=None)
):
    """Upload a ZIP file containing PDFs to predict ICD codes using OpenAI vision model"""
    
    try:
        logger.info(f"Received vision-based ICD prediction request - zip: {zip_file.filename}, pages: {n_pages}, model: {model}, workers: {max_workers}")
        
        # Validate file type
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        # Fetch instruction template if provided
        if instruction_template_id:
            from db_utils import get_prediction_instruction
            template = get_prediction_instruction(instruction_id=instruction_template_id)
            if template:
                custom_instructions = template['instructions_text']
                logger.info(f"Using instruction template '{template['name']}' for ICD prediction")
            else:
                logger.warning(f"Instruction template {instruction_template_id} not found, proceeding without custom instructions")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created vision-based ICD prediction job {job_id}")
        
        # Save uploaded ZIP
        zip_path = f"/tmp/{job_id}_icd_pdfs.zip"
        
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        
        logger.info(f"ZIP saved - path: {zip_path}")
        
        # Start background processing with vision model
        background_tasks.add_task(predict_icd_from_pdfs_background, job_id, zip_path, n_pages, model, max_workers, custom_instructions)
        
        logger.info(f"Background vision-based ICD prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": f"ZIP uploaded and OpenAI {model} vision ICD prediction started"}
        
    except Exception as e:
        logger.error(f"Vision-based ICD prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/split-pdf")
async def split_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    filter_string: str = Form(..., description="Text to search for in PDF pages for splitting")
):
    """Upload a single PDF file to split into sections (OCR-based method)"""
    
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

@app.post("/split-pdf-gemini")
async def split_pdf_gemini(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    filter_string: str = Form(..., description="Text to search for in PDF pages for splitting"),
    batch_size: int = Form(default=5, description="Number of pages to process per API call (1-50)")
):
    """Upload a single PDF file to split into sections using new splitting method"""
    
    try:
        logger.info(f"Received new PDF split request - pdf: {pdf_file.filename}")
        
        # Validate file type
        if not pdf_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Validate batch size
        if batch_size < 1 or batch_size > 50:
            raise HTTPException(status_code=400, detail="Batch size must be between 1 and 50")
        
        # Configuration
        model = "gemini-flash-latest"
        max_workers = 12
        
        logger.info(f"Using batch_size: {batch_size}")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created Gemini split job {job_id}")
        
        # Save uploaded PDF
        pdf_path = f"/tmp/{job_id}_input.pdf"
        
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
        
        logger.info(f"PDF saved - path: {pdf_path}")
        
        # Start background processing with new splitting method
        background_tasks.add_task(split_pdf_gemini_background, job_id, pdf_path, filter_string, batch_size, model, max_workers)
        
        logger.info(f"Background split task started for job {job_id}")
        
        return {"job_id": job_id, "message": "PDF uploaded and splitting started"}
        
    except Exception as e:
        logger.error(f"Gemini PDF split upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/split-pdf-ocrspace")
async def split_pdf_ocrspace(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    filter_string: str = Form(..., description="Text to search for in PDF pages for splitting")
):
    """Upload a PDF file to split using OCR.space API - fastest and most reliable method"""
    
    try:
        logger.info(f"Received OCR.space PDF split request - pdf: {pdf_file.filename}, filter: {filter_string}")
        
        # Validate file type
        if not pdf_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Fixed configuration
        max_workers = 15  # Increased for speed (PRO tier can handle this)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created OCR.space split job {job_id}")
        
        # Save uploaded PDF
        pdf_path = f"/tmp/{job_id}_input.pdf"
        
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
        
        logger.info(f"PDF saved - path: {pdf_path}")
        
        # Start background processing with OCR.space method
        background_tasks.add_task(split_pdf_ocrspace_background, job_id, pdf_path, filter_string, max_workers)
        
        logger.info(f"Background OCR.space split task started for job {job_id}")
        
        return {"job_id": job_id, "message": "PDF uploaded and OCR.space splitting started"}
        
    except Exception as e:
        logger.error(f"OCR.space PDF split upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/download/{job_id}")
async def download_result(job_id: str, format: str = "csv"):
    """Download the processed file in CSV or XLSX format
    
    Args:
        job_id: The job identifier
        format: File format - 'csv' or 'xlsx' (default: 'csv')
    """
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Determine which file to download based on format parameter
    if format.lower() == "xlsx":
        # User wants XLSX format
        if not job.result_file_xlsx or not os.path.exists(job.result_file_xlsx):
            # XLSX not available, fallback to CSV
            logger.warning(f"XLSX file not available for job {job_id}, falling back to CSV")
            result_file = job.result_file
            filename_suffix = "csv"
            media_type = "text/csv"
        else:
            result_file = job.result_file_xlsx
            filename_suffix = "xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        # Default to CSV
        result_file = job.result_file
        filename_suffix = "csv"
        media_type = "text/csv"
    
    if not result_file or not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    # Determine filename based on job type
    if result_file.endswith('.zip'):
        filename = f"split_pdfs_{job_id}.zip"
        media_type = "application/zip"
    elif result_file.endswith('.xlsx'):
        filename = f"processed_data_{job_id}.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        filename = f"processed_data_{job_id}.csv"
        media_type = "text/csv"
    
    # Schedule cleanup after download (in background)
    import asyncio
    asyncio.create_task(cleanup_after_download(job_id))
    
    return FileResponse(
        result_file,
        media_type=media_type,
        filename=filename
    )

async def cleanup_after_download(job_id: str):
    """Clean up job after download with a delay"""
    import asyncio
    await asyncio.sleep(30)  # Wait 30 seconds after download
    
    if job_id in job_status:
        job = job_status[job_id]
        
        # Clean up CSV result file
        if job.result_file and os.path.exists(job.result_file):
            try:
                os.unlink(job.result_file)
                logger.info(f"Auto-cleaned up CSV result file: {job.result_file}")
            except Exception as e:
                logger.error(f"Failed to auto-cleanup CSV result file: {e}")
        
        # Clean up XLSX result file
        if job.result_file_xlsx and os.path.exists(job.result_file_xlsx):
            try:
                os.unlink(job.result_file_xlsx)
                logger.info(f"Auto-cleaned up XLSX result file: {job.result_file_xlsx}")
            except Exception as e:
                logger.error(f"Failed to auto-cleanup XLSX result file: {e}")
        
        # Remove job from memory
        del job_status[job_id]
        logger.info(f"Auto-cleaned up job {job_id} from memory")

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

def convert_uni_background(job_id: str, csv_path: str):
    """Background task to convert UNI CSV using the conversion script"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting UNI CSV conversion..."
        job.progress = 10
        
        # Import the conversion functions
        import sys
        sys.path.append(str(Path(__file__).parent / "uni-conversion"))
        from convert import convert_data
        
        job.message = "Converting UNI CSV data..."
        job.progress = 50
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_converted")
        result_base.parent.mkdir(exist_ok=True)
        
        output_file_csv = result_base.with_suffix('.csv')
        
        # Run the conversion
        success = convert_data(csv_path, str(output_file_csv))
        
        if not success:
            raise Exception("UNI conversion failed")
        
        job.message = "UNI conversion complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not os.path.exists(output_file_csv):
            raise Exception("Output file was not created")
        
        # Create XLSX version
        try:
            output_file_xlsx = convert_csv_to_xlsx(output_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = output_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(output_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = "UNI conversion completed successfully!"
        
        # Clean up input file
        os.unlink(csv_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after UNI conversion")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"UNI conversion failed: {str(e)}"
        logger.error(f"UNI conversion job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

def convert_instructions_background(job_id: str, excel_path: str):
    """Background task to convert Excel file using the instructions conversion script"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting instructions conversion..."
        job.progress = 10
        
        # Import required modules
        import pandas as pd
        import sys
        sys.path.append(str(Path(__file__).parent / "instructions-conversion"))
        from convert_instructions import convert_data
        
        job.message = "Reading Excel file..."
        job.progress = 30
        
        # Convert Excel to CSV first
        temp_csv_path = f"/tmp/{job_id}_temp_input.csv"
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            # Save as CSV
            df.to_csv(temp_csv_path, index=False)
        except Exception as e:
            raise Exception(f"Failed to read Excel file: {str(e)}")
        
        job.message = "Preparing additions file..."
        job.progress = 45
        
        # Copy the default additions.csv file
        import shutil
        additions_dest = Path(temp_csv_path).parent / "additions.csv"
        additions_source = Path(__file__).parent / "instructions-conversion" / "additions.csv"
        
        if additions_source.exists():
            shutil.copy2(additions_source, additions_dest)
            logger.info(f"Copied additions.csv from {additions_source} to {additions_dest}")
        else:
            logger.warning(f"additions.csv not found at {additions_source}")
        
        job.message = "Converting data using instructions script..."
        job.progress = 60
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_converted")
        result_base.parent.mkdir(exist_ok=True)
        
        output_file_csv = result_base.with_suffix('.csv')
        
        # Run the conversion using the convert_instructions script
        success = convert_data(temp_csv_path, str(output_file_csv))
        
        if not success:
            raise Exception("Instructions conversion failed")
        
        job.message = "Conversion complete, preparing Excel output..."
        job.progress = 85
        
        # Convert to Excel format
        excel_output_file = result_base.with_suffix('.xlsx')
        try:
            converted_df = pd.read_csv(output_file_csv, dtype=str)
            converted_df.to_excel(excel_output_file, index=False, engine='openpyxl')
            # Store both file paths
            job.result_file = str(output_file_csv)
            job.result_file_xlsx = str(excel_output_file)
            logger.info(f"Created both CSV and XLSX outputs")
        except Exception as e:
            # If Excel conversion fails, use CSV only
            logger.warning(f"Excel conversion failed, using CSV only: {str(e)}")
            job.result_file = str(output_file_csv)
        
        job.status = "completed"
        job.progress = 100
        job.message = "Instructions conversion completed successfully!"
        
        # Clean up temporary files
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)
        if os.path.exists(additions_dest):
            os.unlink(additions_dest)
        os.unlink(excel_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after instructions conversion")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Instructions conversion failed: {str(e)}"
        logger.error(f"Instructions conversion job {job_id} failed: {str(e)}")
        
        # Clean up on error
        try:
            if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
            if 'additions_dest' in locals() and os.path.exists(additions_dest):
                os.unlink(additions_dest)
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
        
        # Clean up memory even on failure
        gc.collect()

def merge_by_csn_background(job_id: str, csv_path_1: str, csv_path_2: str):
    """Background task to merge two CSV files by CSN column"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Reading CSV files..."
        job.progress = 10
        
        # Read both CSV files
        import pandas as pd
        
        # Try multiple encodings for CSV files
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
        df1 = None
        df2 = None
        
        for encoding in encodings_to_try:
            try:
                df1 = pd.read_csv(csv_path_1, dtype=str, encoding=encoding)
                logger.info(f"Successfully read first file with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df1 is None:
            raise Exception("Could not read first CSV file with any standard encoding")
        
        for encoding in encodings_to_try:
            try:
                df2 = pd.read_csv(csv_path_2, dtype=str, encoding=encoding)
                logger.info(f"Successfully read second file with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df2 is None:
            raise Exception("Could not read second CSV file with any standard encoding")
        
        job.message = "Checking for CSN column..."
        job.progress = 30
        
        # Find CSN column (case-insensitive)
        csn_col_1 = None
        csn_col_2 = None
        
        for col in df1.columns:
            if col.upper() == 'CSN':
                csn_col_1 = col
                break
        
        for col in df2.columns:
            if col.upper() == 'CSN':
                csn_col_2 = col
                break
        
        if csn_col_1 is None:
            raise Exception("First CSV file does not have a CSN column")
        if csn_col_2 is None:
            raise Exception("Second CSV file does not have a CSN column")
        
        job.message = "Merging files by CSN..."
        job.progress = 50
        
        # Merge the dataframes on CSN column (inner join)
        merged_df = pd.merge(
            df1, 
            df2, 
            left_on=csn_col_1, 
            right_on=csn_col_2, 
            how='inner',
            suffixes=('', '_y')
        )
        
        # Remove duplicate CSN column if both had the same name
        if csn_col_1 == csn_col_2:
            # Keep only one CSN column
            if f'{csn_col_1}_y' in merged_df.columns:
                merged_df = merged_df.drop(columns=[f'{csn_col_1}_y'])
        
        # Remove any other duplicate columns (ending with _y)
        cols_to_drop = [col for col in merged_df.columns if col.endswith('_y')]
        if cols_to_drop:
            merged_df = merged_df.drop(columns=cols_to_drop)
        
        job.message = "Preparing merged file..."
        job.progress = 80
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_merged")
        result_base.parent.mkdir(exist_ok=True)
        
        output_file_csv = result_base.with_suffix('.csv')
        
        # Save merged CSV
        merged_df.to_csv(output_file_csv, index=False, encoding='utf-8')
        
        # Create XLSX version
        try:
            output_file_xlsx = convert_csv_to_xlsx(output_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = output_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(output_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = f"Merge completed! {len(merged_df)} rows matched by CSN."
        
        # Clean up input files
        os.unlink(csv_path_1)
        os.unlink(csv_path_2)
        
        logger.info(f"Merge by CSN completed for job {job_id}: {len(merged_df)} rows")
        
    except Exception as e:
        logger.error(f"Merge by CSN background error: {str(e)}")
        job.status = "failed"
        job.error = str(e)
        job.message = f"Merge failed: {str(e)}"

def generate_modifiers_background(job_id: str, csv_path: str, turn_off_medical_direction: bool = False, generate_qk_duplicate: bool = False, limit_anesthesia_time: bool = False, peripheral_blocks_mode: str = "other"):
    """Background task to generate medical modifiers using the modifiers script
    
    Args:
        job_id: Unique job identifier
        csv_path: Path to input CSV file
        turn_off_medical_direction: If True, override all medical direction YES to NO
        generate_qk_duplicate: If True, generate duplicate line when QK modifier is applied with CRNA as Responsible Provider
        limit_anesthesia_time: If True, limit anesthesia time to maximum 480 minutes based on An Start and An Stop columns
        peripheral_blocks_mode: Mode for peripheral block generation - "UNI" (only General) or "other" (not MAC)
    """
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        qk_msg = ' + QK Duplicates' if generate_qk_duplicate else ''
        time_limit_msg = ' + Time Limited' if limit_anesthesia_time else ''
        blocks_mode_msg = f' (Blocks: {peripheral_blocks_mode})' if peripheral_blocks_mode else ''
        job.message = f"Starting modifiers generation{' (Medical Direction OFF)' if turn_off_medical_direction else ''}{qk_msg}{time_limit_msg}{blocks_mode_msg}..."
        job.progress = 10
        
        # Import the modifiers generation function
        import sys
        sys.path.append(str(Path(__file__).parent / "modifiers"))
        from generate_modifiers import generate_modifiers
        
        job.message = "Generating medical modifiers..."
        job.progress = 50
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_with_modifiers")
        result_base.parent.mkdir(exist_ok=True)
        
        output_file_csv = result_base.with_suffix('.csv')
        
        # Run the modifiers generation with medical direction override parameter, QK duplicate parameter, time limiting parameter, and peripheral blocks mode
        success = generate_modifiers(csv_path, str(output_file_csv), turn_off_medical_direction, generate_qk_duplicate, limit_anesthesia_time, peripheral_blocks_mode)
        
        if not success:
            raise Exception("Modifiers generation failed")
        
        job.message = "Modifiers generation complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not os.path.exists(output_file_csv):
            raise Exception("Output file was not created")
        
        # Create XLSX version
        try:
            output_file_xlsx = convert_csv_to_xlsx(output_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = output_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(output_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = "Modifiers generation completed successfully!"
        
        # Clean up input file
        os.unlink(csv_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after modifiers generation")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Modifiers generation failed: {str(e)}"
        logger.error(f"Modifiers generation job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

def predict_insurance_codes_background(job_id: str, data_csv_path: str, special_cases_csv_path: str = None, enable_ai: bool = True):
    """Background task to predict MedNet codes for insurance companies"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting insurance code prediction..."
        job.progress = 10
        
        # Import the prediction function
        import sys
        sys.path.append(str(Path(__file__).parent / "sorting"))
        from predict_mednet import process_insurance_predictions
        
        job.message = "Loading MedNet database..."
        job.progress = 20
        
        # Path to the mednet.csv file
        mednet_csv_path = Path(__file__).parent / "sorting" / "mednet.csv"
        
        if not mednet_csv_path.exists():
            raise Exception("MedNet database not found")
        
        job.message = "Predicting insurance codes..." + (" (AI enabled)" if enable_ai else " (special cases only)")
        job.progress = 30
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_with_insurance_codes")
        result_base.parent.mkdir(exist_ok=True)
        
        output_file_csv = result_base.with_suffix('.csv')
        
        # Run the prediction
        success = process_insurance_predictions(
            data_csv_path, 
            str(mednet_csv_path), 
            str(output_file_csv), 
            special_cases_csv_path,
            max_workers=10,
            enable_ai=enable_ai
        )
        
        if not success:
            raise Exception("Insurance code prediction failed")
        
        job.message = "Prediction complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not os.path.exists(output_file_csv):
            raise Exception("Output file was not created")
        
        # Create XLSX version
        try:
            output_file_xlsx = convert_csv_to_xlsx(output_file_csv, result_base.with_suffix('.xlsx'))
            job.result_file_xlsx = output_file_xlsx
        except Exception as e:
            logger.warning(f"Could not create XLSX version: {e}")
        
        job.result_file = str(output_file_csv)
        job.status = "completed"
        job.progress = 100
        job.message = "Insurance code prediction completed successfully!"
        
        # Clean up input files
        os.unlink(data_csv_path)
        if special_cases_csv_path and os.path.exists(special_cases_csv_path):
            os.unlink(special_cases_csv_path)
        
        # Force garbage collection to free memory
        gc.collect()
        logger.info(f"Job {job_id}: Memory cleaned up after insurance code prediction")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Insurance code prediction failed: {str(e)}"
        logger.error(f"Insurance code prediction job {job_id} failed: {str(e)}")
        
        # Clean up memory even on failure
        gc.collect()

@app.post("/convert-uni")
async def convert_uni(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...)
):
    """Upload a UNI CSV or XLSX file to convert using the conversion script"""
    
    try:
        logger.info(f"Received UNI conversion request - file: {csv_file.filename}")
        
        # Validate file type
        if not csv_file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created UNI conversion job {job_id}")
        
        # Save uploaded file
        input_path = f"/tmp/{job_id}_uni_input{Path(csv_file.filename).suffix}"
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"File saved - path: {input_path}")
        
        # Convert to CSV if needed
        csv_path = ensure_csv_file(input_path, f"/tmp/{job_id}_uni_input.csv")
        
        # Clean up original file if it was converted
        if csv_path != input_path and os.path.exists(input_path):
            os.unlink(input_path)
        
        logger.info(f"CSV ready - path: {csv_path}")
        
        # Start background processing
        background_tasks.add_task(convert_uni_background, job_id, csv_path)
        
        logger.info(f"Background UNI conversion task started for job {job_id}")
        
        return {"job_id": job_id, "message": "UNI file uploaded and conversion started"}
        
    except Exception as e:
        logger.error(f"UNI conversion upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/convert-instructions")
async def convert_instructions(
    background_tasks: BackgroundTasks,
    excel_file: UploadFile = File(...)
):
    """
    Upload an Excel file to convert using the instructions conversion script.
    Uses the default additions.csv file for field instructions.
    """
    
    try:
        logger.info(f"Received instructions conversion request - excel: {excel_file.filename}")
        
        # Validate file type
        if not excel_file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created instructions conversion job {job_id}")
        
        # Save uploaded Excel file
        excel_path = f"/tmp/{job_id}_instructions_input.xlsx"
        
        with open(excel_path, "wb") as f:
            shutil.copyfileobj(excel_file.file, f)
        
        logger.info(f"Instructions Excel saved - path: {excel_path}")
        
        # Start background processing
        background_tasks.add_task(convert_instructions_background, job_id, excel_path)
        
        logger.info(f"Background instructions conversion task started for job {job_id}")
        
        return {"job_id": job_id, "message": "Excel file uploaded and instructions conversion started"}
        
    except Exception as e:
        logger.error(f"Instructions conversion upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/merge-by-csn")
async def merge_by_csn(
    background_tasks: BackgroundTasks,
    csv_file_1: UploadFile = File(...),
    csv_file_2: UploadFile = File(...)
):
    """Upload two CSV or XLSX files to merge them by matching rows based on the CSN column"""
    
    try:
        logger.info(f"Received merge by CSN request - file1: {csv_file_1.filename}, file2: {csv_file_2.filename}")
        
        # Validate file types
        if not csv_file_1.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="First file must be a CSV or XLSX")
        if not csv_file_2.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Second file must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created merge by CSN job {job_id}")
        
        # Save uploaded files
        input_path_1 = f"/tmp/{job_id}_merge_input1{Path(csv_file_1.filename).suffix}"
        input_path_2 = f"/tmp/{job_id}_merge_input2{Path(csv_file_2.filename).suffix}"
        
        with open(input_path_1, "wb") as f:
            shutil.copyfileobj(csv_file_1.file, f)
        with open(input_path_2, "wb") as f:
            shutil.copyfileobj(csv_file_2.file, f)
        
        logger.info(f"Files saved - path1: {input_path_1}, path2: {input_path_2}")
        
        # Convert to CSV if needed
        csv_path_1 = ensure_csv_file(input_path_1, f"/tmp/{job_id}_merge_input1.csv")
        csv_path_2 = ensure_csv_file(input_path_2, f"/tmp/{job_id}_merge_input2.csv")
        
        # Clean up original files if they were converted
        if csv_path_1 != input_path_1 and os.path.exists(input_path_1):
            os.unlink(input_path_1)
        if csv_path_2 != input_path_2 and os.path.exists(input_path_2):
            os.unlink(input_path_2)
        
        logger.info(f"CSV files ready - path1: {csv_path_1}, path2: {csv_path_2}")
        
        # Start background processing
        background_tasks.add_task(merge_by_csn_background, job_id, csv_path_1, csv_path_2)
        
        logger.info(f"Background merge by CSN task started for job {job_id}")
        
        return {"job_id": job_id, "message": "Files uploaded and merge by CSN started"}
        
    except Exception as e:
        logger.error(f"Merge by CSN upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def check_cpt_codes_background(job_id: str, predictions_path: str, ground_truth_path: str):
    """Background task to compare predictions vs ground truth and calculate accuracy"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Reading Excel files..."
        job.progress = 10
        
        import pandas as pd
        
        # Read predictions file (handle both Excel and CSV)
        predictions_path_obj = Path(predictions_path)
        predictions_df = None
        
        if predictions_path_obj.suffix.lower() in ('.xlsx', '.xls'):
            try:
                # Use appropriate engine based on file extension
                if predictions_path_obj.suffix.lower() == '.xlsx':
                    predictions_df = pd.read_excel(predictions_path, dtype=str, engine='openpyxl')
                else:  # .xls file
                    # Try xlrd first, then fall back to auto-detect
                    try:
                        predictions_df = pd.read_excel(predictions_path, dtype=str, engine='xlrd')
                    except Exception:
                        # Auto-detect engine (will use xlrd if available, otherwise openpyxl)
                        predictions_df = pd.read_excel(predictions_path, dtype=str, engine=None)
                logger.info(f"Successfully read predictions file as Excel ({predictions_path_obj.suffix.lower()})")
            except Exception as e:
                logger.warning(f"Failed to read predictions as Excel: {e}, trying CSV with multiple encodings")
                # Try multiple encodings for CSV files
                encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                for encoding in encodings_to_try:
                    try:
                        predictions_df = pd.read_csv(predictions_path, dtype=str, encoding=encoding)
                        logger.info(f"Successfully read predictions file as CSV with encoding: {encoding}")
                        break
                    except (UnicodeDecodeError, Exception) as csv_error:
                        continue
        else:
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            for encoding in encodings_to_try:
                try:
                    predictions_df = pd.read_csv(predictions_path, dtype=str, encoding=encoding)
                    logger.info(f"Successfully read predictions file with encoding: {encoding}")
                    break
                except (UnicodeDecodeError, Exception) as csv_error:
                    continue
        
        if predictions_df is None:
            raise Exception("Could not read predictions file. Please ensure it's a valid Excel (.xlsx, .xls) or CSV file.")
        
        # Read ground truth file (handle both Excel and CSV)
        ground_truth_path_obj = Path(ground_truth_path)
        ground_truth_df = None
        
        if ground_truth_path_obj.suffix.lower() in ('.xlsx', '.xls'):
            try:
                # Use appropriate engine based on file extension
                if ground_truth_path_obj.suffix.lower() == '.xlsx':
                    ground_truth_df = pd.read_excel(ground_truth_path, dtype=str, engine='openpyxl')
                else:  # .xls file
                    # Try xlrd first, then fall back to auto-detect
                    try:
                        ground_truth_df = pd.read_excel(ground_truth_path, dtype=str, engine='xlrd')
                    except Exception:
                        # Auto-detect engine (will use xlrd if available, otherwise openpyxl)
                        ground_truth_df = pd.read_excel(ground_truth_path, dtype=str, engine=None)
                logger.info(f"Successfully read ground truth file as Excel ({ground_truth_path_obj.suffix.lower()})")
            except Exception as e:
                logger.warning(f"Failed to read ground truth as Excel: {e}, trying CSV with multiple encodings")
                # Try multiple encodings for CSV files
                encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                for encoding in encodings_to_try:
                    try:
                        ground_truth_df = pd.read_csv(ground_truth_path, dtype=str, encoding=encoding)
                        logger.info(f"Successfully read ground truth file as CSV with encoding: {encoding}")
                        break
                    except (UnicodeDecodeError, Exception) as csv_error:
                        continue
        else:
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            for encoding in encodings_to_try:
                try:
                    ground_truth_df = pd.read_csv(ground_truth_path, dtype=str, encoding=encoding)
                    logger.info(f"Successfully read ground truth file with encoding: {encoding}")
                    break
                except (UnicodeDecodeError, Exception) as csv_error:
                    continue
        
        if ground_truth_df is None:
            raise Exception("Could not read ground truth file. Please ensure it's a valid Excel (.xlsx, .xls) or CSV file.")
        
        job.message = "Finding required columns..."
        job.progress = 20
        
        # Find AccountId column in predictions (case-insensitive)
        account_id_col_pred = None
        asa_code_col_pred = None
        for col in predictions_df.columns:
            col_upper = col.upper().strip()
            if col_upper == 'ACCOUNTID' or col_upper == 'ACCOUNT ID':
                account_id_col_pred = col
            elif col_upper == 'ASA CODE' or col_upper == 'ASA':
                asa_code_col_pred = col
        
        # Find Acc. # column in ground truth (case-insensitive)
        account_id_col_gt = None
        asa_code_col_gt = None
        for col in ground_truth_df.columns:
            col_upper = col.upper().strip()
            if col_upper == 'ACC. #' or col_upper == 'ACC #' or col_upper == 'ACCOUNTID' or col_upper == 'ACCOUNT ID':
                account_id_col_gt = col
            elif col_upper == 'ASA CODE' or col_upper == 'ASA':
                asa_code_col_gt = col
        
        if account_id_col_pred is None:
            raise Exception("Predictions file must have 'AccountId' column")
        if asa_code_col_pred is None:
            raise Exception("Predictions file must have 'ASA Code' column")
        if account_id_col_gt is None:
            raise Exception("Ground truth file must have 'Acc. #' column")
        if asa_code_col_gt is None:
            raise Exception("Ground truth file must have 'ASA Code' column")
        
        job.message = "Matching accounts and comparing codes..."
        job.progress = 40
        
        # Create a dictionary for ground truth lookup (store full row data)
        # IMPORTANT: Only take the FIRST occurrence of each account ID
        gt_dict = {}
        gt_row_dict = {}  # Store full row data for detailed reporting
        for idx, row in ground_truth_df.iterrows():
            account_id = str(row[account_id_col_gt]).strip()
            # Only add if we haven't seen this account ID before (take FIRST occurrence)
            if account_id not in gt_dict:
                asa_code = str(row[asa_code_col_gt]).strip() if pd.notna(row[asa_code_col_gt]) else ''
                gt_dict[account_id] = asa_code
                gt_row_dict[account_id] = row.to_dict()  # Store full row for detailed report
        
        # Compare predictions with ground truth
        comparison_data = []
        detailed_report_data = []
        case_overview_data = []  # Overview of each case
        matches = 0
        mismatches = 0
        not_found = 0
        
        for idx, row in predictions_df.iterrows():
            account_id = str(row[account_id_col_pred]).strip()
            predicted_code = str(row[asa_code_col_pred]).strip() if pd.notna(row[asa_code_col_pred]) else ''
            
            # Start building detailed report entry
            detailed_entry = {
                'Row Number': idx + 1,  # 1-indexed for readability
                'AccountId': account_id,
                'Match Status': '',
                'Predicted ASA Code': predicted_code,
                'Ground Truth ASA Code': '',
                'Match': '',
                'Difference': '',
            }
            
            # Start building case overview entry - includes ALL columns from both files
            case_overview = {
                'Case #': idx + 1,
                'Account ID': account_id,
                'Status': '',
                'Predicted ASA Code': predicted_code,
                'Ground Truth ASA Code': '',
                'Result': '',
                'Notes': '',
            }
            
            # Add ALL columns from predictions file to overview
            for col in predictions_df.columns:
                if col != account_id_col_pred and col != asa_code_col_pred:
                    value = row[col]
                    detailed_entry[f'Predictions: {col}'] = str(value) if pd.notna(value) else ''
                    case_overview[f'Predictions: {col}'] = str(value) if pd.notna(value) else ''
            
            if account_id in gt_dict:
                ground_truth_code = gt_dict[account_id]
                is_match = predicted_code == ground_truth_code
                
                if is_match:
                    matches += 1
                    detailed_entry['Match Status'] = 'Match'
                    detailed_entry['Match'] = 'Yes'
                    detailed_entry['Difference'] = 'None'
                    case_overview['Status'] = '✅ MATCH'
                    case_overview['Result'] = 'CORRECT'
                    case_overview['Notes'] = f'Predicted code {predicted_code} matches ground truth'
                else:
                    mismatches += 1
                    detailed_entry['Match Status'] = 'Mismatch'
                    detailed_entry['Match'] = 'No'
                    detailed_entry['Difference'] = f"Predicted: {predicted_code} vs Ground Truth: {ground_truth_code}"
                    case_overview['Status'] = '❌ MISMATCH'
                    case_overview['Result'] = 'INCORRECT'
                    case_overview['Notes'] = f'Predicted {predicted_code} but should be {ground_truth_code}'
                
                detailed_entry['Ground Truth ASA Code'] = ground_truth_code
                case_overview['Ground Truth ASA Code'] = ground_truth_code
                
                # Add ALL columns from ground truth file (with prefix)
                gt_row = gt_row_dict[account_id]
                for col in ground_truth_df.columns:
                    if col != account_id_col_gt and col != asa_code_col_gt:
                        value = gt_row[col]
                        detailed_entry[f'Ground Truth: {col}'] = str(value) if pd.notna(value) else ''
                        case_overview[f'Ground Truth: {col}'] = str(value) if pd.notna(value) else ''
                
                # Simple comparison entry (for backward compatibility)
                comparison_data.append({
                    'AccountId': account_id,
                    'Predicted ASA Code': predicted_code,
                    'Ground Truth ASA Code': ground_truth_code,
                    'Match': 'Yes' if is_match else 'No',
                    'Status': 'Match' if is_match else 'Mismatch'
                })
            else:
                not_found += 1
                detailed_entry['Match Status'] = 'Account Not Found'
                detailed_entry['Match'] = 'No'
                detailed_entry['Ground Truth ASA Code'] = 'NOT FOUND'
                detailed_entry['Difference'] = 'Account ID not found in ground truth file'
                case_overview['Status'] = '⚠️ NOT FOUND'
                case_overview['Result'] = 'NO COMPARISON'
                case_overview['Ground Truth ASA Code'] = 'NOT FOUND'
                case_overview['Notes'] = f'Account {account_id} not found in ground truth file'
                
                # Add empty columns for ground truth (to maintain structure)
                for col in ground_truth_df.columns:
                    if col != account_id_col_gt and col != asa_code_col_gt:
                        detailed_entry[f'Ground Truth: {col}'] = ''
                
                # Simple comparison entry (for backward compatibility)
                comparison_data.append({
                    'AccountId': account_id,
                    'Predicted ASA Code': predicted_code,
                    'Ground Truth ASA Code': 'NOT FOUND',
                    'Match': 'No',
                    'Status': 'Account Not Found'
                })
            
            detailed_report_data.append(detailed_entry)
            case_overview_data.append(case_overview)
        
        job.message = "Creating comparison report..."
        job.progress = 70
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create detailed report DataFrame
        detailed_report_df = pd.DataFrame(detailed_report_data)
        
        # Create case overview DataFrame
        case_overview_df = pd.DataFrame(case_overview_data)
        
        # Calculate accuracy metrics
        total_comparable = matches + mismatches
        accuracy = (matches / total_comparable * 100) if total_comparable > 0 else 0
        
        # Create summary sheet with more detailed metrics
        summary_data = {
            'Metric': [
                'Total Predictions',
                'Accounts Found in Ground Truth',
                'Accounts Not Found',
                'Matches',
                'Mismatches',
                'Accuracy (%)',
                'Match Rate (%)',
                'Mismatch Rate (%)',
                'Not Found Rate (%)'
            ],
            'Value': [
                len(predictions_df),
                total_comparable,
                not_found,
                matches,
                mismatches,
                f'{accuracy:.2f}%',
                f'{(matches / len(predictions_df) * 100):.2f}%' if len(predictions_df) > 0 else '0.00%',
                f'{(mismatches / len(predictions_df) * 100):.2f}%' if len(predictions_df) > 0 else '0.00%',
                f'{(not_found / len(predictions_df) * 100):.2f}%' if len(predictions_df) > 0 else '0.00%'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        job.message = "Generating Excel report..."
        job.progress = 85
        
        # Create output file path
        result_base = Path(f"/tmp/results/{job_id}_cpt_comparison")
        result_base.parent.mkdir(exist_ok=True)
        
        output_file_xlsx = result_base.with_suffix('.xlsx')
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(output_file_xlsx, engine='openpyxl') as writer:
            # Summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Case Overview sheet - shows each case with key information (PRIMARY VIEW)
            case_overview_df.to_excel(writer, sheet_name='Case Overview', index=False)
            
            # Detailed report sheet (all cases with full details)
            detailed_report_df.to_excel(writer, sheet_name='Detailed Report', index=False)
            
            # Simple comparison sheet (for backward compatibility)
            comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
            
            # Create a sheet with only mismatches (from case overview)
            mismatches_overview_df = case_overview_df[case_overview_df['Status'].str.contains('MISMATCH', na=False)]
            mismatches_overview_df.to_excel(writer, sheet_name='Mismatches Overview', index=False)
            
            # Create a sheet with only matches (from case overview)
            matches_overview_df = case_overview_df[case_overview_df['Status'].str.contains('MATCH', na=False)]
            matches_overview_df.to_excel(writer, sheet_name='Matches Overview', index=False)
            
            # Create a sheet with accounts not found (from case overview)
            not_found_overview_df = case_overview_df[case_overview_df['Status'].str.contains('NOT FOUND', na=False)]
            not_found_overview_df.to_excel(writer, sheet_name='Not Found Overview', index=False)
            
            # Create a sheet with only mismatches (from detailed report)
            mismatches_detailed_df = detailed_report_df[detailed_report_df['Match Status'] == 'Mismatch']
            mismatches_detailed_df.to_excel(writer, sheet_name='Mismatches (Detailed)', index=False)
            
            # Create a sheet with only matches (from detailed report)
            matches_detailed_df = detailed_report_df[detailed_report_df['Match Status'] == 'Match']
            matches_detailed_df.to_excel(writer, sheet_name='Matches (Detailed)', index=False)
            
            # Create a sheet with accounts not found
            not_found_df = detailed_report_df[detailed_report_df['Match Status'] == 'Account Not Found']
            not_found_df.to_excel(writer, sheet_name='Not Found (Detailed)', index=False)
        
        job.result_file_xlsx = str(output_file_xlsx)
        job.result_file = str(output_file_xlsx)  # Use XLSX as primary result
        job.status = "completed"
        job.progress = 100
        job.message = f"Comparison completed! Accuracy: {accuracy:.2f}% ({matches}/{total_comparable} matches)"
        
        # Clean up input files
        if os.path.exists(predictions_path):
            os.unlink(predictions_path)
        if os.path.exists(ground_truth_path):
            os.unlink(ground_truth_path)
        
        logger.info(f"CPT codes check completed for job {job_id}: {accuracy:.2f}% accuracy")
        
    except Exception as e:
        logger.error(f"CPT codes check background error: {str(e)}")
        job.status = "failed"
        job.error = str(e)
        job.message = f"Comparison failed: {str(e)}"
        
        # Clean up memory even on failure
        gc.collect()

@app.post("/check-cpt-codes")
async def check_cpt_codes(
    background_tasks: BackgroundTasks,
    predictions_file: UploadFile = File(...),
    ground_truth_file: UploadFile = File(...)
):
    """Upload two XLSX files (predictions and ground truth) to compare CPT codes and calculate accuracy"""
    
    try:
        logger.info(f"Received CPT codes check request - predictions: {predictions_file.filename}, ground_truth: {ground_truth_file.filename}")
        
        # Validate file types
        if not predictions_file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Predictions file must be a CSV or XLSX")
        if not ground_truth_file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Ground truth file must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created CPT codes check job {job_id}")
        
        # Save uploaded files
        predictions_path = f"/tmp/{job_id}_predictions{Path(predictions_file.filename).suffix}"
        ground_truth_path = f"/tmp/{job_id}_ground_truth{Path(ground_truth_file.filename).suffix}"
        
        with open(predictions_path, "wb") as f:
            shutil.copyfileobj(predictions_file.file, f)
        with open(ground_truth_path, "wb") as f:
            shutil.copyfileobj(ground_truth_file.file, f)
        
        logger.info(f"Files saved - predictions: {predictions_path}, ground_truth: {ground_truth_path}")
        
        # Start background processing
        background_tasks.add_task(check_cpt_codes_background, job_id, predictions_path, ground_truth_path)
        
        logger.info(f"Background CPT codes check task started for job {job_id}")
        
        return {"job_id": job_id, "message": "Files uploaded and CPT codes comparison started"}
        
    except Exception as e:
        logger.error(f"CPT codes check upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/generate-modifiers")
async def generate_modifiers_route(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    turn_off_medical_direction: bool = Form(False),
    generate_qk_duplicate: bool = Form(False),
    limit_anesthesia_time: bool = Form(False),
    peripheral_blocks_mode: str = Form("other")
):
    """Upload a CSV or XLSX file to generate medical modifiers
    
    Args:
        csv_file: CSV or XLSX file with billing data
        turn_off_medical_direction: If True, override all medical direction YES to NO
        generate_qk_duplicate: If True, generate duplicate line when QK modifier is applied with CRNA as Responsible Provider
        limit_anesthesia_time: If True, limit anesthesia time to maximum 480 minutes based on An Start and An Stop columns
        peripheral_blocks_mode: Mode for peripheral block generation - "UNI" (only General) or "other" (not MAC)
    """
    
    try:
        logger.info(f"Received modifiers generation request - file: {csv_file.filename}, turn_off_medical_direction: {turn_off_medical_direction}, generate_qk_duplicate: {generate_qk_duplicate}, limit_anesthesia_time: {limit_anesthesia_time}, peripheral_blocks_mode: {peripheral_blocks_mode}")
        
        # Validate file type
        if not csv_file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created modifiers generation job {job_id}")
        
        # Save uploaded file
        input_path = f"/tmp/{job_id}_modifiers_input{Path(csv_file.filename).suffix}"
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"File saved - path: {input_path}")
        
        # Convert to CSV if needed
        csv_path = ensure_csv_file(input_path, f"/tmp/{job_id}_modifiers_input.csv")
        
        # Clean up original file if it was converted
        if csv_path != input_path and os.path.exists(input_path):
            os.unlink(input_path)
        
        logger.info(f"CSV ready - path: {csv_path}")
        
        # Start background processing
        background_tasks.add_task(generate_modifiers_background, job_id, csv_path, turn_off_medical_direction, generate_qk_duplicate, limit_anesthesia_time, peripheral_blocks_mode)
        
        logger.info(f"Background modifiers generation task started for job {job_id}")
        
        qk_msg = ' + QK Duplicates' if generate_qk_duplicate else ''
        time_limit_msg = ' + Time Limited' if limit_anesthesia_time else ''
        blocks_mode_msg = f' (Blocks: {peripheral_blocks_mode})' if peripheral_blocks_mode else ''
        return {"job_id": job_id, "message": f"File uploaded and modifiers generation started{' (Medical Direction OFF)' if turn_off_medical_direction else ''}{qk_msg}{time_limit_msg}{blocks_mode_msg}"}
        
    except Exception as e:
        logger.error(f"Modifiers generation upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/predict-insurance-codes")
async def predict_insurance_codes(
    background_tasks: BackgroundTasks,
    data_csv: UploadFile = File(...),
    special_cases_csv: UploadFile = File(None),
    special_cases_template_id: int = Form(None),
    enable_ai: bool = Form(True)
):
    """
    Upload data CSV/XLSX and provide special cases either via:
    - Upload CSV file (special_cases_csv)
    - Select saved template (special_cases_template_id)
    """
    
    try:
        logger.info(f"Received insurance code prediction request - data: {data_csv.filename}, AI enabled: {enable_ai}")
        
        # Validate file type
        if not data_csv.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Data file must be a CSV or XLSX")
        
        if special_cases_csv and not special_cases_csv.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Special cases file must be a CSV or XLSX")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created insurance prediction job {job_id}")
        
        # Save uploaded data file
        data_input_path = f"/tmp/{job_id}_data_input{Path(data_csv.filename).suffix}"
        
        with open(data_input_path, "wb") as f:
            shutil.copyfileobj(data_csv.file, f)
        
        logger.info(f"Data file saved - path: {data_input_path}")
        
        # Convert to CSV if needed
        data_csv_path = ensure_csv_file(data_input_path, f"/tmp/{job_id}_data_input.csv")
        
        # Clean up original file if it was converted
        if data_csv_path != data_input_path and os.path.exists(data_input_path):
            os.unlink(data_input_path)
        
        logger.info(f"Data CSV ready - path: {data_csv_path}")
        
        # Handle special cases - either from upload or from template
        special_cases_csv_path = None
        if special_cases_template_id:
            # Load from database template
            from db_utils import get_special_cases_template
            template = get_special_cases_template(template_id=special_cases_template_id)
            
            if not template:
                raise HTTPException(status_code=404, detail=f"Special cases template {special_cases_template_id} not found")
            
            # Create CSV file from template mappings
            import pandas as pd
            mappings_df = pd.DataFrame(template['mappings'])
            mappings_df.columns = ['Company name', 'Mednet code']
            
            special_cases_csv_path = f"/tmp/{job_id}_special_cases.csv"
            mappings_df.to_csv(special_cases_csv_path, index=False)
            logger.info(f"Special cases loaded from template {special_cases_template_id} - {len(template['mappings'])} mappings")
            
        elif special_cases_csv:
            # Load from uploaded file
            special_cases_input_path = f"/tmp/{job_id}_special_cases{Path(special_cases_csv.filename).suffix}"
            with open(special_cases_input_path, "wb") as f:
                shutil.copyfileobj(special_cases_csv.file, f)
            logger.info(f"Special cases file saved - path: {special_cases_input_path}")
            
            # Convert to CSV if needed
            special_cases_csv_path = ensure_csv_file(special_cases_input_path, f"/tmp/{job_id}_special_cases.csv")
            
            # Clean up original file if it was converted
            if special_cases_csv_path != special_cases_input_path and os.path.exists(special_cases_input_path):
                os.unlink(special_cases_input_path)
            
            logger.info(f"Special cases CSV ready - path: {special_cases_csv_path}")
        
        # Start background processing
        background_tasks.add_task(predict_insurance_codes_background, job_id, data_csv_path, special_cases_csv_path, enable_ai)
        
        logger.info(f"Background insurance prediction task started for job {job_id} with AI: {enable_ai}")
        
        return {"job_id": job_id, "message": "File uploaded and insurance code prediction started"}
        
    except Exception as e:
        logger.error(f"Insurance prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/memory")
async def memory_status():
    """Memory usage and job status endpoint"""
    import psutil
    
    # Get memory info
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Count jobs by status
    job_counts = {}
    for job in job_status.values():
        status = job.status
        job_counts[status] = job_counts.get(status, 0) + 1
    
    # Count result files
    results_dir = Path("/tmp/results")
    result_files = list(results_dir.glob("*")) if results_dir.exists() else []
    total_result_size = sum(f.stat().st_size for f in result_files if f.is_file())
    
    return {
        "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
        "memory_percent": round(process.memory_percent(), 2),
        "total_jobs": len(job_status),
        "job_counts": job_counts,
        "result_files_count": len(result_files),
        "result_files_size_mb": round(total_result_size / 1024 / 1024, 2),
        "gc_counts": gc.get_count()
    }

@app.post("/force-gc")
async def force_garbage_collection():
    """Force garbage collection to free memory"""
    import psutil
    
    # Get memory before
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory after
    memory_after = process.memory_info().rss / 1024 / 1024
    memory_freed = memory_before - memory_after
    
    logger.info(f"Forced garbage collection: collected {collected} objects, freed {memory_freed:.2f} MB")
    
    return {
        "message": "Garbage collection completed",
        "objects_collected": collected,
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "memory_freed_mb": round(memory_freed, 2)
    }

# ============================================================================
# Modifiers Configuration API Endpoints
# ============================================================================

@app.get("/api/modifiers")
async def get_modifiers(page: int = 1, page_size: int = 50, search: str = None):
    """Get modifier configurations from database with pagination"""
    try:
        from db_utils import get_all_modifiers
        result = get_all_modifiers(page=page, page_size=page_size, search=search)
        return result
    except Exception as e:
        logger.error(f"Failed to get modifiers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve modifiers: {str(e)}")

@app.get("/api/modifiers/{mednet_code}")
async def get_modifier(mednet_code: str):
    """Get a specific modifier configuration by MedNet code"""
    try:
        from db_utils import get_modifier as get_modifier_by_code
        modifier = get_modifier_by_code(mednet_code)
        if modifier:
            return modifier
        else:
            raise HTTPException(status_code=404, detail=f"Modifier {mednet_code} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get modifier {mednet_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve modifier: {str(e)}")

@app.post("/api/modifiers")
async def create_or_update_modifier(
    mednet_code: str = Form(...),
    medicare_modifiers: bool = Form(...),
    bill_medical_direction: bool = Form(...)
):
    """Create or update a modifier configuration"""
    try:
        from db_utils import upsert_modifier
        success = upsert_modifier(mednet_code, medicare_modifiers, bill_medical_direction)
        if success:
            return {
                "message": "Modifier saved successfully",
                "mednet_code": mednet_code,
                "medicare_modifiers": medicare_modifiers,
                "bill_medical_direction": bill_medical_direction
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save modifier")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save modifier {mednet_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save modifier: {str(e)}")

@app.put("/api/modifiers/{mednet_code}")
async def update_modifier(
    mednet_code: str,
    medicare_modifiers: bool = Form(...),
    bill_medical_direction: bool = Form(...)
):
    """Update an existing modifier configuration"""
    try:
        from db_utils import upsert_modifier
        success = upsert_modifier(mednet_code, medicare_modifiers, bill_medical_direction)
        if success:
            return {
                "message": "Modifier updated successfully",
                "mednet_code": mednet_code,
                "medicare_modifiers": medicare_modifiers,
                "bill_medical_direction": bill_medical_direction
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update modifier")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update modifier {mednet_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update modifier: {str(e)}")

@app.delete("/api/modifiers/{mednet_code}")
async def delete_modifier(mednet_code: str):
    """Delete a modifier configuration"""
    try:
        from db_utils import delete_modifier as delete_modifier_by_code
        success = delete_modifier_by_code(mednet_code)
        if success:
            return {"message": f"Modifier {mednet_code} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete modifier")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete modifier {mednet_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete modifier: {str(e)}")


# ============================================================================
# Instruction Templates API Endpoints
# ============================================================================

@app.get("/api/templates")
async def get_templates(page: int = 1, page_size: int = 50, search: str = None):
    """Get instruction templates from database with pagination"""
    try:
        from db_utils import get_all_templates
        result = get_all_templates(page=page, page_size=page_size, search=search)
        return result
    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve templates: {str(e)}")


@app.get("/api/templates/{template_id}")
async def get_template(template_id: int):
    """Get a specific instruction template by ID"""
    try:
        from db_utils import get_template as get_template_by_id
        template = get_template_by_id(template_id=template_id)
        if template:
            return template
        else:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve template: {str(e)}")


@app.get("/api/templates/by-name/{template_name}")
async def get_template_by_name(template_name: str):
    """Get a specific instruction template by name"""
    try:
        from db_utils import get_template
        template = get_template(template_name=template_name)
        if template:
            return template
        else:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template '{template_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve template: {str(e)}")


@app.post("/api/templates/upload")
async def upload_template(
    name: str = Form(...),
    description: str = Form(""),
    excel_file: UploadFile = File(...)
):
    """Upload an Excel file and save it as an instruction template"""
    import pandas as pd
    
    try:
        # Validate file type
        if not excel_file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        try:
            content = await excel_file.read()
            temp_file.write(content)
            temp_file.close()
            
            # Parse the Excel file to extract field definitions
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'current'))
            from field_definitions import load_field_definitions_from_excel
            
            field_definitions = load_field_definitions_from_excel(temp_file.name)
            
            if not field_definitions:
                raise HTTPException(status_code=400, detail="Failed to parse Excel file or no fields found")
            
            # Store template in database
            from db_utils import create_template
            template_id = create_template(
                name=name,
                description=description,
                template_data={'fields': field_definitions}
            )
            
            if template_id:
                return {
                    "message": "Template uploaded successfully",
                    "template_id": template_id,
                    "name": name,
                    "fields_count": len(field_definitions)
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to save template to database")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload template: {str(e)}")


@app.put("/api/templates/{template_id}")
async def update_template(
    template_id: int,
    name: str = Form(None),
    description: str = Form(None),
    excel_file: UploadFile = File(None)
):
    """Update an existing instruction template"""
    import pandas as pd
    
    try:
        # Check if template exists
        from db_utils import get_template, update_template as update_template_in_db
        existing_template = get_template(template_id=template_id)
        if not existing_template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        # Prepare update parameters
        template_data = None
        
        # If new Excel file is provided, parse it
        if excel_file and excel_file.filename:
            if not excel_file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
            try:
                content = await excel_file.read()
                temp_file.write(content)
                temp_file.close()
                
                # Parse the Excel file
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'current'))
                from field_definitions import load_field_definitions_from_excel
                
                field_definitions = load_field_definitions_from_excel(temp_file.name)
                
                if not field_definitions:
                    raise HTTPException(status_code=400, detail="Failed to parse Excel file or no fields found")
                
                template_data = {'fields': field_definitions}
            
            finally:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
        
        # Update template in database
        success = update_template_in_db(
            template_id=template_id,
            name=name,
            description=description,
            template_data=template_data
        )
        
        if success:
            # Get updated template
            updated_template = get_template(template_id=template_id)
            return {
                "message": "Template updated successfully",
                "template": updated_template
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update template")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")


@app.put("/api/templates/{template_id}/fields")
async def update_template_fields(template_id: int, request: dict):
    """Update template fields directly via JSON"""
    try:
        from db_utils import get_template, update_template as update_template_in_db
        
        # Check if template exists
        existing_template = get_template(template_id=template_id)
        if not existing_template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        # Get template_data from request body
        template_data = request.get('template_data')
        if not template_data:
            raise HTTPException(status_code=400, detail="template_data is required")
        
        # Validate that fields exist
        fields = template_data.get('fields', [])
        if not fields:
            raise HTTPException(status_code=400, detail="At least one field is required")
        
        # Validate that all fields have names
        for field in fields:
            if not field.get('name'):
                raise HTTPException(status_code=400, detail="All fields must have a name")
        
        # Update template in database
        success = update_template_in_db(
            template_id=template_id,
            template_data=template_data
        )
        
        if success:
            # Get updated template
            updated_template = get_template(template_id=template_id)
            return {
                "message": "Template fields updated successfully",
                "template": updated_template
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update template")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template fields {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template fields: {str(e)}")


@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: int):
    """Delete an instruction template"""
    try:
        from db_utils import delete_template as delete_template_by_id
        success = delete_template_by_id(template_id)
        if success:
            return {"message": f"Template {template_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete template")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")


@app.post("/api/templates/{template_id}/export")
async def export_template_as_excel(template_id: int):
    """Export a template back to Excel format for download"""
    import pandas as pd
    
    try:
        from db_utils import get_template
        template = get_template(template_id=template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        # Extract field definitions from template
        fields = template['template_data'].get('fields', [])
        
        if not fields:
            raise HTTPException(status_code=400, detail="Template has no field definitions")
        
        # Create DataFrame in the expected format
        # Columns are field names, rows are description, location, output_format, priority
        data = {}
        for field in fields:
            data[field['name']] = [
                field.get('description', ''),
                field.get('location', ''),
                field.get('output_format', ''),
                'YES' if field.get('priority', False) else 'NO'
            ]
        
        df = pd.DataFrame(data)
        
        # Create temporary Excel file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        df.to_excel(temp_file.name, index=False, engine='openpyxl')
        temp_file.close()
        
        # Return file
        filename = f"{template['name'].replace(' ', '_')}.xlsx"
        return FileResponse(
            path=temp_file.name,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export template: {str(e)}")


# ============================================================================
# Prediction Instructions API Endpoints (CPT/ICD)
# ============================================================================

@app.get("/api/prediction-instructions")
async def get_prediction_instructions(
    instruction_type: str = None,
    page: int = 1,
    page_size: int = 50,
    search: str = None
):
    """Get prediction instruction templates (CPT/ICD) with pagination"""
    try:
        from db_utils import get_all_prediction_instructions
        result = get_all_prediction_instructions(
            instruction_type=instruction_type,
            page=page,
            page_size=page_size,
            search=search
        )
        return result
    except Exception as e:
        logger.error(f"Failed to get prediction instructions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve instructions: {str(e)}")


@app.get("/api/prediction-instructions/{instruction_id}")
async def get_prediction_instruction(instruction_id: int):
    """Get a specific prediction instruction by ID"""
    try:
        from db_utils import get_prediction_instruction as get_instruction_by_id
        instruction = get_instruction_by_id(instruction_id=instruction_id)
        if instruction:
            return instruction
        else:
            raise HTTPException(status_code=404, detail=f"Instruction {instruction_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction instruction {instruction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve instruction: {str(e)}")


@app.post("/api/prediction-instructions")
async def create_prediction_instruction(request: dict):
    """Create a new prediction instruction template"""
    try:
        name = request.get("name")
        instruction_type = request.get("instruction_type")
        instructions_text = request.get("instructions_text")
        description = request.get("description", "")
        
        if not name or not instruction_type or not instructions_text:
            raise HTTPException(
                status_code=400,
                detail="name, instruction_type, and instructions_text are required"
            )
        
        if instruction_type not in ['cpt', 'icd']:
            raise HTTPException(
                status_code=400,
                detail="instruction_type must be 'cpt' or 'icd'"
            )
        
        from db_utils import create_prediction_instruction
        instruction_id = create_prediction_instruction(
            name=name,
            instruction_type=instruction_type,
            instructions_text=instructions_text,
            description=description
        )
        
        if instruction_id:
            return {
                "message": "Instruction template created successfully",
                "instruction_id": instruction_id,
                "name": name,
                "instruction_type": instruction_type
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create instruction template")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create prediction instruction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create instruction: {str(e)}")


@app.put("/api/prediction-instructions/{instruction_id}")
async def update_prediction_instruction(instruction_id: int, request: dict):
    """Update an existing prediction instruction template"""
    try:
        from db_utils import get_prediction_instruction, update_prediction_instruction as update_instruction_in_db
        
        # Check if instruction exists
        existing = get_prediction_instruction(instruction_id=instruction_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Instruction {instruction_id} not found")
        
        name = request.get("name")
        description = request.get("description")
        instructions_text = request.get("instructions_text")
        
        success = update_instruction_in_db(
            instruction_id=instruction_id,
            name=name,
            description=description,
            instructions_text=instructions_text
        )
        
        if success:
            updated = get_prediction_instruction(instruction_id=instruction_id)
            return {
                "message": "Instruction template updated successfully",
                "instruction": updated
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update instruction template")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update prediction instruction {instruction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update instruction: {str(e)}")


@app.delete("/api/prediction-instructions/{instruction_id}")
async def delete_prediction_instruction(instruction_id: int):
    """Delete a prediction instruction template"""
    try:
        from db_utils import delete_prediction_instruction as delete_instruction_by_id
        success = delete_instruction_by_id(instruction_id)
        if success:
            return {"message": f"Instruction {instruction_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete instruction template")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prediction instruction {instruction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete instruction: {str(e)}")


# ============================================================================
# Insurance Mappings API Endpoints
# ============================================================================

@app.get("/api/insurance-mappings")
async def get_insurance_mappings(page: int = 1, page_size: int = 50, search: str = None):
    """Get insurance mappings from database with pagination"""
    try:
        from db_utils import get_all_insurance_mappings
        result = get_all_insurance_mappings(page=page, page_size=page_size, search=search)
        return result
    except Exception as e:
        logger.error(f"Failed to get insurance mappings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve mappings: {str(e)}")


@app.get("/api/insurance-mappings/{mapping_id}")
async def get_insurance_mapping(mapping_id: int):
    """Get a specific insurance mapping by ID"""
    try:
        from db_utils import get_insurance_mapping as get_mapping_by_id
        mapping = get_mapping_by_id(mapping_id=mapping_id)
        if mapping:
            return mapping
        else:
            raise HTTPException(status_code=404, detail=f"Mapping {mapping_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get insurance mapping {mapping_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve mapping: {str(e)}")


@app.post("/api/insurance-mappings")
async def create_or_update_insurance_mapping(
    input_code: str = Form(...),
    output_code: str = Form(...),
    description: str = Form(default="")
):
    """Create or update an insurance mapping"""
    try:
        from db_utils import upsert_insurance_mapping
        success = upsert_insurance_mapping(input_code, output_code, description)
        if success:
            return {
                "message": "Insurance mapping saved successfully",
                "input_code": input_code,
                "output_code": output_code,
                "description": description
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save insurance mapping")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save insurance mapping {input_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save mapping: {str(e)}")


@app.put("/api/insurance-mappings/{mapping_id}")
async def update_insurance_mapping(
    mapping_id: int,
    input_code: str = Form(...),
    output_code: str = Form(...),
    description: str = Form(default="")
):
    """Update an existing insurance mapping"""
    try:
        from db_utils import upsert_insurance_mapping
        success = upsert_insurance_mapping(input_code, output_code, description)
        if success:
            return {
                "message": "Insurance mapping updated successfully",
                "input_code": input_code,
                "output_code": output_code,
                "description": description
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update insurance mapping")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update insurance mapping {mapping_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update mapping: {str(e)}")


@app.delete("/api/insurance-mappings/{mapping_id}")
async def delete_insurance_mapping(mapping_id: int):
    """Delete an insurance mapping"""
    try:
        from db_utils import delete_insurance_mapping as delete_mapping_by_id
        success = delete_mapping_by_id(mapping_id)
        if success:
            return {"message": f"Insurance mapping {mapping_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete insurance mapping")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete insurance mapping {mapping_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete mapping: {str(e)}")


@app.get("/api/special-cases-templates")
async def get_special_cases_templates(page: int = 1, page_size: int = 50, search: str = None):
    """Get all special cases templates with pagination and search"""
    try:
        from db_utils import get_all_special_cases_templates
        result = get_all_special_cases_templates(page=page, page_size=page_size, search=search)
        return result
    except Exception as e:
        logger.error(f"Failed to get special cases templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@app.get("/api/special-cases-templates/{template_id}")
async def get_special_cases_template(template_id: int):
    """Get a specific special cases template by ID"""
    try:
        from db_utils import get_special_cases_template as get_template
        template = get_template(template_id=template_id)
        if template:
            return template
        else:
            raise HTTPException(status_code=404, detail="Template not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get special cases template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@app.post("/api/special-cases-templates/upload")
async def upload_special_cases_template(
    csv_file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(default="")
):
    """
    Upload a CSV file and save as a special cases template.
    CSV format: Company name,Mednet code
    """
    try:
        from db_utils import create_special_cases_template
        import pandas as pd
        import io
        
        # Read CSV file
        contents = await csv_file.read()
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(io.BytesIO(contents), dtype=str, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise HTTPException(status_code=400, detail="Could not read CSV file with any standard encoding")
        
        # Validate columns
        if 'Company name' not in df.columns or 'Mednet code' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must have 'Company name' and 'Mednet code' columns"
            )
        
        # Parse mappings
        mappings = []
        for _, row in df.iterrows():
            company_name = str(row['Company name']).strip() if pd.notna(row['Company name']) else ''
            mednet_code = str(row['Mednet code']).strip() if pd.notna(row['Mednet code']) else ''
            
            if company_name and mednet_code:
                mappings.append({
                    'company_name': company_name,
                    'mednet_code': mednet_code
                })
        
        if not mappings:
            raise HTTPException(status_code=400, detail="No valid mappings found in CSV")
        
        # Create template
        template_id = create_special_cases_template(name, mappings, description)
        
        if template_id:
            return {
                "message": "Special cases template created successfully",
                "template_id": template_id,
                "name": name,
                "mappings_count": len(mappings)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create template")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload special cases template: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.put("/api/special-cases-templates/{template_id}")
async def update_special_cases_template_endpoint(
    template_id: int,
    name: str = Form(None),
    description: str = Form(None)
):
    """Update a special cases template's metadata"""
    try:
        from db_utils import update_special_cases_template
        success = update_special_cases_template(template_id, name=name, description=description)
        if success:
            return {"message": f"Template {template_id} updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update template")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update special cases template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")


@app.put("/api/special-cases-templates/{template_id}/mappings")
async def update_special_cases_mappings(template_id: int, mappings: list = Body(...)):
    """Update the mappings in a special cases template"""
    try:
        from db_utils import update_special_cases_template
        success = update_special_cases_template(template_id, mappings=mappings)
        if success:
            return {"message": f"Mappings updated successfully", "count": len(mappings)}
        else:
            raise HTTPException(status_code=500, detail="Failed to update mappings")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update mappings for template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update mappings: {str(e)}")


@app.delete("/api/special-cases-templates/{template_id}")
async def delete_special_cases_template_endpoint(template_id: int):
    """Delete a special cases template"""
    try:
        from db_utils import delete_special_cases_template
        success = delete_special_cases_template(template_id)
        if success:
            return {"message": f"Template {template_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete template")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete special cases template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")




@app.post("/api/insurance-mappings/bulk-import")
async def bulk_import_insurance_mappings(
    csv_file: UploadFile = File(...),
    clear_existing: bool = Form(default=False)
):
    """
    Bulk import insurance mappings from CSV file.
    CSV format: InputValue,OutputValue
    
    Args:
        csv_file: CSV file with InputValue,OutputValue columns
        clear_existing: If true, delete all existing mappings before importing
    """
    try:
        from db_utils import bulk_import_insurance_mappings as bulk_import
        import pandas as pd
        import io
        
        # Read CSV file
        contents = await csv_file.read()
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(io.BytesIO(contents), dtype=str, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise HTTPException(status_code=400, detail="Could not read CSV file with any standard encoding")
        
        # Validate columns
        if 'InputValue' not in df.columns or 'OutputValue' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="CSV must have 'InputValue' and 'OutputValue' columns"
            )
        
        # Prepare mappings data
        mappings_data = []
        for _, row in df.iterrows():
            input_val = str(row['InputValue']).strip() if pd.notna(row['InputValue']) else ''
            output_val = str(row['OutputValue']).strip() if pd.notna(row['OutputValue']) else ''
            
            if input_val and output_val:
                mappings_data.append({
                    'input_code': input_val,
                    'output_code': output_val,
                    'description': f'Imported from CSV'
                })
        
        if not mappings_data:
            raise HTTPException(status_code=400, detail="No valid mappings found in CSV")
        
        # Bulk import
        result = bulk_import(mappings_data, clear_existing=clear_existing)
        
        if result['success']:
            return {
                "message": "Bulk import completed",
                "imported": result['imported'],
                "updated": result['updated'],
                "skipped": result['skipped'],
                "total": result['total'],
                "errors": result['errors']
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Import failed: {'; '.join(result['errors'])}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to bulk import insurance mappings: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


def process_unified_background(
    job_id: str,
    zip_path: str,
    excel_path: str,
    excel_filename: str,
    # Extraction params
    enable_extraction: bool,
    extraction_n_pages: int,
    extraction_model: str,
    extraction_max_workers: int,
    worktracker_group: str,
    worktracker_batch: str,
    extract_csn: bool,
    # CPT params
    enable_cpt: bool,
    cpt_vision_mode: bool,
    cpt_client: str,
    cpt_vision_pages: int,
    cpt_max_workers: int,
    cpt_custom_instructions: str,
    cpt_instruction_template_id: Optional[int],
    # ICD params
    enable_icd: bool,
    icd_n_pages: int,
    icd_max_workers: int,
    icd_custom_instructions: str,
    icd_instruction_template_id: Optional[int]
):
    """Unified background task to run extraction + CPT + ICD prediction"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting unified processing..."
        job.progress = 5
        
        # Create temporary directory for processing
        temp_dir = Path(f"/tmp/unified_{job_id}")
        temp_dir.mkdir(exist_ok=True)
        
        extraction_csv_path = None
        cpt_csv_path = None
        icd_csv_path = None
        
        # ==================== STEP 1: Data Extraction ====================
        if enable_extraction:
            job.message = "Step 1/3: Extracting data from PDFs..."
            job.progress = 10
            logger.info(f"[Unified {job_id}] Starting data extraction")
            
            # Unzip files
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir / "input")
            
            # Copy Excel file to temp directory
            excel_dest = temp_dir / "instructions" / excel_filename
            excel_dest.parent.mkdir(exist_ok=True)
            shutil.copy2(excel_path, excel_dest)
            
            # Set up environment for the processing script
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent / "current")
            env['OPENBLAS_NUM_THREADS'] = '12'
            env['OMP_NUM_THREADS'] = '12'
            env['MKL_NUM_THREADS'] = '12'
            
            # Run the extraction script
            script_path = Path(__file__).parent / "current" / "2-extract_info.py"
            
            job.message = f"Extracting data (first {extraction_n_pages} pages per patient)..."
            job.progress = 15
            
            try:
                cmd = [
                    sys.executable,
                    str(script_path),
                    str(temp_dir / "input"),
                    str(excel_dest),
                    str(extraction_n_pages),
                    str(extraction_max_workers),  # Configurable max_workers
                    extraction_model
                ]
                
                if worktracker_group:
                    cmd.append(worktracker_group)
                else:
                    cmd.append("")
                    
                if worktracker_batch:
                    cmd.append(worktracker_batch)
                else:
                    cmd.append("")
                
                if extract_csn:
                    cmd.append("true")
                else:
                    cmd.append("false")
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=temp_dir, env=env)
                stdout, stderr = process.communicate(timeout=1800)
                
                logger.info(f"[Unified {job_id}] Extraction stdout: {stdout}")
                if stderr:
                    logger.info(f"[Unified {job_id}] Extraction stderr: {stderr}")
                
                if process.returncode != 0:
                    raise Exception(f"Extraction failed: {stderr}")
                    
            except subprocess.TimeoutExpired:
                if process:
                    logger.warning(f"[Unified {job_id}] Extraction timed out")
                    kill_process_tree(process)
                    gc.collect()
                raise Exception("Extraction timed out after 30 minutes")
            
            # Find the extraction output CSV
            extracted_files = list(temp_dir.glob("extracted/combined_patient_data_*.csv"))
            if extracted_files:
                extraction_csv_path = str(extracted_files[0])
                logger.info(f"[Unified {job_id}] Found extraction CSV: {extraction_csv_path}")
            else:
                raise Exception("Extraction completed but no output CSV found")
                
            job.message = "Data extraction completed"
            job.progress = 30
            gc.collect()
        
        # ==================== STEP 2 & 3: CPT and ICD Code Predictions ====================
        # Check if we can run CPT and ICD vision predictions in parallel
        # Both must be enabled, CPT must be in vision mode, and extraction must be completed or disabled
        run_cpt_icd_parallel = (
            enable_cpt and cpt_vision_mode and 
            enable_icd and
            (not enable_extraction or (enable_extraction and extraction_csv_path is not None))
        )
        
        if run_cpt_icd_parallel:
            # Run CPT and ICD vision predictions in parallel for maximum speed
            job.message = "Step 2/2: Predicting CPT and ICD codes in parallel..."
            job.progress = 35
            logger.info(f"[Unified {job_id}] Running CPT and ICD vision predictions in parallel")
            
            # Ensure PDFs are unzipped
            if not (temp_dir / "input").exists():
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir / "input")
            
            # Import prediction functions
            general_coding_path = Path(__file__).parent / "general-coding"
            sys.path.insert(0, str(general_coding_path))
            from predict_general import predict_codes_from_pdfs_api, predict_icd_codes_from_pdfs_api
            
            # Create output paths
            cpt_csv_path = str(temp_dir / "cpt_predictions.csv")
            icd_csv_path = str(temp_dir / "icd_predictions.csv")
            Path(cpt_csv_path).parent.mkdir(exist_ok=True)
            Path(icd_csv_path).parent.mkdir(exist_ok=True)
            
            # Get OpenRouter API key
            api_key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY')
            
            # Thread-safe progress tracking
            import threading
            cpt_completed = [0]
            icd_completed = [0]
            cpt_total = [0]
            icd_total = [0]
            lock = threading.Lock()
            
            def cpt_progress(completed, total, message):
                with lock:
                    cpt_completed[0] = completed
                    cpt_total[0] = total
                    total_completed = cpt_completed[0] + icd_completed[0]
                    total_tasks = cpt_total[0] + icd_total[0]
                    if total_tasks > 0:
                        progress_pct = 35 + int((total_completed / total_tasks) * 50)
                        job.progress = min(progress_pct, 85)
                        job.message = f"CPT: {cpt_completed[0]}/{cpt_total[0]}, ICD: {icd_completed[0]}/{icd_total[0]}"
            
            def icd_progress(completed, total, message):
                with lock:
                    icd_completed[0] = completed
                    icd_total[0] = total
                    total_completed = cpt_completed[0] + icd_completed[0]
                    total_tasks = cpt_total[0] + icd_total[0]
                    if total_tasks > 0:
                        progress_pct = 35 + int((total_completed / total_tasks) * 50)
                        job.progress = min(progress_pct, 85)
                        job.message = f"CPT: {cpt_completed[0]}/{cpt_total[0]}, ICD: {icd_completed[0]}/{icd_total[0]}"
            
            # Run both predictions in parallel using threads
            import threading
            cpt_result = [None]
            icd_result = [None]
            cpt_error = [None]
            icd_error = [None]
            
            def run_cpt():
                try:
                    result = predict_codes_from_pdfs_api(
                        pdf_folder=str(temp_dir / "input"),
                        output_file=cpt_csv_path,
                        n_pages=cpt_vision_pages,
                        model="openai/gpt-5:online",
                        api_key=api_key,
                        max_workers=cpt_max_workers,
                        progress_callback=cpt_progress,
                        custom_instructions=cpt_custom_instructions
                    )
                    cpt_result[0] = result
                except Exception as e:
                    cpt_error[0] = str(e)
            
            def run_icd():
                try:
                    result = predict_icd_codes_from_pdfs_api(
                        pdf_folder=str(temp_dir / "input"),
                        output_file=icd_csv_path,
                        n_pages=icd_n_pages,
                        model="openai/gpt-5:online",
                        api_key=api_key,
                        max_workers=icd_max_workers,
                        progress_callback=icd_progress,
                        custom_instructions=icd_custom_instructions
                    )
                    icd_result[0] = result
                except Exception as e:
                    icd_error[0] = str(e)
            
            # Start both threads
            cpt_thread = threading.Thread(target=run_cpt)
            icd_thread = threading.Thread(target=run_icd)
            
            cpt_thread.start()
            icd_thread.start()
            
            # Wait for both to complete
            cpt_thread.join()
            icd_thread.join()
            
            # Check results
            if cpt_error[0]:
                raise Exception(f"CPT prediction failed: {cpt_error[0]}")
            if icd_error[0]:
                raise Exception(f"ICD prediction failed: {icd_error[0]}")
            if not cpt_result[0]:
                raise Exception("CPT prediction failed")
            if not icd_result[0]:
                raise Exception("ICD prediction failed")
            
            # Verify files were created
            if not os.path.exists(cpt_csv_path):
                raise Exception(f"CPT prediction reported success but file not found: {cpt_csv_path}")
            if not os.path.exists(icd_csv_path):
                raise Exception(f"ICD prediction reported success but file not found: {icd_csv_path}")
            
            logger.info(f"[Unified {job_id}] CPT and ICD predictions completed in parallel")
            job.message = "CPT and ICD predictions completed"
            job.progress = 85
            gc.collect()
        
        elif enable_cpt:
            # ==================== STEP 2: CPT Code Prediction ====================
            job.message = "Step 2/3: Predicting CPT codes..."
            job.progress = 35
            logger.info(f"[Unified {job_id}] Starting CPT prediction (vision_mode={cpt_vision_mode}, client={cpt_client})")
            
            if cpt_vision_mode:
                # Use vision model with PDFs (always uses GPT-5)
                job.message = "Predicting CPT codes from PDF images..."
                
                # Ensure PDFs are unzipped
                if not (temp_dir / "input").exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir / "input")
                
                # Import prediction function
                general_coding_path = Path(__file__).parent / "general-coding"
                sys.path.insert(0, str(general_coding_path))
                from predict_general import predict_codes_from_pdfs_api
                
                # Create output path with .csv extension
                cpt_csv_path = str(temp_dir / "cpt_predictions.csv")
                Path(cpt_csv_path).parent.mkdir(exist_ok=True)
                
                # Get OpenRouter API key
                api_key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY')
                
                # Progress callback
                def cpt_progress(completed, total, message):
                    progress_pct = 35 + int((completed / total) * 20)
                    job.progress = min(progress_pct, 55)
                    job.message = f"CPT prediction: {message}"
                
                # Run CPT prediction with vision (always uses openai/gpt-5:online)
                success = predict_codes_from_pdfs_api(
                    pdf_folder=str(temp_dir / "input"),
                    output_file=cpt_csv_path,
                    n_pages=cpt_vision_pages,
                    model="openai/gpt-5:online",  # Vision mode always uses GPT-5
                    api_key=api_key,
                    max_workers=cpt_max_workers,
                    progress_callback=cpt_progress,
                    custom_instructions=cpt_custom_instructions
                )
                
                if success:
                    # Verify file was actually created
                    if not os.path.exists(cpt_csv_path):
                        raise Exception(f"CPT prediction reported success but file not found: {cpt_csv_path}")
                    logger.info(f"[Unified {job_id}] CPT prediction completed: {cpt_csv_path}")
                else:
                    raise Exception("CPT prediction failed")
                    
            else:
                # Use CSV-based prediction (requires extraction first)
                if not extraction_csv_path:
                    raise Exception("CPT prediction requires data extraction to be enabled when not using vision mode")
                
                job.message = f"Predicting CPT codes from CSV using {cpt_client} model..."
                
                # Determine which prediction method to use based on client
                if cpt_client == "tan-esc":
                    # Use custom trained model
                    custom_coding_path = Path(__file__).parent / "custom-coding"
                    sys.path.insert(0, str(custom_coding_path))
                    from predict import predict_codes_api
                    
                    cpt_csv_path = str(temp_dir / "with_cpt_codes.csv")
                    
                    success = predict_codes_api(
                        input_file=extraction_csv_path,
                        output_file=cpt_csv_path,
                        model_dir=str(custom_coding_path),
                        confidence_threshold=0.5
                    )
                    
                elif cpt_client == "general":
                    # Use OpenAI general model
                    general_coding_path = Path(__file__).parent / "general-coding"
                    sys.path.insert(0, str(general_coding_path))
                    from predict_general import predict_codes_general_api
                    
                    cpt_csv_path = str(temp_dir / "with_cpt_codes.csv")
                    api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
                    
                    def cpt_progress(completed, total, message):
                        progress_pct = 35 + int((completed / total) * 20)
                        job.progress = min(progress_pct, 55)
                        job.message = f"CPT prediction: {message}"
                    
                    success = predict_codes_general_api(
                        input_file=extraction_csv_path,
                        output_file=cpt_csv_path,
                        model="gpt5",
                        api_key=api_key,
                        max_workers=cpt_max_workers,
                        progress_callback=cpt_progress,
                        custom_instructions=cpt_custom_instructions
                    )
                    
                else:
                    # Use client-specific prediction (UNI, SIO-STL, GAP-FIN, APO-UTP)
                    custom_coding_path = Path(__file__).parent / "custom-coding"
                    sys.path.insert(0, str(custom_coding_path))
                    from predict import predict_codes_api
                    
                    cpt_csv_path = str(temp_dir / "with_cpt_codes.csv")
                    
                    # Progress callback for custom model
                    def cpt_progress(completed, total, message):
                        progress_pct = 35 + int((completed / total) * 20)
                        job.progress = min(progress_pct, 55)
                        job.message = f"CPT prediction: {message}"
                    
                    success = predict_codes_api(
                        input_file=extraction_csv_path,
                        output_file=cpt_csv_path,
                        model_dir=str(custom_coding_path),
                        confidence_threshold=0.5
                    )
                
                if not success:
                    raise Exception("CPT prediction failed")
                
                # Verify file was actually created
                if not os.path.exists(cpt_csv_path):
                    raise Exception(f"CPT prediction reported success but file not found: {cpt_csv_path}")
                
                logger.info(f"[Unified {job_id}] CPT prediction completed: {cpt_csv_path}")
            
            job.message = "CPT prediction completed"
            job.progress = 55
            gc.collect()
        
        # ==================== STEP 3: ICD Code Prediction ====================
        if enable_icd and not run_cpt_icd_parallel:
            job.message = "Step 3/3: Predicting ICD codes..."
            job.progress = 60
            logger.info(f"[Unified {job_id}] Starting ICD prediction")
            
            # Ensure PDFs are unzipped
            if not (temp_dir / "input").exists():
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir / "input")
            
            # Import prediction function
            general_coding_path = Path(__file__).parent / "general-coding"
            sys.path.insert(0, str(general_coding_path))
            from predict_general import predict_icd_codes_from_pdfs_api
            
            # Create output path with .csv extension
            icd_csv_path = str(temp_dir / "icd_predictions.csv")
            Path(icd_csv_path).parent.mkdir(exist_ok=True)
            
            # Get OpenRouter API key
            api_key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY')
            
            # Progress callback
            def icd_progress(completed, total, message):
                progress_pct = 60 + int((completed / total) * 25)
                job.progress = min(progress_pct, 85)
                job.message = f"ICD prediction: {message}"
            
            # Run ICD prediction (always uses GPT-5 vision)
            success = predict_icd_codes_from_pdfs_api(
                pdf_folder=str(temp_dir / "input"),
                output_file=icd_csv_path,
                n_pages=icd_n_pages,
                model="openai/gpt-5:online",  # ICD always uses GPT-5 vision
                api_key=api_key,
                max_workers=icd_max_workers,
                progress_callback=icd_progress,
                custom_instructions=icd_custom_instructions
            )
            
            if success:
                # Verify file was actually created
                if not os.path.exists(icd_csv_path):
                    raise Exception(f"ICD prediction reported success but file not found: {icd_csv_path}")
                logger.info(f"[Unified {job_id}] ICD prediction completed: {icd_csv_path}")
            else:
                raise Exception("ICD prediction failed")
            
            job.message = "ICD prediction completed"
            job.progress = 85
            gc.collect()
        
        # ==================== STEP 4: Merge Results ====================
        job.message = "Merging results..."
        job.progress = 90
        logger.info(f"[Unified {job_id}] Merging results")
        
        import pandas as pd
        
        # Determine the base dataframe
        if extraction_csv_path:
            # Verify file exists before reading
            if not os.path.exists(extraction_csv_path):
                raise Exception(f"Extraction CSV file not found: {extraction_csv_path}")
            # Start with extraction data
            base_df = pd.read_csv(extraction_csv_path, dtype=str)
            logger.info(f"[Unified {job_id}] Base: extraction CSV with {len(base_df)} rows and columns: {list(base_df.columns)[:10]}")
        elif cpt_csv_path and cpt_vision_mode:
            # Verify file exists before reading
            if not os.path.exists(cpt_csv_path):
                raise Exception(f"CPT CSV file not found: {cpt_csv_path}")
            # If no extraction but CPT vision mode, use CPT as base
            base_df = pd.read_csv(cpt_csv_path, dtype=str)
            logger.info(f"[Unified {job_id}] Base: CPT CSV with {len(base_df)} rows")
        elif icd_csv_path:
            # Verify file exists before reading
            if not os.path.exists(icd_csv_path):
                raise Exception(f"ICD CSV file not found: {icd_csv_path}")
            # If only ICD, use ICD as base
            base_df = pd.read_csv(icd_csv_path, dtype=str)
            logger.info(f"[Unified {job_id}] Base: ICD CSV with {len(base_df)} rows")
        else:
            raise Exception("No data to merge - at least one processing step must be enabled")
        
        # Merge CPT results if needed
        if enable_cpt and cpt_csv_path:
            # Verify file exists before reading
            if not os.path.exists(cpt_csv_path):
                raise Exception(f"CPT predictions file not found: {cpt_csv_path}")
            
            if cpt_vision_mode and not extraction_csv_path:
                # CPT is already the base
                pass
            elif cpt_vision_mode:
                # Merge CPT vision results with extraction
                cpt_df = pd.read_csv(cpt_csv_path, dtype=str)
                logger.info(f"[Unified {job_id}] CPT vision CSV columns: {list(cpt_df.columns)}")
                logger.info(f"[Unified {job_id}] Base DF columns: {list(base_df.columns)[:10]}")
                
                # Check for both 'Patient Filename' and 'Filename' column names
                filename_col = None
                if 'Patient Filename' in cpt_df.columns:
                    filename_col = 'Patient Filename'
                elif 'Filename' in cpt_df.columns:
                    filename_col = 'Filename'
                
                if filename_col and 'source_file' in base_df.columns:
                    # Extract CPT-related columns (actual column names from predict_codes_from_pdfs_api)
                    cpt_cols_to_merge = ['ASA Code', 'Procedure Code', 'Model Source', 'Tokens Used', 'Cost (USD)', 'Error Message']
                    cpt_cols_available = [col for col in cpt_cols_to_merge if col in cpt_df.columns]
                    
                    # Include filename column for merging
                    merge_cols = [filename_col] + cpt_cols_available
                    cpt_df_merge = cpt_df[merge_cols]
                    
                    # Rename filename column to match for merge
                    cpt_df_merge = cpt_df_merge.rename(columns={filename_col: 'source_file'})
                    
                    base_df = base_df.merge(cpt_df_merge, on='source_file', how='left')
                    logger.info(f"[Unified {job_id}] Merged CPT vision results with columns: {cpt_cols_available}")
                else:
                    logger.warning(f"[Unified {job_id}] Cannot merge CPT: filename_col={filename_col}, source_file in base={('source_file' in base_df.columns)}")
            else:
                # CPT was run on extraction CSV, so it should already have all extraction columns
                base_df = pd.read_csv(cpt_csv_path, dtype=str)
                logger.info(f"[Unified {job_id}] Using CPT CSV as it contains extraction + CPT data")
        
        # Merge ICD results if needed
        if enable_icd and icd_csv_path:
            # Verify file exists before reading
            if not os.path.exists(icd_csv_path):
                raise Exception(f"ICD predictions file not found: {icd_csv_path}")
            icd_df = pd.read_csv(icd_csv_path, dtype=str)
            logger.info(f"[Unified {job_id}] ICD CSV columns: {list(icd_df.columns)}")
            logger.info(f"[Unified {job_id}] Base DF columns before ICD merge: {list(base_df.columns)[:10]}")
            
            # IMPORTANT: Remove any existing ICD columns from extraction data BEFORE merging
            # This prevents pandas from adding _x/_y suffixes
            icd_columns_to_remove = ['ICD1', 'ICD2', 'ICD3', 'ICD4']
            existing_icd_cols = [col for col in icd_columns_to_remove if col in base_df.columns]
            if existing_icd_cols:
                logger.info(f"[Unified {job_id}] Removing existing ICD columns from extraction data: {existing_icd_cols}")
                base_df = base_df.drop(columns=existing_icd_cols, errors='ignore')
            
            # Check for both 'Patient Filename' and 'Filename' column names
            filename_col = None
            if 'Patient Filename' in icd_df.columns:
                filename_col = 'Patient Filename'
            elif 'Filename' in icd_df.columns:
                filename_col = 'Filename'
            
            # Extract ICD columns to merge
            icd_cols_to_merge = ['ICD1', 'ICD2', 'ICD3', 'ICD4', 'Model Source', 'Tokens Used', 'Cost (USD)', 'Error Message']
            icd_cols_available = [col for col in icd_cols_to_merge if col in icd_df.columns]
            
            if filename_col and 'source_file' in base_df.columns:
                # Merge on source_file
                merge_cols = [filename_col] + icd_cols_available
                icd_df_merge = icd_df[merge_cols]
                # Rename filename column to match for merge
                icd_df_merge = icd_df_merge.rename(columns={filename_col: 'source_file'})
                base_df = base_df.merge(icd_df_merge, on='source_file', how='left', suffixes=('', '_drop'))
                # Drop any columns with _drop suffix (shouldn't happen now, but just in case)
                base_df = base_df.drop(columns=[col for col in base_df.columns if col.endswith('_drop')], errors='ignore')
                logger.info(f"[Unified {job_id}] Merged ICD results with columns: {icd_cols_available}")
            elif filename_col and filename_col in base_df.columns:
                # Both have Filename column (or Patient Filename)
                merge_cols = [filename_col] + icd_cols_available
                icd_df_merge = icd_df[merge_cols]
                base_df = base_df.merge(icd_df_merge, on=filename_col, how='left', suffixes=('', '_drop'))
                # Drop any columns with _drop suffix (shouldn't happen now, but just in case)
                base_df = base_df.drop(columns=[col for col in base_df.columns if col.endswith('_drop')], errors='ignore')
                logger.info(f"[Unified {job_id}] Merged ICD results on {filename_col}")
            else:
                logger.warning(f"[Unified {job_id}] Cannot merge ICD: filename_col={filename_col}, source_file in base={('source_file' in base_df.columns)}")
        
        # Clean up any remaining merge suffixes (_x, _y) that might have been created
        # This is a safety measure in case any columns slipped through
        columns_to_clean = []
        for col in base_df.columns:
            if col.endswith('_x') or col.endswith('_y'):
                # Check if there's a base version without suffix
                base_col = col.rsplit('_', 1)[0]
                if base_col in base_df.columns:
                    # Keep the one without suffix, drop the suffixed one
                    columns_to_clean.append(col)
                else:
                    # Rename the suffixed column to remove the suffix
                    base_df = base_df.rename(columns={col: base_col})
        
        if columns_to_clean:
            logger.info(f"[Unified {job_id}] Cleaning up merge suffix columns: {columns_to_clean}")
            base_df = base_df.drop(columns=columns_to_clean, errors='ignore')
        
        # Reorder columns to put ICD columns at the end (after all other data)
        icd_cols = ['ICD1', 'ICD2', 'ICD3', 'ICD4']
        icd_cols_present = [col for col in icd_cols if col in base_df.columns]
        if icd_cols_present:
            # Get all non-ICD columns
            other_cols = [col for col in base_df.columns if col not in icd_cols_present]
            # Reorder: other columns first, then ICD columns
            base_df = base_df[other_cols + icd_cols_present]
            logger.info(f"[Unified {job_id}] Reordered columns to put ICD columns at the end")
        
        # Log final columns before saving
        logger.info(f"[Unified {job_id}] Final merged dataframe has {len(base_df)} rows and columns: {list(base_df.columns)}")
        
        # Save final result
        result_base = Path(f"/tmp/results/{job_id}_unified_result")
        result_base.parent.mkdir(exist_ok=True)
        
        # Save as both CSV and XLSX
        result_csv = f"{result_base}.csv"
        result_xlsx = f"{result_base}.xlsx"
        
        base_df.to_csv(result_csv, index=False)
        
        # Convert to XLSX
        try:
            base_df.to_excel(result_xlsx, index=False, engine='openpyxl')
        except Exception as e:
            logger.warning(f"[Unified {job_id}] Failed to create XLSX: {e}")
            result_xlsx = None
        
        job.result_file = result_csv
        job.result_file_xlsx = result_xlsx
        job.status = "completed"
        job.message = "Unified processing completed successfully"
        job.progress = 100
        
        logger.info(f"[Unified {job_id}] Processing completed: {result_csv}")
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"[Unified {job_id}] Failed to clean up temp directory: {e}")
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        logger.error(f"[Unified {job_id}] Error: {str(e)}")
        job.status = "failed"
        job.error = str(e)
        job.message = f"Processing failed: {str(e)}"
        
        # Clean up on failure
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass
        
        gc.collect()


@app.post("/process-unified")
async def process_unified(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    excel_file: UploadFile = File(None),  # Optional - can use template instead
    template_id: Optional[int] = Form(default=None),  # Use saved template instead of excel file
    # Extraction parameters
    enable_extraction: bool = Form(default=True),
    extraction_n_pages: int = Form(default=2),
    extraction_model: str = Form(default="gemini-2.5-flash"),
    extraction_max_workers: int = Form(default=10),  # Configurable extraction parallelism
    worktracker_group: str = Form(default=""),
    worktracker_batch: str = Form(default=""),
    extract_csn: bool = Form(default=False),
    # CPT parameters
    enable_cpt: bool = Form(default=True),
    cpt_vision_mode: bool = Form(default=False),
    cpt_client: str = Form(default="uni"),  # For non-vision mode
    cpt_vision_pages: int = Form(default=1),  # For vision mode
    cpt_max_workers: int = Form(default=10),  # Increased for better parallelism
    cpt_custom_instructions: str = Form(default=""),
    cpt_instruction_template_id: Optional[int] = Form(default=None),  # For template selection
    # ICD parameters
    enable_icd: bool = Form(default=True),
    icd_n_pages: int = Form(default=1),
    icd_max_workers: int = Form(default=10),  # Increased for better parallelism
    icd_custom_instructions: str = Form(default=""),
    icd_instruction_template_id: Optional[int] = Form(default=None)  # For template selection
):
    """
    Unified endpoint to run data extraction + CPT prediction + ICD prediction
    Results are intelligently merged into a single output file
    """
    try:
        if excel_file and excel_file.filename:
            excel_filename_log = excel_file.filename
        else:
            excel_filename_log = f"template_id={template_id}" if template_id else "None"
        logger.info(f"Received unified processing request - zip: {zip_file.filename}, excel: {excel_filename_log}")
        logger.info(f"Enabled: extraction={enable_extraction}, CPT={enable_cpt}, ICD={enable_icd}")
        
        # Validate at least one step is enabled
        if not (enable_extraction or enable_cpt or enable_icd):
            raise HTTPException(status_code=400, detail="At least one processing step must be enabled")
        
        # Validate file types
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Patient documents must be in a ZIP file")
        
        # Validate that either excel_file or template_id is provided
        if not excel_file and not template_id:
            raise HTTPException(status_code=400, detail="Either an Excel file or a template ID must be provided")
        
        if excel_file and excel_file.filename and not excel_file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Instructions file must be an Excel file")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created unified processing job {job_id}")
        
        # Save uploaded ZIP file
        zip_path = f"/tmp/{job_id}_input.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        
        # Handle excel file or template
        excel_path = None
        excel_filename = None
        
        if template_id:
            # Export template to temporary Excel file
            from db_utils import get_template
            template = get_template(template_id)
            if not template:
                raise HTTPException(status_code=404, detail=f"Template with ID {template_id} not found")
            
            logger.info(f"Using template '{template['name']}' (ID: {template_id}) for extraction")
            
            # Create temporary Excel file from template
            import pandas as pd
            excel_path = f"/tmp/{job_id}_instructions.xlsx"
            excel_filename = f"{template['name']}.xlsx"
            
            # Convert template fields to Excel format (same format as /upload endpoint)
            fields = template['template_data'].get('fields', [])
            
            if not fields:
                raise HTTPException(status_code=400, detail="Template has no field definitions")
            
            # Create DataFrame in the expected format (field names as columns, metadata as rows)
            data = {}
            for field in fields:
                data[field['name']] = [
                    field.get('description', ''),
                    field.get('location', ''),
                    field.get('output_format', ''),
                    'YES' if field.get('priority', False) else 'NO'
                ]
            
            df = pd.DataFrame(data)
            df.to_excel(excel_path, index=False, engine='openpyxl')
            logger.info(f"Exported template to Excel: {excel_path}")
        else:
            # Use uploaded Excel file
            if not excel_file or not excel_file.filename:
                raise HTTPException(status_code=400, detail="Excel file is required when template_id is not provided")
            excel_path = f"/tmp/{job_id}_instructions{Path(excel_file.filename).suffix}"
            excel_filename = excel_file.filename
            with open(excel_path, "wb") as f:
                shutil.copyfileobj(excel_file.file, f)
        
        logger.info(f"Files saved - zip: {zip_path}, excel: {excel_path}")
        
        # Fetch instruction templates if provided
        if cpt_instruction_template_id:
            from db_utils import get_prediction_instruction
            template = get_prediction_instruction(instruction_id=cpt_instruction_template_id)
            if template:
                cpt_custom_instructions = template['instructions_text']
                logger.info(f"Using CPT instruction template '{template['name']}' for unified processing")
        
        if icd_instruction_template_id:
            from db_utils import get_prediction_instruction
            template = get_prediction_instruction(instruction_id=icd_instruction_template_id)
            if template:
                icd_custom_instructions = template['instructions_text']
                logger.info(f"Using ICD instruction template '{template['name']}' for unified processing")
        
        # Start background processing
        background_tasks.add_task(
            process_unified_background,
            job_id=job_id,
            zip_path=zip_path,
            excel_path=excel_path,
            excel_filename=excel_filename,
            enable_extraction=enable_extraction,
            extraction_n_pages=extraction_n_pages,
            extraction_model=extraction_model,
            extraction_max_workers=extraction_max_workers,
            worktracker_group=worktracker_group,
            worktracker_batch=worktracker_batch,
            extract_csn=extract_csn,
            enable_cpt=enable_cpt,
            cpt_vision_mode=cpt_vision_mode,
            cpt_client=cpt_client,
            cpt_vision_pages=cpt_vision_pages,
            cpt_max_workers=cpt_max_workers,
            cpt_custom_instructions=cpt_custom_instructions,
            cpt_instruction_template_id=cpt_instruction_template_id,
            enable_icd=enable_icd,
            icd_n_pages=icd_n_pages,
            icd_max_workers=icd_max_workers,
            icd_custom_instructions=icd_custom_instructions,
            icd_instruction_template_id=icd_instruction_template_id
        )
        
        logger.info(f"Background unified processing task started for job {job_id}")
        
        return {"job_id": job_id, "message": "Unified processing started"}
        
    except Exception as e:
        logger.error(f"Unified processing upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    # This is for local development only
    # Railway will use uvicorn directly via railway.json
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info") 