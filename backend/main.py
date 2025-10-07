import os
import sys
import logging
import gc
import signal

# Set environment variables to limit threading and prevent resource exhaustion
os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_pdfs_background(job_id: str, zip_path: str, excel_path: str, n_pages: int, excel_filename: str, model: str = "gemini-2.5-flash"):
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
            process = subprocess.Popen([
                sys.executable, str(script_path),
                str(temp_dir / "input"),
                str(excel_dest),
                str(n_pages),  # n_pages parameter
                "7",  # max_workers
                model  # model parameter
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=temp_dir, env=env)
            
            # Wait for process with timeout (30 minutes max)
            stdout, stderr = process.communicate(timeout=1800)
            
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
            
            client = genai.Client(
                vertexai=True,
                project="835764687231",
                location="us-central1",
            )
        else:
            # Fallback to API key method
            client = genai.Client(vertexai=True, api_key="AQ.Ab8RN6LnO1TE5YbcCw1PLVGe2qxhL7TuOVtVm3GnhXndEM0nsw")
        
        fallback_client = genai.Client(vertexai=True, api_key="AQ.Ab8RN6LnO1TE5YbcCw1PLVGe2qxhL7TuOVtVm3GnhXndEM0nsw")
        
        # Client model mapping
        client_models = {
            "uni": "projects/835764687231/locations/us-central1/endpoints/1866721154824142848",
            "sio-stl": "projects/835764687231/locations/us-central1/endpoints/1567080046999371776",
            "gap-fin": "projects/835764687231/locations/us-central1/endpoints/9219461073994776576",
            "apo-utp": "projects/835764687231/locations/us-central1/endpoints/1107985563891269632"
        }
        
        custom_model = client_models.get(client, client_models["uni"])
        fallback_model = "gemini-2.5-pro"
        
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

        def get_prediction_and_review(procedure, retries=5):
            """Two-stage prediction: 1) Custom model predicts, 2) Gemini Flash reviews"""
            
            # Stage 1: Get initial prediction from custom model
            prompt = f'For this procedure: "{procedure}" give me the most appropriate anesthesia CPT code'
            last_error = "Unknown error"
            initial_prediction = None
            model_source = None
            failure_reason = None

            # Try custom model first
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
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
                    response = fallback_client.models.generate_content(
                        model=fallback_model,
                        contents=[types.Content(role="user", parts=[{"text": prompt + "Only answer with the code, absolutely nothing else, no other text."}])],
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

        job.message = "Adding predictions to CSV..."
        job.progress = 85
        
        # Insert predictions, model sources, and failure reasons into dataframe
        insert_index = df.columns.get_loc("Procedure Description") + 1
        df.insert(insert_index, "ASA Code", predictions)
        df.insert(insert_index + 1, "Procedure Code", predictions)
        df.insert(insert_index + 2, "Model Source", model_sources)
        df.insert(insert_index + 3, "Base Model Failure Reason", failure_reasons)
        
        # Save result
        result_file = Path(f"/tmp/results/{job_id}_with_codes.csv")
        result_file.parent.mkdir(exist_ok=True)
        df.to_csv(result_file, index=False)
        
        job.result_file = str(result_file)
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
        result_file = Path(f"/tmp/results/{job_id}_with_codes.csv")
        result_file.parent.mkdir(exist_ok=True)
        
        # Run the prediction using the custom model
        # Model files should be in backend/custom-coding/ directory
        model_dir = custom_coding_path
        
        job.message = "Making predictions with TAN-ESC model..."
        job.progress = 50
        
        success = predict_codes_api(
            input_file=csv_path,
            output_file=str(result_file),
            model_dir=str(model_dir),
            confidence_threshold=confidence_threshold
        )
        
        if not success:
            raise Exception("Custom model prediction failed")
        
        job.message = "Prediction complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not result_file.exists():
            raise Exception("Output file was not created")
        
        job.result_file = str(result_file)
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

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    excel_file: UploadFile = File(...),
    n_pages: int = Form(..., ge=1, le=50),  # Validate page count between 1-50
    model: str = Form(default="gemini-2.5-flash")  # Model parameter with default
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

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up a completed job and its files"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    # Clean up result file if it exists
    if job.result_file and os.path.exists(job.result_file):
        try:
            os.unlink(job.result_file)
            logger.info(f"Cleaned up result file: {job.result_file}")
        except Exception as e:
            logger.error(f"Failed to clean up result file: {e}")
    
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
            # Clean up result file if it exists
            if job.result_file and os.path.exists(job.result_file):
                try:
                    file_size = os.path.getsize(job.result_file)
                    os.unlink(job.result_file)
                    total_size_freed += file_size
                    logger.info(f"Cleaned up result file: {job.result_file}")
                except Exception as e:
                    logger.error(f"Failed to clean up result file: {e}")
            
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
    """Upload a CSV file to predict CPT codes for procedures"""
    
    try:
        logger.info(f"Received CPT prediction request - csv: {csv_file.filename}, client: {client}")
        
        # Validate file type
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created CPT prediction job {job_id}")
        
        # Save uploaded CSV
        csv_path = f"/tmp/{job_id}_input.csv"
        
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"CSV saved - path: {csv_path}")
        
        # Start background processing
        background_tasks.add_task(predict_cpt_background, job_id, csv_path, client)
        
        logger.info(f"Background CPT prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": "CSV uploaded and CPT prediction started"}
        
    except Exception as e:
        logger.error(f"CPT prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/predict-cpt-custom")
async def predict_cpt_custom(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5)
):
    """Upload a CSV file to predict CPT codes using custom TAN-ESC model"""
    
    try:
        logger.info(f"Received custom model CPT prediction request - csv: {csv_file.filename}, threshold: {confidence_threshold}")
        
        # Validate file type
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created custom model CPT prediction job {job_id}")
        
        # Save uploaded CSV
        csv_path = f"/tmp/{job_id}_custom_input.csv"
        
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"CSV saved - path: {csv_path}")
        
        # Start background processing with custom model
        background_tasks.add_task(predict_cpt_custom_background, job_id, csv_path, confidence_threshold)
        
        logger.info(f"Background custom model CPT prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": "CSV uploaded and TAN-ESC prediction started"}
        
    except Exception as e:
        logger.error(f"Custom model CPT prediction upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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
    elif job.result_file.endswith('.xlsx'):
        filename = f"converted_data_{job_id}.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        filename = f"processed_data_{job_id}.csv"
        media_type = "text/csv"
    
    # Schedule cleanup after download (in background)
    import asyncio
    asyncio.create_task(cleanup_after_download(job_id))
    
    return FileResponse(
        job.result_file,
        media_type=media_type,
        filename=filename
    )

async def cleanup_after_download(job_id: str):
    """Clean up job after download with a delay"""
    import asyncio
    await asyncio.sleep(30)  # Wait 30 seconds after download
    
    if job_id in job_status:
        job = job_status[job_id]
        
        # Clean up result file
        if job.result_file and os.path.exists(job.result_file):
            try:
                os.unlink(job.result_file)
                logger.info(f"Auto-cleaned up result file: {job.result_file}")
            except Exception as e:
                logger.error(f"Failed to auto-cleanup result file: {e}")
        
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
        output_file = f"/tmp/results/{job_id}_converted.csv"
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Run the conversion
        success = convert_data(csv_path, output_file)
        
        if not success:
            raise Exception("UNI conversion failed")
        
        job.message = "UNI conversion complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not os.path.exists(output_file):
            raise Exception("Output file was not created")
        
        job.result_file = output_file
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
        
        job.message = "Copying additions.csv file..."
        job.progress = 45
        
        # Copy the additions.csv file to the same directory as the temp CSV
        # The convert_data function expects additions.csv to be in the same directory as the input file
        import shutil
        additions_source = Path(__file__).parent / "instructions-conversion" / "additions.csv"
        additions_dest = Path(temp_csv_path).parent / "additions.csv"
        
        if additions_source.exists():
            shutil.copy2(additions_source, additions_dest)
            logger.info(f"Copied additions.csv from {additions_source} to {additions_dest}")
        else:
            logger.warning(f"additions.csv not found at {additions_source}")
        
        job.message = "Converting data using instructions script..."
        job.progress = 60
        
        # Create output file path
        output_file = f"/tmp/results/{job_id}_converted.csv"
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Run the conversion using the convert_instructions script
        success = convert_data(temp_csv_path, output_file)
        
        if not success:
            raise Exception("Instructions conversion failed")
        
        job.message = "Conversion complete, preparing Excel output..."
        job.progress = 85
        
        # Convert back to Excel format
        excel_output_file = f"/tmp/results/{job_id}_converted.xlsx"
        try:
            converted_df = pd.read_csv(output_file)
            converted_df.to_excel(excel_output_file, index=False)
            # Use Excel file as the final result
            job.result_file = excel_output_file
        except Exception as e:
            # If Excel conversion fails, use CSV
            logger.warning(f"Excel conversion failed, using CSV: {str(e)}")
            job.result_file = output_file
        
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

def generate_modifiers_background(job_id: str, csv_path: str):
    """Background task to generate medical modifiers using the modifiers script"""
    job = job_status[job_id]
    
    try:
        job.status = "processing"
        job.message = "Starting modifiers generation..."
        job.progress = 10
        
        # Import the modifiers generation function
        import sys
        sys.path.append(str(Path(__file__).parent / "modifiers"))
        from generate_modifiers import generate_modifiers
        
        job.message = "Generating medical modifiers..."
        job.progress = 50
        
        # Create output file path
        output_file = f"/tmp/results/{job_id}_with_modifiers.csv"
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Run the modifiers generation
        success = generate_modifiers(csv_path, output_file)
        
        if not success:
            raise Exception("Modifiers generation failed")
        
        job.message = "Modifiers generation complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not os.path.exists(output_file):
            raise Exception("Output file was not created")
        
        job.result_file = output_file
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

def predict_insurance_codes_background(job_id: str, data_csv_path: str, special_cases_csv_path: str = None):
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
        
        job.message = "Predicting insurance codes..."
        job.progress = 30
        
        # Create output file path
        output_file = f"/tmp/results/{job_id}_with_insurance_codes.csv"
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Run the prediction
        success = process_insurance_predictions(
            data_csv_path, 
            str(mednet_csv_path), 
            output_file, 
            special_cases_csv_path,
            max_workers=10
        )
        
        if not success:
            raise Exception("Insurance code prediction failed")
        
        job.message = "Prediction complete, preparing results..."
        job.progress = 90
        
        # Verify output file exists
        if not os.path.exists(output_file):
            raise Exception("Output file was not created")
        
        job.result_file = output_file
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
    """Upload a UNI CSV file to convert using the conversion script"""
    
    try:
        logger.info(f"Received UNI conversion request - csv: {csv_file.filename}")
        
        # Validate file type
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created UNI conversion job {job_id}")
        
        # Save uploaded CSV
        csv_path = f"/tmp/{job_id}_uni_input.csv"
        
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"UNI CSV saved - path: {csv_path}")
        
        # Start background processing
        background_tasks.add_task(convert_uni_background, job_id, csv_path)
        
        logger.info(f"Background UNI conversion task started for job {job_id}")
        
        return {"job_id": job_id, "message": "UNI CSV uploaded and conversion started"}
        
    except Exception as e:
        logger.error(f"UNI conversion upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/convert-instructions")
async def convert_instructions(
    background_tasks: BackgroundTasks,
    excel_file: UploadFile = File(...)
):
    """Upload an Excel file to convert using the instructions conversion script"""
    
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

@app.post("/generate-modifiers")
async def generate_modifiers_route(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...)
):
    """Upload a CSV file to generate medical modifiers"""
    
    try:
        logger.info(f"Received modifiers generation request - csv: {csv_file.filename}")
        
        # Validate file type
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created modifiers generation job {job_id}")
        
        # Save uploaded CSV
        csv_path = f"/tmp/{job_id}_modifiers_input.csv"
        
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        logger.info(f"CSV saved - path: {csv_path}")
        
        # Start background processing
        background_tasks.add_task(generate_modifiers_background, job_id, csv_path)
        
        logger.info(f"Background modifiers generation task started for job {job_id}")
        
        return {"job_id": job_id, "message": "CSV uploaded and modifiers generation started"}
        
    except Exception as e:
        logger.error(f"Modifiers generation upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/predict-insurance-codes")
async def predict_insurance_codes(
    background_tasks: BackgroundTasks,
    data_csv: UploadFile = File(...),
    special_cases_csv: UploadFile = File(None)
):
    """Upload data CSV and optional special cases CSV to predict MedNet codes"""
    
    try:
        logger.info(f"Received insurance code prediction request - data: {data_csv.filename}")
        
        # Validate file type
        if not data_csv.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Data file must be a CSV")
        
        if special_cases_csv and not special_cases_csv.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Special cases file must be a CSV")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job = ProcessingJob(job_id)
        job_status[job_id] = job
        
        logger.info(f"Created insurance prediction job {job_id}")
        
        # Save uploaded data CSV
        data_csv_path = f"/tmp/{job_id}_data_input.csv"
        
        with open(data_csv_path, "wb") as f:
            shutil.copyfileobj(data_csv.file, f)
        
        logger.info(f"Data CSV saved - path: {data_csv_path}")
        
        # Save special cases CSV if provided
        special_cases_csv_path = None
        if special_cases_csv:
            special_cases_csv_path = f"/tmp/{job_id}_special_cases.csv"
            with open(special_cases_csv_path, "wb") as f:
                shutil.copyfileobj(special_cases_csv.file, f)
            logger.info(f"Special cases CSV saved - path: {special_cases_csv_path}")
        
        # Start background processing
        background_tasks.add_task(predict_insurance_codes_background, job_id, data_csv_path, special_cases_csv_path)
        
        logger.info(f"Background insurance prediction task started for job {job_id}")
        
        return {"job_id": job_id, "message": "CSV uploaded and insurance code prediction started"}
        
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

if __name__ == "__main__":
    # This is for local development only
    # Railway will use uvicorn directly via railway.json
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info") 