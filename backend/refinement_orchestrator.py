#!/usr/bin/env python3
"""
Refinement orchestrator - coordinates iterative instruction refinement.
Runs CPT refinement first, then ICD refinement, until target accuracy is reached.
"""

import os
import uuid
import logging
import zipfile
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from db_utils import (
    create_refinement_job,
    update_refinement_job,
    get_refinement_job,
    get_prediction_instruction,
    create_prediction_instruction
)
from accuracy_utils import calculate_accuracy, get_error_cases
from instruction_refinement import refine_cpt_instructions, refine_icd_instructions, refine_instructions_focused_mode, preload_pdf_images, clear_pdf_image_cache
from email_utils import send_iteration_report, send_completion_report

logger = logging.getLogger(__name__)


def run_cpt_prediction_only(
    job_id: str,
    extraction_csv_path: str,
    pdf_folder: str,
    cpt_instructions: str,
    cpt_vision_mode: bool,
    cpt_client: str,
    cpt_vision_pages: int,
    cpt_vision_model: str,
    cpt_include_code_list: bool,
    cpt_max_workers: int,
    temp_dir: Path
) -> str:
    """
    Run only CPT prediction (no extraction) using extraction CSV or PDFs.
    Returns path to results CSV.
    """
    import sys
    import os
    
    if cpt_vision_mode:
        # Vision mode: predict from PDFs
        logger.info(f"[Refinement {job_id}] Running CPT vision prediction...")
        general_coding_path = Path(__file__).parent / "general-coding"
        sys.path.insert(0, str(general_coding_path))
        from predict_general import predict_codes_from_pdfs_api, is_gemini_model
        
        using_gemini = is_gemini_model(cpt_vision_model)
        api_key = os.getenv("GOOGLE_API_KEY") if using_gemini else os.getenv("OPENROUTER_API_KEY")
        
        cpt_csv_path = str(temp_dir / "cpt_predictions.csv")
        success = predict_codes_from_pdfs_api(
            pdf_folder=pdf_folder,
            output_file=cpt_csv_path,
            n_pages=cpt_vision_pages,
            model=cpt_vision_model,
            api_key=api_key,
            max_workers=cpt_max_workers,
            progress_callback=None,
            custom_instructions=cpt_instructions,
            include_code_list=cpt_include_code_list
        )
        
        if not success or not os.path.exists(cpt_csv_path):
            raise Exception("CPT vision prediction failed")
        
        return cpt_csv_path
    else:
        # Non-vision mode: predict from extraction CSV
        logger.info(f"[Refinement {job_id}] Running CPT CSV-based prediction...")
        
        if cpt_client == "general":
            general_coding_path = Path(__file__).parent / "general-coding"
            sys.path.insert(0, str(general_coding_path))
            from predict_general import predict_codes_general_api
            
            cpt_csv_path = str(temp_dir / "cpt_predictions.csv")
            api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
            
            success = predict_codes_general_api(
                input_file=extraction_csv_path,
                output_file=cpt_csv_path,
                model="gpt5",
                api_key=api_key,
                max_workers=cpt_max_workers,
                progress_callback=None,
                custom_instructions=cpt_instructions
            )
        else:
            # Custom model (tan-esc, uni, etc.)
            custom_coding_path = Path(__file__).parent / "custom-coding"
            sys.path.insert(0, str(custom_coding_path))
            from predict import predict_codes_api
            
            cpt_csv_path = str(temp_dir / "cpt_predictions.csv")
            success = predict_codes_api(
                input_file=extraction_csv_path,
                output_file=cpt_csv_path,
                model_dir=str(custom_coding_path),
                confidence_threshold=0.5
            )
        
        if not success or not os.path.exists(cpt_csv_path):
            raise Exception("CPT CSV prediction failed")
        
        return cpt_csv_path


def run_icd_prediction_only(
    job_id: str,
    pdf_folder: str,
    icd_instructions: str,
    icd_n_pages: int,
    icd_vision_model: str,
    icd_max_workers: int,
    temp_dir: Path
) -> str:
    """
    Run only ICD prediction (no extraction) using PDFs.
    Returns path to results CSV.
    """
    import sys
    import os
    
    logger.info(f"[Refinement {job_id}] Running ICD prediction...")
    general_coding_path = Path(__file__).parent / "general-coding"
    sys.path.insert(0, str(general_coding_path))
    from predict_general import predict_icd_codes_from_pdfs_api, is_gemini_model
    
    using_gemini = is_gemini_model(icd_vision_model)
    api_key = os.getenv("GOOGLE_API_KEY") if using_gemini else (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    
    icd_csv_path = str(temp_dir / "icd_predictions.csv")
    success = predict_icd_codes_from_pdfs_api(
        pdf_folder=pdf_folder,
        output_file=icd_csv_path,
        n_pages=icd_n_pages,
        model=icd_vision_model,
        api_key=api_key,
        max_workers=icd_max_workers,
        progress_callback=None,
        custom_instructions=icd_instructions
    )
    
    if not success or not os.path.exists(icd_csv_path):
        raise Exception("ICD prediction failed")
    
    return icd_csv_path


def run_refinement_job(
    job_id: str,
    zip_path: str,
    excel_path: str,
    excel_filename: str,
    ground_truth_path: str,
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
    cpt_vision_model: str,
    cpt_include_code_list: bool,
    cpt_max_workers: int,
    original_cpt_template_id: int,
    # ICD params
    enable_icd: bool,
    icd_n_pages: int,
    icd_vision_model: str,
    icd_max_workers: int,
    original_icd_template_id: int,
    # Refinement params
    target_cpt_accuracy: float,
    target_icd_accuracy: float,
    max_iterations: int,
    notification_email: str,
    refinement_guidance: Optional[str] = None,
    refinement_mode: str = "batch",  # "batch" or "focused"
    batch_size: int = 10,  # Number of errors per batch in batch mode
    refinement_model: str = "gemini-3-flash-preview"  # Model to use for refinement
):
    """
    Main refinement job orchestrator.
    Runs iterative refinement for CPT first, then ICD.
    """
    try:
        logger.info(f"[Refinement {job_id}] Starting refinement job")
        
        # Create refinement job record
        create_refinement_job(
            job_id=job_id,
            user_email=notification_email,
            original_cpt_template_id=original_cpt_template_id,
            original_icd_template_id=original_icd_template_id,
            current_cpt_template_id=original_cpt_template_id,
            current_icd_template_id=original_icd_template_id,
            phase="cpt",
            status="running"
        )
        
        # Extract ZIP to get PDF mapping
        temp_dir = Path(f"/tmp/refinement_{job_id}")
        temp_dir.mkdir(exist_ok=True)
        
        # Extract ZIP for processing
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir / "input")
        
        # PDF mapping will be built from extraction results CSV (after first iteration)
        # This ensures account IDs match between predictions CSV and PDF mapping
        pdf_mapping = {}  # account_id -> pdf_path
        pdf_folder = temp_dir / "input"
        
        logger.info(f"[Refinement {job_id}] PDF mapping will be built from extraction results")
        
        # Import unified processing function (lazy import to avoid circular dependencies)
        # We'll import it when needed inside the loop
        
        # Get original templates (only if enabled)
        original_cpt_template = None
        original_icd_template = None
        
        if enable_cpt:
            original_cpt_template = get_prediction_instruction(instruction_id=original_cpt_template_id)
            if not original_cpt_template:
                raise Exception("CPT template not found")
        
        if enable_icd:
            original_icd_template = get_prediction_instruction(instruction_id=original_icd_template_id)
            if not original_icd_template:
                raise Exception("ICD template not found")
        
        if not enable_cpt and not enable_icd:
            raise Exception("At least one refinement type (CPT or ICD) must be enabled")
        
        current_cpt_template_id = original_cpt_template_id if enable_cpt else None
        current_icd_template_id = original_icd_template_id if enable_icd else None
        best_cpt_template_id = original_cpt_template_id if enable_cpt else None
        best_icd_template_id = original_icd_template_id if enable_icd else None
        best_cpt_accuracy = 0.0
        best_icd1_accuracy = 0.0
        final_results_csv_path = None  # Track final results CSV for detailed email
        
        # Track instruction history for learning from previous attempts
        cpt_instruction_history = []
        icd_instruction_history = []
        
        # ==================== STEP 0: Run extraction ONCE and cache PDF images ====================
        extraction_csv_path = None
        pdf_image_cache = {}  # account_id -> list of base64 images
        
        if enable_extraction:
            logger.info(f"[Refinement {job_id}] Running extraction once at the start...")
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from main import ProcessingJob, job_status, process_unified_background
            
            extraction_job_id = f"{job_id}_extraction"
            extraction_zip_path = f"/tmp/{extraction_job_id}_input.zip"
            shutil.copy2(zip_path, extraction_zip_path)
            
            extraction_job = ProcessingJob(extraction_job_id)
            job_status[extraction_job_id] = extraction_job
            
            # Run only extraction (no CPT/ICD)
            process_unified_background(
                job_id=extraction_job_id,
                zip_path=extraction_zip_path,
                excel_path=excel_path,
                excel_filename=excel_filename,
                enable_extraction=True,
                extraction_n_pages=extraction_n_pages,
                extraction_model=extraction_model,
                extraction_max_workers=extraction_max_workers,
                worktracker_group=worktracker_group,
                worktracker_batch=worktracker_batch,
                extract_csn=extract_csn,
                enable_cpt=False,
                cpt_vision_mode=False,
                cpt_client=cpt_client,
                cpt_vision_pages=cpt_vision_pages,
                cpt_vision_model=cpt_vision_model,
                cpt_include_code_list=cpt_include_code_list,
                cpt_max_workers=cpt_max_workers,
                cpt_custom_instructions="",
                cpt_instruction_template_id=None,
                enable_icd=False,
                icd_n_pages=icd_n_pages,
                icd_vision_model=icd_vision_model,
                icd_max_workers=icd_max_workers,
                icd_custom_instructions="",
                icd_instruction_template_id=None
            )
            
            # Wait for extraction to complete
            import time
            while extraction_job.status == "processing":
                time.sleep(2)
            
            if extraction_job.status == "failed":
                raise Exception(f"Extraction failed: {extraction_job.error}")
            
            extraction_csv_path = extraction_job.result_file
            if not extraction_csv_path or not os.path.exists(extraction_csv_path):
                raise Exception("Extraction CSV not found")
            
            logger.info(f"[Refinement {job_id}] Extraction completed: {extraction_csv_path}")
            
            # Build PDF mapping from extraction CSV
            logger.info(f"[Refinement {job_id}] Building PDF mapping from extraction results CSV")
            import pandas as pd
            try:
                results_df = pd.read_csv(extraction_csv_path, dtype=str)
                
                account_id_col = None
                source_file_col = None
                
                for col in results_df.columns:
                    col_upper = col.upper().strip()
                    if col_upper in ['ACCOUNT #', 'ACCOUNTID', 'ACCOUNT ID', 'ACCOUNT', 'ID', 'ACC. #', 'ACC #']:
                        account_id_col = col
                    elif col_upper in ['SOURCE_FILE', 'SOURCE FILE', 'PATIENT FILENAME', 'FILENAME']:
                        source_file_col = col
                
                if account_id_col and source_file_col:
                    for idx, row in results_df.iterrows():
                        account_id = str(row[account_id_col]).strip()
                        source_file = str(row[source_file_col]).strip()
                        
                        pdf_path = pdf_folder / source_file
                        if pdf_path.exists():
                            pdf_mapping[account_id] = str(pdf_path)
                    
                    logger.info(f"[Refinement {job_id}] Built PDF mapping with {len(pdf_mapping)} account IDs")
                else:
                    logger.warning(f"[Refinement {job_id}] Could not find account ID or source file columns")
            except Exception as e:
                logger.error(f"[Refinement {job_id}] Error building PDF mapping: {e}")
            
            # Pre-load PDF images into cache
            logger.info(f"[Refinement {job_id}] Pre-loading PDF images into cache...")
            pdf_image_cache = preload_pdf_images(pdf_mapping, max_pages=None)
            logger.info(f"[Refinement {job_id}] Cached images for {len(pdf_image_cache)} PDFs")
        
        # ==================== PHASE 1: CPT REFINEMENT ====================
        if enable_cpt:
            logger.info(f"[Refinement {job_id}] Starting CPT refinement phase")
            update_refinement_job(job_id, phase="cpt", status="running")
        
        cpt_iteration = 0
        previous_cpt_accuracy = None
        
        # Add original instructions to history (before first iteration)
        if enable_cpt:
            original_cpt_template = get_prediction_instruction(instruction_id=original_cpt_template_id)
            cpt_instruction_history.append({
                "instructions": original_cpt_template['instructions_text'],
                "accuracy": None,  # Will be filled after first prediction
                "iteration": 0  # Original
            })
        
        while enable_cpt and cpt_iteration < max_iterations:
            cpt_iteration += 1
            logger.info(f"[Refinement {job_id}] CPT Iteration {cpt_iteration}/{max_iterations}")
            
            # Get current CPT template
            current_cpt_template = get_prediction_instruction(instruction_id=current_cpt_template_id)
            cpt_instructions = current_cpt_template['instructions_text']
            
            # Run ONLY CPT prediction (reuse extraction CSV and cached PDFs)
            logger.info(f"[Refinement {job_id}] Running CPT prediction only (iteration {cpt_iteration})...")
            
            try:
                if not extraction_csv_path:
                    raise Exception("Extraction CSV not available - extraction must run first")
                
                # Run CPT prediction only
                cpt_csv_path = run_cpt_prediction_only(
                    job_id=job_id,
                    extraction_csv_path=extraction_csv_path,
                    pdf_folder=str(pdf_folder),
                    cpt_instructions=cpt_instructions,
                    cpt_vision_mode=cpt_vision_mode,
                    cpt_client=cpt_client,
                    cpt_vision_pages=cpt_vision_pages,
                    cpt_vision_model=cpt_vision_model,
                    cpt_include_code_list=cpt_include_code_list,
                    cpt_max_workers=cpt_max_workers,
                    temp_dir=temp_dir
                )
                
                # Merge CPT predictions with extraction CSV to create final results
                import pandas as pd
                extraction_df = pd.read_csv(extraction_csv_path, dtype=str)
                cpt_df = pd.read_csv(cpt_csv_path, dtype=str)
                
                # Handle merging based on vision mode vs CSV mode
                if cpt_vision_mode:
                    # Vision mode: CPT CSV has 'Patient Filename', extraction has 'source_file'
                    # Match by filename (same logic as unified processing)
                    filename_col_cpt = None
                    if 'Patient Filename' in cpt_df.columns:
                        filename_col_cpt = 'Patient Filename'
                    elif 'Filename' in cpt_df.columns:
                        filename_col_cpt = 'Filename'
                    
                    source_file_col_ext = None
                    if 'source_file' in extraction_df.columns:
                        source_file_col_ext = 'source_file'
                    else:
                        for col in extraction_df.columns:
                            if col.upper().strip() in ['SOURCE_FILE', 'SOURCE FILE', 'PATIENT FILENAME', 'FILENAME']:
                                source_file_col_ext = col
                                break
                    
                    if filename_col_cpt and source_file_col_ext:
                        # Extract CPT columns to merge
                        cpt_cols_to_merge = ['ASA Code', 'Procedure Code', 'Model Source', 'Error Message']
                        if 'CPT' in cpt_df.columns:
                            cpt_cols_to_merge.append('CPT')
                        cpt_cols_available = [col for col in cpt_cols_to_merge if col in cpt_df.columns]
                        
                        # Include filename column for merging
                        merge_cols = [filename_col_cpt] + cpt_cols_available
                        cpt_df_merge = cpt_df[merge_cols]
                        
                        # Rename filename column to match for merge
                        cpt_df_merge = cpt_df_merge.rename(columns={filename_col_cpt: source_file_col_ext})
                        
                        results_df = extraction_df.merge(cpt_df_merge, on=source_file_col_ext, how='left')
                        logger.info(f"[Refinement {job_id}] Merged CPT vision results with columns: {cpt_cols_available}")
                    else:
                        # Fallback: assume same order
                        logger.warning(f"[Refinement {job_id}] Cannot merge CPT properly, using fallback")
                        results_df = extraction_df.copy()
                        if 'ASA Code' in cpt_df.columns:
                            results_df['ASA Code'] = cpt_df['ASA Code'].values[:len(results_df)]
                        if 'CPT' in cpt_df.columns:
                            results_df['CPT'] = cpt_df['CPT'].values[:len(results_df)]
                else:
                    # CSV mode: Both have Account ID, merge on that
                    account_id_col_ext = None
                    account_id_col_cpt = None
                    
                    for col in extraction_df.columns:
                        if col.upper().strip() in ['ACCOUNT #', 'ACCOUNTID', 'ACCOUNT ID', 'ACCOUNT', 'ID', 'ACC. #', 'ACC #']:
                            account_id_col_ext = col
                            break
                    
                    for col in cpt_df.columns:
                        if col.upper().strip() in ['ACCOUNT #', 'ACCOUNTID', 'ACCOUNT ID', 'ACCOUNT', 'ID', 'ACC. #', 'ACC #']:
                            account_id_col_cpt = col
                            break
                    
                    if account_id_col_ext and account_id_col_cpt:
                        # Merge CPT predictions into extraction CSV
                        cpt_cols_to_merge = ['CPT', 'ASA Code']
                        cpt_cols_available = [col for col in cpt_cols_to_merge if col in cpt_df.columns]
                        
                        results_df = extraction_df.merge(
                            cpt_df[[account_id_col_cpt] + cpt_cols_available].rename(columns={account_id_col_cpt: account_id_col_ext}),
                            on=account_id_col_ext,
                            how='left'
                        )
                    else:
                        # Fallback: assume same order
                        results_df = extraction_df.copy()
                        if 'CPT' in cpt_df.columns:
                            results_df['CPT'] = cpt_df['CPT'].values[:len(results_df)]
                        if 'ASA Code' in cpt_df.columns:
                            results_df['ASA Code'] = cpt_df['ASA Code'].values[:len(results_df)]
                
                # Save merged results
                results_csv_path = str(temp_dir / f"cpt_iter_{cpt_iteration}_results.csv")
                results_df.to_csv(results_csv_path, index=False)
                logger.info(f"[Refinement {job_id}] CPT prediction completed: {results_csv_path}")
                # Track final results CSV (last iteration) - update if CPT is enabled
                if enable_cpt:
                    final_results_csv_path = results_csv_path
                
            except Exception as e:
                logger.error(f"[Refinement {job_id}] CPT prediction failed: {e}")
                update_refinement_job(job_id, status="failed", error_message=str(e))
                return
            
            # Calculate accuracy
            try:
                cpt_accuracy, icd1_accuracy, all_errors = calculate_accuracy(
                    predictions_path=results_csv_path,
                    ground_truth_path=ground_truth_path,
                    pdf_mapping=pdf_mapping
                )
                
                # Update best accuracy (only if CPT was calculated)
                if enable_cpt and cpt_accuracy is not None:
                    # Update accuracy for the instructions we just tested
                    # Find the history entry for these instructions (by iteration number)
                    history_entry = None
                    for entry in cpt_instruction_history:
                        if entry['iteration'] == cpt_iteration - 1:  # Instructions from previous iteration that we just tested
                            history_entry = entry
                            break
                    
                    if history_entry is not None:
                        # Update the accuracy for the instructions we just tested
                        history_entry['accuracy'] = cpt_accuracy
                        logger.info(f"[Refinement {job_id}] Updated CPT history for iteration {cpt_iteration - 1} with accuracy {cpt_accuracy:.2%}")
                    elif cpt_iteration == 1:
                        # First iteration: update original (iteration 0) with its accuracy
                        cpt_instruction_history[0]['accuracy'] = cpt_accuracy
                        logger.info(f"[Refinement {job_id}] Updated original CPT instructions (iteration 0) with accuracy {cpt_accuracy:.2%}")
                    
                    if cpt_accuracy > best_cpt_accuracy:
                        best_cpt_accuracy = cpt_accuracy
                        best_cpt_template_id = current_cpt_template_id
                    
                    logger.info(f"[Refinement {job_id}] CPT Accuracy: {cpt_accuracy:.2%}")
                    
                    # Update job status
                    update_refinement_job(
                        job_id,
                        iteration=cpt_iteration,
                        cpt_accuracy=cpt_accuracy,
                        best_cpt_accuracy=best_cpt_accuracy,
                        current_cpt_template_id=current_cpt_template_id,
                        best_cpt_template_id=best_cpt_template_id
                    )
                    
                    # Get error cases for this iteration
                    cpt_errors = get_error_cases(
                        predictions_path=results_csv_path,
                        ground_truth_path=ground_truth_path,
                        error_type='cpt',
                        pdf_mapping=pdf_mapping,
                        limit=10
                    )
                    
                    # Determine status
                    if cpt_accuracy >= target_cpt_accuracy:
                        status = "target_reached"
                        logger.info(f"[Refinement {job_id}] CPT target accuracy reached: {cpt_accuracy:.2%} >= {target_cpt_accuracy:.2%}")
                    elif cpt_iteration >= max_iterations:
                        status = "max_iterations"
                        logger.info(f"[Refinement {job_id}] CPT max iterations reached")
                    else:
                        status = "continuing"
                    
                    # Send email report
                    send_iteration_report(
                        to_email=notification_email,
                        iteration=cpt_iteration,
                        phase="cpt",
                        previous_accuracy=previous_cpt_accuracy,
                        current_accuracy=cpt_accuracy,
                        best_accuracy=best_cpt_accuracy,
                        old_instructions=cpt_instructions,
                        new_instructions=cpt_instructions,  # Will be updated below if refining
                        error_cases=cpt_errors,
                        status=status
                    )
                    
                    # Check if we should continue
                    if cpt_accuracy >= target_cpt_accuracy or cpt_iteration >= max_iterations:
                        break
                    
                    # Refine instructions using Gemini
                    if refinement_mode == "focused":
                        # FOCUSED MODE: Process one error at a time
                        logger.info(f"[Refinement {job_id}] Refining CPT instructions in FOCUSED MODE (one error at a time)...")
                        
                        # Get ALL error cases (not just 10) for focused mode
                        all_cpt_errors = get_error_cases(
                            predictions_path=results_csv_path,
                            ground_truth_path=ground_truth_path,
                            error_type='cpt',
                            pdf_mapping=pdf_mapping,
                            limit=None  # Get all errors
                        )
                        
                        logger.info(f"[Refinement {job_id}] Processing {len(all_cpt_errors)} CPT errors one by one...")
                        
                        # Start with current instructions
                        working_instructions = cpt_instructions
                        all_reasonings = []
                        rule_type_counts = {"general": 0, "hardcoded": 0}
                        
                        # Process each error one at a time
                        for error_idx, error_case in enumerate(all_cpt_errors, 1):
                            account_id = error_case.get('account_id', 'N/A')
                            pdf_path = pdf_mapping.get(account_id, error_case.get('pdf_path', ''))
                            
                            logger.info(f"[Refinement {job_id}] Processing error {error_idx}/{len(all_cpt_errors)}: Predicted '{error_case.get('predicted', 'N/A')}' -> Expected '{error_case.get('expected', 'N/A')}'")
                            
                            # Refine with this single error
                            cached_images = pdf_image_cache.get(account_id) if pdf_image_cache else None
                            improved_instructions, reasoning, rule_type = refine_instructions_focused_mode(
                                current_instructions=working_instructions,
                                single_error_case=error_case,
                                pdf_path=pdf_path,
                                instruction_type="cpt",
                                model=refinement_model,
                                user_guidance=refinement_guidance
                            )
                            
                            if not improved_instructions:
                                logger.warning(f"[Refinement {job_id}] Failed to refine for error {error_idx}: {reasoning}")
                                continue
                            
                            # Update working instructions for next iteration
                            working_instructions = improved_instructions
                            all_reasonings.append(f"Error {error_idx} (Account {account_id}): {reasoning}")
                            
                            if rule_type:
                                rule_type_counts[rule_type] = rule_type_counts.get(rule_type, 0) + 1
                        
                        # Final improved instructions after processing all errors
                        improved_instructions = working_instructions
                        reasoning = f"Processed {len(all_cpt_errors)} errors. Rules added: {rule_type_counts['general']} general, {rule_type_counts['hardcoded']} hardcoded.\n\n" + "\n".join(all_reasonings[:5])  # Show first 5 reasonings
                        
                        if not improved_instructions:
                            logger.error(f"[Refinement {job_id}] Failed to refine CPT instructions in focused mode")
                            update_refinement_job(job_id, status="failed", error_message="Instruction refinement failed in focused mode")
                            return
                    else:
                        # BATCH MODE: Process all errors in batches, refining after each batch
                        logger.info(f"[Refinement {job_id}] Refining CPT instructions in BATCH MODE (processing all errors in batches)...")
                        
                        # Get ALL error cases
                        all_cpt_errors = get_error_cases(
                            predictions_path=results_csv_path,
                            ground_truth_path=ground_truth_path,
                            error_type='cpt',
                            pdf_mapping=pdf_mapping,
                            limit=None  # Get all errors
                        )
                        
                        logger.info(f"[Refinement {job_id}] Processing {len(all_cpt_errors)} CPT errors in batches of {batch_size}...")
                        
                        # Start with current instructions
                        working_instructions = cpt_instructions
                        all_reasonings = []
                        
                        # Process errors in batches
                        for batch_start in range(0, len(all_cpt_errors), batch_size):
                            batch_end = min(batch_start + batch_size, len(all_cpt_errors))
                            batch_errors = all_cpt_errors[batch_start:batch_end]
                            batch_num = (batch_start // batch_size) + 1
                            total_batches = (len(all_cpt_errors) + batch_size - 1) // batch_size
                            
                            logger.info(f"[Refinement {job_id}] Processing batch {batch_num}/{total_batches} (errors {batch_start+1}-{batch_end} of {len(all_cpt_errors)})...")
                            
                            # Refine with this batch of errors (pass instruction history)
                            improved_instructions, reasoning = refine_cpt_instructions(
                                current_instructions=working_instructions,
                                error_cases=batch_errors,
                                pdf_mapping=pdf_mapping,
                                model=refinement_model,
                                user_guidance=refinement_guidance,
                                pdf_image_cache=pdf_image_cache,
                                instruction_history=cpt_instruction_history
                            )
                            
                            if not improved_instructions:
                                logger.warning(f"[Refinement {job_id}] Failed to refine batch {batch_num}: {reasoning}")
                                continue
                            
                            # Update working instructions for next batch
                            working_instructions = improved_instructions
                            all_reasonings.append(f"Batch {batch_num}: {reasoning}")
                        
                        # Final improved instructions after processing all batches
                        improved_instructions = working_instructions
                        reasoning = f"Processed {len(all_cpt_errors)} errors in {total_batches} batches.\n\n" + "\n".join(all_reasonings)
                        
                        if not improved_instructions:
                            logger.error(f"[Refinement {job_id}] Failed to refine CPT instructions in batch mode")
                            update_refinement_job(job_id, status="failed", error_message=f"Instruction refinement failed: {reasoning}")
                            return
                    
                    # Create new template version with naming pattern: base_name (iteration)
                    # Extract base name (remove any existing iteration suffix like " (1)", " (2)", etc.)
                    base_name = original_cpt_template['name']
                    # Remove pattern like " (1)", " (2)" if it exists
                    base_name = re.sub(r'\s*\(\d+\)\s*$', '', base_name)
                    new_template_name = f"{base_name} ({cpt_iteration})"
                    new_template_id = create_prediction_instruction(
                        name=new_template_name,
                        instruction_type="cpt",
                        instructions_text=improved_instructions,
                        description=f"Auto-refined iteration {cpt_iteration} of {original_cpt_template['name']}"
                    )
                    
                    if not new_template_id:
                        raise Exception("Failed to create new CPT template")
                    
                    logger.info(f"[Refinement {job_id}] Created new CPT template: {new_template_name} (ID: {new_template_id})")
                    
                    # Add improved instructions to history AFTER refinement (accuracy will be filled in next iteration when tested)
                    cpt_instruction_history.append({
                        "instructions": improved_instructions,
                        "accuracy": None,  # Will be updated in next iteration when these instructions are tested
                        "iteration": cpt_iteration
                    })
                    logger.info(f"[Refinement {job_id}] Added CPT iteration {cpt_iteration} to history (accuracy will be measured in next iteration)")
                    
                    # Update current template
                    current_cpt_template_id = new_template_id
                    previous_cpt_accuracy = cpt_accuracy
                    
                    # Send updated email with new instructions
                    send_iteration_report(
                        to_email=notification_email,
                        iteration=cpt_iteration,
                        phase="cpt",
                        previous_accuracy=previous_cpt_accuracy,
                        current_accuracy=cpt_accuracy,
                        best_accuracy=best_cpt_accuracy,
                        old_instructions=cpt_instructions,
                        new_instructions=improved_instructions,
                        error_cases=cpt_errors,
                        status=status,
                        gemini_reasoning=reasoning
                    )
                else:
                    # If CPT is disabled, just break after first iteration
                    logger.info(f"[Refinement {job_id}] CPT refinement disabled, skipping CPT phase")
                    break
                
            except Exception as e:
                logger.error(f"[Refinement {job_id}] Error in CPT iteration {cpt_iteration}: {e}")
                update_refinement_job(job_id, status="failed", error_message=str(e))
                return
        
        # ==================== PHASE 2: ICD REFINEMENT ====================
        if enable_icd:
            logger.info(f"[Refinement {job_id}] Starting ICD refinement phase")
            update_refinement_job(job_id, phase="icd", status="running")
            
            icd_iteration = 0
            previous_icd_accuracy = None
            
            # Add original ICD instructions to history (before first iteration)
            original_icd_template = get_prediction_instruction(instruction_id=original_icd_template_id)
            icd_instruction_history.append({
                "instructions": original_icd_template['instructions_text'],
                "accuracy": None,  # Will be filled after first prediction
                "iteration": 0  # Original
            })
            
            # Use best CPT template for ICD phase (if CPT was enabled)
            if enable_cpt:
                current_cpt_template_id = best_cpt_template_id
            else:
                # If CPT is disabled, use original CPT template (or None if not needed)
                current_cpt_template_id = original_cpt_template_id if original_cpt_template_id else None
            
            while icd_iteration < max_iterations:
                icd_iteration += 1
                logger.info(f"[Refinement {job_id}] ICD Iteration {icd_iteration}/{max_iterations}")
                
                # Get current ICD template
                current_icd_template = get_prediction_instruction(instruction_id=current_icd_template_id)
                icd_instructions = current_icd_template['instructions_text']
                
                # Run ONLY ICD prediction (reuse extraction CSV and cached PDFs)
                logger.info(f"[Refinement {job_id}] Running ICD prediction only (iteration {icd_iteration})...")
                
                try:
                    if not extraction_csv_path:
                        raise Exception("Extraction CSV not available - extraction must run first")
                    
                    # Run ICD prediction only
                    icd_csv_path = run_icd_prediction_only(
                        job_id=job_id,
                        pdf_folder=str(pdf_folder),
                        icd_instructions=icd_instructions,
                        icd_n_pages=icd_n_pages,
                        icd_vision_model=icd_vision_model,
                        icd_max_workers=icd_max_workers,
                        temp_dir=temp_dir
                    )
                    
                    # Merge ICD predictions with extraction CSV (and best CPT if available)
                    import pandas as pd
                    extraction_df = pd.read_csv(extraction_csv_path, dtype=str)
                    icd_df = pd.read_csv(icd_csv_path, dtype=str)
                    
                    # Start with extraction CSV
                    results_df = extraction_df.copy()
                    
                    # Add best CPT predictions if available
                    if enable_cpt and best_cpt_template_id:
                        # Get best CPT predictions from previous iteration
                        best_cpt_csv = str(temp_dir / f"cpt_iter_best_results.csv")
                        if os.path.exists(best_cpt_csv):
                            best_cpt_df = pd.read_csv(best_cpt_csv, dtype=str)
                            # Find account ID columns
                            account_id_col_ext = None
                            account_id_col_cpt = None
                            for col in results_df.columns:
                                if col.upper().strip() in ['ACCOUNT #', 'ACCOUNTID', 'ACCOUNT ID', 'ACCOUNT', 'ID', 'ACC. #', 'ACC #']:
                                    account_id_col_ext = col
                                    break
                            for col in best_cpt_df.columns:
                                if col.upper().strip() in ['ACCOUNT #', 'ACCOUNTID', 'ACCOUNT ID', 'ACCOUNT', 'ID', 'ACC. #', 'ACC #']:
                                    account_id_col_cpt = col
                                    break
                            
                            if account_id_col_ext and account_id_col_cpt:
                                cpt_cols = ['CPT', 'ASA Code'] if 'CPT' in best_cpt_df.columns else []
                                if cpt_cols:
                                    results_df = results_df.merge(
                                        best_cpt_df[[account_id_col_cpt] + cpt_cols].rename(columns={account_id_col_cpt: account_id_col_ext}),
                                        on=account_id_col_ext,
                                        how='left'
                                    )
                    
                    # Merge ICD predictions (ICD uses 'Patient Filename', same logic as unified processing)
                    filename_col_icd = None
                    if 'Patient Filename' in icd_df.columns:
                        filename_col_icd = 'Patient Filename'
                    elif 'Filename' in icd_df.columns:
                        filename_col_icd = 'Filename'
                    
                    source_file_col_ext = None
                    if 'source_file' in results_df.columns:
                        source_file_col_ext = 'source_file'
                    else:
                        for col in results_df.columns:
                            if col.upper().strip() in ['SOURCE_FILE', 'SOURCE FILE', 'PATIENT FILENAME', 'FILENAME']:
                                source_file_col_ext = col
                                break
                    
                    # Remove existing ICD columns before merging
                    icd_columns_to_remove = ['ICD1', 'ICD2', 'ICD3', 'ICD4']
                    existing_icd_cols = [col for col in icd_columns_to_remove if col in results_df.columns]
                    if existing_icd_cols:
                        results_df = results_df.drop(columns=existing_icd_cols, errors='ignore')
                    
                    # Extract ICD columns to merge
                    icd_cols_to_merge = ['ICD1', 'ICD2', 'ICD3', 'ICD4', 'Model Source', 'Tokens Used', 'Cost (USD)', 'Error Message']
                    icd_cols_available = [col for col in icd_cols_to_merge if col in icd_df.columns]
                    
                    if filename_col_icd and source_file_col_ext:
                        # Merge on source_file (same logic as unified processing)
                        merge_cols = [filename_col_icd] + icd_cols_available
                        icd_df_merge = icd_df[merge_cols]
                        # Rename filename column to match for merge
                        icd_df_merge = icd_df_merge.rename(columns={filename_col_icd: source_file_col_ext})
                        results_df = results_df.merge(icd_df_merge, on=source_file_col_ext, how='left', suffixes=('', '_drop'))
                        # Drop any columns with _drop suffix
                        results_df = results_df.drop(columns=[col for col in results_df.columns if col.endswith('_drop')], errors='ignore')
                        logger.info(f"[Refinement {job_id}] Merged ICD results with columns: {icd_cols_available}")
                    elif filename_col_icd and filename_col_icd in results_df.columns:
                        # Both have Filename column
                        merge_cols = [filename_col_icd] + icd_cols_available
                        icd_df_merge = icd_df[merge_cols]
                        results_df = results_df.merge(icd_df_merge, on=filename_col_icd, how='left', suffixes=('', '_drop'))
                        results_df = results_df.drop(columns=[col for col in results_df.columns if col.endswith('_drop')], errors='ignore')
                        logger.info(f"[Refinement {job_id}] Merged ICD results on {filename_col_icd}")
                    else:
                        # Fallback: assume same order
                        logger.warning(f"[Refinement {job_id}] Cannot merge ICD properly, using fallback")
                        icd_cols = [col for col in icd_df.columns if col.startswith('ICD')]
                        for col in icd_cols:
                            if col in icd_df.columns:
                                results_df[col] = icd_df[col].values[:len(results_df)]
                    
                    # Save merged results
                    results_csv_path = str(temp_dir / f"icd_iter_{icd_iteration}_results.csv")
                    results_df.to_csv(results_csv_path, index=False)
                    logger.info(f"[Refinement {job_id}] ICD prediction completed: {results_csv_path}")
                    # Track final results CSV (last iteration)
                    final_results_csv_path = results_csv_path
                    
                except Exception as e:
                    logger.error(f"[Refinement {job_id}] ICD prediction failed: {e}")
                    update_refinement_job(job_id, status="failed", error_message=str(e))
                    return
                
                # Calculate accuracy
                try:
                    cpt_accuracy, icd1_accuracy, all_errors = calculate_accuracy(
                        predictions_path=results_csv_path,
                        ground_truth_path=ground_truth_path,
                        pdf_mapping=pdf_mapping
                    )
                    
                    # Update best accuracy (only if ICD was calculated)
                    if enable_icd and icd1_accuracy is not None:
                        # Update accuracy for the instructions we just tested
                        # Find the history entry for these instructions (by iteration number)
                        history_entry = None
                        for entry in icd_instruction_history:
                            if entry['iteration'] == icd_iteration - 1:  # Instructions from previous iteration that we just tested
                                history_entry = entry
                                break
                        
                        if history_entry is not None:
                            # Update the accuracy for the instructions we just tested
                            history_entry['accuracy'] = icd1_accuracy
                            logger.info(f"[Refinement {job_id}] Updated history for iteration {icd_iteration - 1} with accuracy {icd1_accuracy:.2%}")
                        elif icd_iteration == 1:
                            # First iteration: update original (iteration 0) with its accuracy
                            icd_instruction_history[0]['accuracy'] = icd1_accuracy
                            logger.info(f"[Refinement {job_id}] Updated original instructions (iteration 0) with accuracy {icd1_accuracy:.2%}")
                        
                        if icd1_accuracy > best_icd1_accuracy:
                            best_icd1_accuracy = icd1_accuracy
                            best_icd_template_id = current_icd_template_id
                        
                        logger.info(f"[Refinement {job_id}] ICD1 Accuracy: {icd1_accuracy:.2%}")
                        
                        # Update job status
                        update_refinement_job(
                            job_id,
                            iteration=icd_iteration,
                            icd1_accuracy=icd1_accuracy,
                            best_icd1_accuracy=best_icd1_accuracy,
                            current_icd_template_id=current_icd_template_id,
                            best_icd_template_id=best_icd_template_id
                        )
                        
                        # Get error cases for this iteration
                        icd_errors = get_error_cases(
                            predictions_path=results_csv_path,
                            ground_truth_path=ground_truth_path,
                            error_type='icd1',
                            pdf_mapping=pdf_mapping,
                            limit=10
                        )
                        
                        # Determine status
                        if icd1_accuracy >= target_icd_accuracy:
                            status = "target_reached"
                            logger.info(f"[Refinement {job_id}] ICD target accuracy reached: {icd1_accuracy:.2%} >= {target_icd_accuracy:.2%}")
                        elif icd_iteration >= max_iterations:
                            status = "max_iterations"
                            logger.info(f"[Refinement {job_id}] ICD max iterations reached")
                        else:
                            status = "continuing"
                        
                        # Send email report
                        send_iteration_report(
                            to_email=notification_email,
                            iteration=icd_iteration,
                            phase="icd",
                            previous_accuracy=previous_icd_accuracy,
                            current_accuracy=icd1_accuracy,
                            best_accuracy=best_icd1_accuracy,
                            old_instructions=icd_instructions,
                            new_instructions=icd_instructions,  # Will be updated below if refining
                            error_cases=icd_errors,
                            status=status
                        )
                        
                        # Check if we should continue
                        if icd1_accuracy >= target_icd_accuracy or icd_iteration >= max_iterations:
                            break
                    else:
                        # If ICD is disabled, just break after first iteration
                        logger.info(f"[Refinement {job_id}] ICD refinement disabled, skipping ICD phase")
                        break
                    
                    # Refine instructions using Gemini
                    if refinement_mode == "focused":
                        # FOCUSED MODE: Process one error at a time
                        logger.info(f"[Refinement {job_id}] Refining ICD instructions in FOCUSED MODE (one error at a time)...")
                        
                        # Get ALL error cases (not just 10) for focused mode
                        all_icd_errors = get_error_cases(
                            predictions_path=results_csv_path,
                            ground_truth_path=ground_truth_path,
                            error_type='icd1',
                            pdf_mapping=pdf_mapping,
                            limit=None  # Get all errors
                        )
                        
                        logger.info(f"[Refinement {job_id}] Processing {len(all_icd_errors)} ICD errors one by one...")
                        
                        # Start with current instructions
                        working_instructions = icd_instructions
                        all_reasonings = []
                        rule_type_counts = {"general": 0, "hardcoded": 0}
                        
                        # Process each error one at a time
                        for error_idx, error_case in enumerate(all_icd_errors, 1):
                            account_id = error_case.get('account_id', 'N/A')
                            pdf_path = pdf_mapping.get(account_id, error_case.get('pdf_path', ''))
                            
                            logger.info(f"[Refinement {job_id}] Processing error {error_idx}/{len(all_icd_errors)}: Predicted '{error_case.get('predicted', 'N/A')}' -> Expected '{error_case.get('expected', 'N/A')}'")
                            
                            # Refine with this single error
                            improved_instructions, reasoning, rule_type = refine_instructions_focused_mode(
                                current_instructions=working_instructions,
                                single_error_case=error_case,
                                pdf_path=pdf_path,
                                instruction_type="icd",
                                model=refinement_model,
                                user_guidance=refinement_guidance
                            )
                            
                            if not improved_instructions:
                                logger.warning(f"[Refinement {job_id}] Failed to refine for error {error_idx}: {reasoning}")
                                continue
                            
                            # Update working instructions for next iteration
                            working_instructions = improved_instructions
                            all_reasonings.append(f"Error {error_idx} (Account {account_id}): {reasoning}")
                            
                            if rule_type:
                                rule_type_counts[rule_type] = rule_type_counts.get(rule_type, 0) + 1
                        
                        # Final improved instructions after processing all errors
                        improved_instructions = working_instructions
                        reasoning = f"Processed {len(all_icd_errors)} errors. Rules added: {rule_type_counts['general']} general, {rule_type_counts['hardcoded']} hardcoded.\n\n" + "\n".join(all_reasonings[:5])  # Show first 5 reasonings
                        
                        if not improved_instructions:
                            logger.error(f"[Refinement {job_id}] Failed to refine ICD instructions in focused mode")
                            update_refinement_job(job_id, status="failed", error_message="Instruction refinement failed in focused mode")
                            return
                    else:
                        # BATCH MODE: Process all errors in batches, refining after each batch
                        logger.info(f"[Refinement {job_id}] Refining ICD instructions in BATCH MODE (processing all errors in batches)...")
                        
                        # Get ALL error cases
                        all_icd_errors = get_error_cases(
                            predictions_path=results_csv_path,
                            ground_truth_path=ground_truth_path,
                            error_type='icd1',
                            pdf_mapping=pdf_mapping,
                            limit=None  # Get all errors
                        )
                        
                        logger.info(f"[Refinement {job_id}] Processing {len(all_icd_errors)} ICD errors in batches of {batch_size}...")
                        
                        # Start with current instructions
                        working_instructions = icd_instructions
                        all_reasonings = []
                        
                        # Process errors in batches
                        for batch_start in range(0, len(all_icd_errors), batch_size):
                            batch_end = min(batch_start + batch_size, len(all_icd_errors))
                            batch_errors = all_icd_errors[batch_start:batch_end]
                            batch_num = (batch_start // batch_size) + 1
                            total_batches = (len(all_icd_errors) + batch_size - 1) // batch_size
                            
                            logger.info(f"[Refinement {job_id}] Processing batch {batch_num}/{total_batches} (errors {batch_start+1}-{batch_end} of {len(all_icd_errors)})...")
                            
                            # Refine with this batch of errors (pass instruction history)
                            improved_instructions, reasoning = refine_icd_instructions(
                                current_instructions=working_instructions,
                                error_cases=batch_errors,
                                pdf_mapping=pdf_mapping,
                                model=refinement_model,
                                user_guidance=refinement_guidance,
                                pdf_image_cache=pdf_image_cache,
                                instruction_history=icd_instruction_history
                            )
                            
                            if not improved_instructions:
                                logger.warning(f"[Refinement {job_id}] Failed to refine batch {batch_num}: {reasoning}")
                                continue
                            
                            # Update working instructions for next batch
                            working_instructions = improved_instructions
                            all_reasonings.append(f"Batch {batch_num}: {reasoning}")
                        
                        # Final improved instructions after processing all batches
                        improved_instructions = working_instructions
                        reasoning = f"Processed {len(all_icd_errors)} errors in {total_batches} batches.\n\n" + "\n".join(all_reasonings)
                        
                        if not improved_instructions:
                            logger.error(f"[Refinement {job_id}] Failed to refine ICD instructions in batch mode")
                            update_refinement_job(job_id, status="failed", error_message=f"Instruction refinement failed: {reasoning}")
                            return
                    
                    # Create new template version with naming pattern: base_name (iteration)
                    # Extract base name (remove any existing iteration suffix like " (1)", " (2)", etc.)
                    base_name = original_icd_template['name']
                    # Remove pattern like " (1)", " (2)" if it exists
                    base_name = re.sub(r'\s*\(\d+\)\s*$', '', base_name)
                    new_template_name = f"{base_name} ({icd_iteration})"
                    new_template_id = create_prediction_instruction(
                        name=new_template_name,
                        instruction_type="icd",
                        instructions_text=improved_instructions,
                        description=f"Auto-refined iteration {icd_iteration} of {original_icd_template['name']}"
                    )
                    
                    if not new_template_id:
                        raise Exception("Failed to create new ICD template")
                    
                    logger.info(f"[Refinement {job_id}] Created new ICD template: {new_template_name} (ID: {new_template_id})")
                    
                    # Add improved instructions to history AFTER refinement (accuracy will be filled in next iteration when tested)
                    icd_instruction_history.append({
                        "instructions": improved_instructions,
                        "accuracy": None,  # Will be updated in next iteration when these instructions are tested
                        "iteration": icd_iteration
                    })
                    logger.info(f"[Refinement {job_id}] Added ICD iteration {icd_iteration} to history (accuracy will be measured in next iteration)")
                    
                    # Update current template
                    current_icd_template_id = new_template_id
                    previous_icd_accuracy = icd1_accuracy
                    
                    # Send updated email with new instructions
                    send_iteration_report(
                        to_email=notification_email,
                        iteration=icd_iteration,
                        phase="icd",
                        previous_accuracy=previous_icd_accuracy,
                        current_accuracy=icd1_accuracy,
                        best_accuracy=best_icd1_accuracy,
                        old_instructions=icd_instructions,
                        new_instructions=improved_instructions,
                        error_cases=icd_errors,
                        status=status,
                        gemini_reasoning=reasoning
                    )
                    
                except Exception as e:
                    logger.error(f"[Refinement {job_id}] Error in ICD iteration {icd_iteration}: {e}")
                    update_refinement_job(job_id, status="failed", error_message=str(e))
                    return
        
        # ==================== COMPLETION ====================
        logger.info(f"[Refinement {job_id}] Refinement complete")
        update_refinement_job(job_id, phase="complete", status="completed")
        
        # Send final completion email with detailed metrics
        send_completion_report(
            to_email=notification_email,
            job_id=job_id,
            final_cpt_accuracy=best_cpt_accuracy,
            final_icd1_accuracy=best_icd1_accuracy,
            best_cpt_template_id=best_cpt_template_id,
            best_icd_template_id=best_icd_template_id,
            total_iterations=cpt_iteration + icd_iteration,
            predictions_path=final_results_csv_path,
            ground_truth_path=ground_truth_path,
            enable_cpt=enable_cpt,
            enable_icd=enable_icd,
            pdf_mapping=pdf_mapping,
            cpt_instruction_history=cpt_instruction_history if enable_cpt else None,
            icd_instruction_history=icd_instruction_history if enable_icd else None
        )
        
        # Cleanup
        clear_pdf_image_cache()  # Free memory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"[Refinement {job_id}] Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        update_refinement_job(job_id, status="failed", error_message=str(e))

