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
from instruction_refinement import refine_cpt_instructions, refine_icd_instructions
from email_utils import send_iteration_report, send_completion_report

logger = logging.getLogger(__name__)


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
    notification_email: str
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
        
        # ==================== PHASE 1: CPT REFINEMENT ====================
        if enable_cpt:
            logger.info(f"[Refinement {job_id}] Starting CPT refinement phase")
            update_refinement_job(job_id, phase="cpt", status="running")
        
        cpt_iteration = 0
        previous_cpt_accuracy = None
        
        while enable_cpt and cpt_iteration < max_iterations:
            cpt_iteration += 1
            logger.info(f"[Refinement {job_id}] CPT Iteration {cpt_iteration}/{max_iterations}")
            
            # Get current CPT template
            current_cpt_template = get_prediction_instruction(instruction_id=current_cpt_template_id)
            cpt_instructions = current_cpt_template['instructions_text']
            
            # Run unified processing with current instructions
            iteration_job_id = f"{job_id}_cpt_iter_{cpt_iteration}"
            iteration_zip_path = f"/tmp/{iteration_job_id}_input.zip"
            shutil.copy2(zip_path, iteration_zip_path)
            
            # Create a temporary job status for this iteration
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from main import ProcessingJob, job_status, process_unified_background
            iteration_job = ProcessingJob(iteration_job_id)
            job_status[iteration_job_id] = iteration_job
            
            logger.info(f"[Refinement {job_id}] Running unified processing for CPT iteration {cpt_iteration}")
            
            try:
                process_unified_background(
                    job_id=iteration_job_id,
                    zip_path=iteration_zip_path,
                    excel_path=excel_path,
                    excel_filename=excel_filename,
                    enable_extraction=enable_extraction,
                    extraction_n_pages=extraction_n_pages,
                    extraction_model=extraction_model,
                    extraction_max_workers=extraction_max_workers,
                    worktracker_group=worktracker_group,
                    worktracker_batch=worktracker_batch,
                    extract_csn=extract_csn,
                    enable_cpt=True,
                    cpt_vision_mode=cpt_vision_mode,
                    cpt_client=cpt_client,
                    cpt_vision_pages=cpt_vision_pages,
                    cpt_vision_model=cpt_vision_model,
                    cpt_include_code_list=cpt_include_code_list,
                    cpt_max_workers=cpt_max_workers,
                    cpt_custom_instructions=cpt_instructions,
                    cpt_instruction_template_id=current_cpt_template_id,
                    enable_icd=False,  # Disable ICD for CPT phase
                    icd_n_pages=icd_n_pages,
                    icd_vision_model=icd_vision_model,
                    icd_max_workers=icd_max_workers,
                    icd_custom_instructions="",
                    icd_instruction_template_id=None
                )
                
                # Wait for job to complete
                import time
                while iteration_job.status == "processing":
                    time.sleep(2)
                
                if iteration_job.status == "failed":
                    raise Exception(f"Unified processing failed: {iteration_job.error}")
                
                # Get results CSV path
                results_csv_path = iteration_job.result_file
                if not results_csv_path or not os.path.exists(results_csv_path):
                    raise Exception("Results CSV not found")
                
                # Build PDF mapping from results CSV (if not already built)
                if not pdf_mapping:
                    logger.info(f"[Refinement {job_id}] Building PDF mapping from extraction results CSV")
                    import pandas as pd
                    try:
                        results_df = pd.read_csv(results_csv_path, dtype=str)
                        
                        # Find account ID and source file columns
                        account_id_col = None
                        source_file_col = None
                        
                        for col in results_df.columns:
                            col_upper = col.upper().strip()
                            if col_upper in ['ACCOUNTID', 'ACCOUNT ID', 'ACCOUNT', 'ID']:
                                account_id_col = col
                            elif col_upper in ['SOURCE_FILE', 'SOURCE FILE', 'PATIENT FILENAME', 'FILENAME']:
                                source_file_col = col
                        
                        if account_id_col and source_file_col:
                            for idx, row in results_df.iterrows():
                                account_id = str(row[account_id_col]).strip()
                                source_file = str(row[source_file_col]).strip()
                                
                                # Build full PDF path
                                pdf_path = pdf_folder / source_file
                                if pdf_path.exists():
                                    pdf_mapping[account_id] = str(pdf_path)
                            
                            logger.info(f"[Refinement {job_id}] Built PDF mapping with {len(pdf_mapping)} account IDs from extraction CSV")
                        else:
                            logger.warning(f"[Refinement {job_id}] Could not find account ID or source file columns in results CSV")
                    except Exception as e:
                        logger.error(f"[Refinement {job_id}] Error building PDF mapping from CSV: {e}")
                
            except Exception as e:
                logger.error(f"[Refinement {job_id}] Unified processing failed: {e}")
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
                    logger.info(f"[Refinement {job_id}] Refining CPT instructions with Gemini...")
                    improved_instructions, reasoning = refine_cpt_instructions(
                        current_instructions=cpt_instructions,
                        error_cases=cpt_errors,
                        pdf_mapping=pdf_mapping
                    )
                    
                    if not improved_instructions:
                        logger.error(f"[Refinement {job_id}] Failed to refine CPT instructions: {reasoning}")
                        update_refinement_job(job_id, status="failed", error_message=f"Instruction refinement failed: {reasoning}")
                        return
                    
                    # Create new template version
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_template_name = f"{original_cpt_template['name']}_iteration_{cpt_iteration}_{timestamp}"
                    new_template_id = create_prediction_instruction(
                        name=new_template_name,
                        instruction_type="cpt",
                        instructions_text=improved_instructions,
                        description=f"Auto-refined iteration {cpt_iteration} of {original_cpt_template['name']}"
                    )
                    
                    if not new_template_id:
                        raise Exception("Failed to create new CPT template")
                    
                    logger.info(f"[Refinement {job_id}] Created new CPT template: {new_template_name} (ID: {new_template_id})")
                    
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
                
                # Run unified processing with best CPT + current ICD
                iteration_job_id = f"{job_id}_icd_iter_{icd_iteration}"
                iteration_zip_path = f"/tmp/{iteration_job_id}_input.zip"
                shutil.copy2(zip_path, iteration_zip_path)
                
                # Create a temporary job status for this iteration
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from main import ProcessingJob, job_status, process_unified_background
                iteration_job = ProcessingJob(iteration_job_id)
                job_status[iteration_job_id] = iteration_job
                
                logger.info(f"[Refinement {job_id}] Running unified processing for ICD iteration {icd_iteration}")
                
                try:
                    # Get best CPT instructions if CPT is enabled
                    best_cpt_instructions = ""
                    best_cpt_template_id_for_icd = None
                    if enable_cpt and best_cpt_template_id:
                        best_cpt_template = get_prediction_instruction(instruction_id=best_cpt_template_id)
                        if best_cpt_template:
                            best_cpt_instructions = best_cpt_template['instructions_text']
                            best_cpt_template_id_for_icd = best_cpt_template_id
                    
                    process_unified_background(
                        job_id=iteration_job_id,
                        zip_path=iteration_zip_path,
                        excel_path=excel_path,
                        excel_filename=excel_filename,
                        enable_extraction=enable_extraction,
                        extraction_n_pages=extraction_n_pages,
                        extraction_model=extraction_model,
                        extraction_max_workers=extraction_max_workers,
                        worktracker_group=worktracker_group,
                        worktracker_batch=worktracker_batch,
                        extract_csn=extract_csn,
                        enable_cpt=enable_cpt,  # Use enable_cpt flag
                        cpt_vision_mode=cpt_vision_mode,
                        cpt_client=cpt_client,
                        cpt_vision_pages=cpt_vision_pages,
                        cpt_vision_model=cpt_vision_model,
                        cpt_include_code_list=cpt_include_code_list,
                        cpt_max_workers=cpt_max_workers,
                        cpt_custom_instructions=best_cpt_instructions,
                        cpt_instruction_template_id=best_cpt_template_id_for_icd,
                        enable_icd=True,  # Always enable ICD during ICD phase
                        icd_n_pages=icd_n_pages,
                        icd_vision_model=icd_vision_model,
                        icd_max_workers=icd_max_workers,
                        icd_custom_instructions=icd_instructions,
                        icd_instruction_template_id=current_icd_template_id
                    )
                    
                    # Wait for job to complete
                    import time
                    while iteration_job.status == "processing":
                        time.sleep(2)
                    
                    if iteration_job.status == "failed":
                        raise Exception(f"Unified processing failed: {iteration_job.error}")
                    
                    # Get results CSV path
                    results_csv_path = iteration_job.result_file
                    if not results_csv_path or not os.path.exists(results_csv_path):
                        raise Exception("Results CSV not found")
                    
                    # Update PDF mapping from results CSV (if needed for ICD phase)
                    if not pdf_mapping:
                        logger.info(f"[Refinement {job_id}] Building PDF mapping from ICD iteration results CSV")
                        import pandas as pd
                        try:
                            results_df = pd.read_csv(results_csv_path, dtype=str)
                            
                            # Find account ID and source file columns
                            account_id_col = None
                            source_file_col = None
                            
                            for col in results_df.columns:
                                col_upper = col.upper().strip()
                                if col_upper in ['ACCOUNTID', 'ACCOUNT ID', 'ACCOUNT', 'ID']:
                                    account_id_col = col
                                elif col_upper in ['SOURCE_FILE', 'SOURCE FILE', 'PATIENT FILENAME', 'FILENAME']:
                                    source_file_col = col
                            
                            if account_id_col and source_file_col:
                                for idx, row in results_df.iterrows():
                                    account_id = str(row[account_id_col]).strip()
                                    source_file = str(row[source_file_col]).strip()
                                    
                                    # Build full PDF path
                                    pdf_path = pdf_folder / source_file
                                    if pdf_path.exists():
                                        pdf_mapping[account_id] = str(pdf_path)
                                
                                logger.info(f"[Refinement {job_id}] Built PDF mapping with {len(pdf_mapping)} account IDs from ICD iteration CSV")
                            else:
                                logger.warning(f"[Refinement {job_id}] Could not find account ID or source file columns in ICD results CSV")
                        except Exception as e:
                            logger.error(f"[Refinement {job_id}] Error building PDF mapping from ICD CSV: {e}")
                    
                except Exception as e:
                    logger.error(f"[Refinement {job_id}] Unified processing failed: {e}")
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
                    logger.info(f"[Refinement {job_id}] Refining ICD instructions with Gemini...")
                    improved_instructions, reasoning = refine_icd_instructions(
                        current_instructions=icd_instructions,
                        error_cases=icd_errors,
                        pdf_mapping=pdf_mapping
                    )
                    
                    if not improved_instructions:
                        logger.error(f"[Refinement {job_id}] Failed to refine ICD instructions: {reasoning}")
                        update_refinement_job(job_id, status="failed", error_message=f"Instruction refinement failed: {reasoning}")
                        return
                    
                    # Create new template version
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_template_name = f"{original_icd_template['name']}_iteration_{icd_iteration}_{timestamp}"
                    new_template_id = create_prediction_instruction(
                        name=new_template_name,
                        instruction_type="icd",
                        instructions_text=improved_instructions,
                        description=f"Auto-refined iteration {icd_iteration} of {original_icd_template['name']}"
                    )
                    
                    if not new_template_id:
                        raise Exception("Failed to create new ICD template")
                    
                    logger.info(f"[Refinement {job_id}] Created new ICD template: {new_template_name} (ID: {new_template_id})")
                    
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
        
        # Send final completion email
        send_completion_report(
            to_email=notification_email,
            job_id=job_id,
            final_cpt_accuracy=best_cpt_accuracy,
            final_icd1_accuracy=best_icd1_accuracy,
            best_cpt_template_id=best_cpt_template_id,
            best_icd_template_id=best_icd_template_id,
            total_iterations=cpt_iteration + icd_iteration
        )
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"[Refinement {job_id}] Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        update_refinement_job(job_id, status="failed", error_message=str(e))

