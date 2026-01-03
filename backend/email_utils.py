#!/usr/bin/env python3
"""
Email utility functions for sending iteration reports via Resend.
"""

import os
import logging
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False

logger = logging.getLogger(__name__)

# Email configuration
FROM_EMAIL = "office@novayachts.eu"
DEFAULT_TO_EMAIL = "cvetkovskileon@gmail.com"


def init_resend():
    """Initialize Resend with API key from environment."""
    if not RESEND_AVAILABLE:
        logger.error("Resend package not installed. Install with: pip install resend")
        return False
    
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.error("RESEND_API_KEY environment variable not set")
        return False
    
    resend.api_key = api_key
    return True


def send_iteration_report(
    to_email: str,
    iteration: int,
    phase: str,  # 'cpt' or 'icd'
    previous_accuracy: Optional[float],
    current_accuracy: float,
    best_accuracy: float,
    old_instructions: str,
    new_instructions: str,
    error_cases: List[Dict[str, Any]],
    status: str,  # 'continuing', 'target_reached', 'max_iterations', 'completed'
    gemini_reasoning: Optional[str] = None
) -> bool:
    """
    Send an email report after each refinement iteration.
    
    Args:
        to_email: Recipient email address
        iteration: Current iteration number
        phase: 'cpt' or 'icd'
        previous_accuracy: Previous iteration accuracy (None for first iteration)
        current_accuracy: Current iteration accuracy
        best_accuracy: Best accuracy achieved so far
        old_instructions: Previous instruction text
        new_instructions: New instruction text
        error_cases: List of error case dictionaries with keys: account_id, pdf_path, predicted, expected, error_type
        status: Current status
        gemini_reasoning: Optional reasoning from Gemini about the changes
    
    Returns:
        True if email sent successfully, False otherwise
    """
    if not init_resend():
        logger.error("Failed to initialize Resend")
        return False
    
    phase_name = "CPT" if phase == "cpt" else "ICD"
    phase_display = "CPT Refinement" if phase == "cpt" else "ICD Refinement"
    
    # Calculate accuracy change
    accuracy_change = None
    change_symbol = ""
    change_color = "#6c757d"
    if previous_accuracy is not None:
        accuracy_change = current_accuracy - previous_accuracy
        change_symbol = "↑" if accuracy_change > 0 else "↓" if accuracy_change < 0 else "→"
        if accuracy_change and accuracy_change > 0:
            change_color = "#28a745"
        elif accuracy_change and accuracy_change < 0:
            change_color = "#dc3545"
    
    # Format accuracy as percentage
    def fmt_pct(val):
        return f"{val * 100:.2f}%" if val is not None else "N/A"
    
    # Build error cases section
    error_section = ""
    if error_cases:
        error_section = "\nTop 10 Remaining Errors:\n\n"
        for idx, error in enumerate(error_cases[:10], 1):
            account_id = error.get('account_id', 'N/A')
            pdf_path = error.get('pdf_path', 'N/A')
            predicted = error.get('predicted', 'N/A')
            expected = error.get('expected', 'N/A')
            error_section += f"{idx}. Account: {account_id} | PDF: {pdf_path}\n"
            error_section += f"   Predicted: {predicted}\n"
            error_section += f"   Expected:  {expected}\n\n"
    else:
        error_section = "\nNo errors found!\n"
    
    # Determine next step message
    next_step = ""
    if status == "target_reached":
        if phase == "cpt":
            next_step = "Moving to ICD refinement phase"
        else:
            next_step = "Job complete - both phases reached target accuracy"
    elif status == "max_iterations":
        if phase == "cpt":
            next_step = "Moving to ICD refinement phase (max iterations reached)"
        else:
            next_step = "Job complete - max iterations reached"
    elif status == "continuing":
        next_step = f"Starting iteration {iteration + 1}"
    elif status == "completed":
        next_step = "Job complete"
    
    # Build change paragraph separately to avoid nested f-string issues
    change_paragraph = ""
    if accuracy_change is not None:
        change_paragraph = f'<p style="text-align: center; font-size: 18px; color: {change_color};">Change: {change_symbol} {abs(accuracy_change * 100):.2f}%</p>'
    
    # Build email HTML
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; max-width: 800px; margin: 0 auto; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .section h3 {{ margin-top: 0; color: #667eea; }}
            .metrics {{ display: flex; justify-content: space-around; margin: 15px 0; }}
            .metric {{ text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
            .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
            .instructions {{ background: white; padding: 15px; border-left: 4px solid #667eea; margin: 10px 0; }}
            .instructions pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 12px; }}
            .error-case {{ background: white; padding: 10px; margin: 5px 0; border-left: 3px solid #e74c3c; }}
            .status {{ padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }}
            .status.continuing {{ background: #fff3cd; color: #856404; }}
            .status.target_reached {{ background: #d4edda; color: #155724; }}
            .status.max_iterations {{ background: #d1ecf1; color: #0c5460; }}
            .status.completed {{ background: #d4edda; color: #155724; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AI Refinement - Iteration {iteration}</h1>
            <p>{phase_display}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h3>Accuracy Metrics</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{fmt_pct(previous_accuracy)}</div>
                        <div class="metric-label">Previous {phase_name}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{fmt_pct(current_accuracy)}</div>
                        <div class="metric-label">Current {phase_name}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{fmt_pct(best_accuracy)}</div>
                        <div class="metric-label">Best So Far</div>
                    </div>
                </div>
                {change_paragraph}
            </div>
            
            <div class="section">
                <h3>Instruction Changes</h3>
                <div class="instructions">
                    <strong>OLD INSTRUCTIONS:</strong>
                    <pre>{old_instructions}</pre>
                </div>
                <div class="instructions">
                    <strong>NEW INSTRUCTIONS:</strong>
                    <pre>{new_instructions}</pre>
                </div>
                {f'<div class="instructions"><strong>GEMINI REASONING:</strong><pre>{gemini_reasoning}</pre></div>' if gemini_reasoning else ''}
            </div>
            
            <div class="section">
                <h3>Error Analysis</h3>
                {error_section}
            </div>
            
            <div class="section">
                <h3>Progress</h3>
                <p><strong>Phase:</strong> {phase_display}</p>
                <p><strong>Status:</strong> <span class="status {status}">{status.replace('_', ' ').title()}</span></p>
                <p><strong>Next Step:</strong> {next_step}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Build plain text version
    change_text = f"{change_symbol} {abs(accuracy_change * 100):.2f}%" if accuracy_change is not None else "N/A"
    text_content = f"""
========================================
ITERATION {iteration} - {phase_name} REFINEMENT
========================================

ACCURACY METRICS
----------------
Previous {phase_name}: {fmt_pct(previous_accuracy)}
Current {phase_name}:  {fmt_pct(current_accuracy)}
Change: {change_text}

Best So Far: {fmt_pct(best_accuracy)}

INSTRUCTION CHANGES
-------------------
OLD INSTRUCTIONS:
{old_instructions}

NEW INSTRUCTIONS:
{new_instructions}
{f'GEMINI REASONING:\n{gemini_reasoning}\n' if gemini_reasoning else ''}
ERROR ANALYSIS
--------------
{error_section}

PROGRESS
--------
Phase: {phase_display}
Status: {status.replace('_', ' ').title()}
Next Step: {next_step}

========================================
    """
    
    try:
        params: resend.Emails.SendParams = {
            "from": f"Medical Data Processor <{FROM_EMAIL}>",
            "to": [to_email],
            "subject": f"AI Refinement - Iteration {iteration} - {phase_name} - Accuracy: {fmt_pct(current_accuracy)}",
            "html": html_content,
            "text": text_content
        }
        
        email = resend.Emails.send(params)
        logger.info(f"Email sent successfully to {to_email} for iteration {iteration}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def _build_iteration_summary(
    enable_cpt: bool,
    enable_icd: bool,
    cpt_instruction_history: Optional[List[Dict[str, Any]]],
    icd_instruction_history: Optional[List[Dict[str, Any]]],
    fmt_pct: callable
) -> str:
    """
    Build HTML summary of all iterations tried during refinement.
    
    Args:
        enable_cpt: Whether CPT was enabled
        enable_icd: Whether ICD was enabled
        cpt_instruction_history: List of CPT instruction history entries
        icd_instruction_history: List of ICD instruction history entries
        fmt_pct: Function to format percentage values
    
    Returns:
        HTML string with iteration summary
    """
    summary_sections = []
    
    # CPT Iteration Summary
    if enable_cpt and cpt_instruction_history:
        cpt_section = ["<div class='section'>", "<h3>📊 CPT Refinement History - All Iterations Tried</h3>"]
        
        for idx, entry in enumerate(cpt_instruction_history):
            iteration = entry.get('iteration', idx)
            accuracy = entry.get('accuracy')
            instructions = entry.get('instructions', '')
            
            # Truncate instructions for display (first 500 chars)
            instructions_preview = instructions[:500] + "..." if len(instructions) > 500 else instructions
            # Basic HTML escaping and line breaks
            instructions_preview = instructions_preview.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            
            # Determine status badge
            if iteration == 0:
                status_badge = "<span style='background: #6c757d; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px;'>ORIGINAL</span>"
            elif accuracy is None:
                status_badge = "<span style='background: #ffc107; color: black; padding: 3px 8px; border-radius: 3px; font-size: 11px;'>PENDING</span>"
            else:
                status_badge = "<span style='background: #28a745; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px;'>TESTED</span>"
            
            accuracy_display = fmt_pct(accuracy) if accuracy is not None else "Not yet tested"
            
            cpt_section.append(f"""
            <div style='background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                    <h4 style='margin: 0; color: #007bff;'>Iteration {iteration}</h4>
                    <div>
                        {status_badge}
                        <span style='margin-left: 10px; font-size: 18px; font-weight: bold; color: #007bff;'>{accuracy_display}</span>
                    </div>
                </div>
                <div style='background: #f9f9f9; padding: 10px; border-radius: 3px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; line-height: 1.4;'>
                    {instructions_preview}
                </div>
            </div>
            """)
        
        cpt_section.append("</div>")
        summary_sections.append("\n".join(cpt_section))
    
    # ICD Iteration Summary
    if enable_icd and icd_instruction_history:
        icd_section = ["<div class='section'>", "<h3>📊 ICD Refinement History - All Iterations Tried</h3>"]
        
        for idx, entry in enumerate(icd_instruction_history):
            iteration = entry.get('iteration', idx)
            accuracy = entry.get('accuracy')
            instructions = entry.get('instructions', '')
            
            # Truncate instructions for display (first 500 chars)
            instructions_preview = instructions[:500] + "..." if len(instructions) > 500 else instructions
            # Basic HTML escaping and line breaks
            instructions_preview = instructions_preview.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            
            # Determine status badge
            if iteration == 0:
                status_badge = "<span style='background: #6c757d; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px;'>ORIGINAL</span>"
            elif accuracy is None:
                status_badge = "<span style='background: #ffc107; color: black; padding: 3px 8px; border-radius: 3px; font-size: 11px;'>PENDING</span>"
            else:
                status_badge = "<span style='background: #28a745; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px;'>TESTED</span>"
            
            accuracy_display = fmt_pct(accuracy) if accuracy is not None else "Not yet tested"
            
            icd_section.append(f"""
            <div style='background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                    <h4 style='margin: 0; color: #28a745;'>Iteration {iteration}</h4>
                    <div>
                        {status_badge}
                        <span style='margin-left: 10px; font-size: 18px; font-weight: bold; color: #28a745;'>{accuracy_display}</span>
                    </div>
                </div>
                <div style='background: #f9f9f9; padding: 10px; border-radius: 3px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; line-height: 1.4;'>
                    {instructions_preview}
                </div>
            </div>
            """)
        
        icd_section.append("</div>")
        summary_sections.append("\n".join(icd_section))
    
    if not summary_sections:
        return ""
    
    return "\n".join(summary_sections)


def send_completion_report(
    to_email: str,
    job_id: str,
    final_cpt_accuracy: Optional[float],
    final_icd1_accuracy: Optional[float],
    best_cpt_template_id: Optional[int],
    best_icd_template_id: Optional[int],
    total_iterations: int,
    predictions_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    enable_cpt: bool = True,
    enable_icd: bool = True,
    pdf_mapping: Optional[Dict[str, str]] = None,
    cpt_instruction_history: Optional[List[Dict[str, Any]]] = None,
    icd_instruction_history: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Send final completion email when refinement job finishes with detailed metrics and case-by-case table.
    
    Args:
        to_email: Recipient email address
        job_id: Job ID
        final_cpt_accuracy: Final CPT accuracy achieved
        final_icd1_accuracy: Final ICD1 accuracy achieved
        best_cpt_template_id: ID of best CPT template
        best_icd_template_id: ID of best ICD template
        total_iterations: Total iterations completed
        predictions_path: Path to final predictions CSV (optional)
        ground_truth_path: Path to ground truth CSV (optional)
        enable_cpt: Whether CPT was enabled
        enable_icd: Whether ICD was enabled
        pdf_mapping: Optional mapping from Account ID to PDF file path
        cpt_instruction_history: Optional list of CPT instruction history entries
        icd_instruction_history: Optional list of ICD instruction history entries
    
    Returns:
        True if email sent successfully, False otherwise
    """
    if not init_resend():
        logger.error("Failed to initialize Resend")
        return False
    
    # Fetch template names from database
    best_cpt_template_name = "N/A"
    best_icd_template_name = "N/A"
    
    if best_cpt_template_id:
        from db_utils import get_prediction_instruction
        cpt_template = get_prediction_instruction(instruction_id=best_cpt_template_id)
        if cpt_template:
            best_cpt_template_name = cpt_template['name']
    
    if best_icd_template_id:
        from db_utils import get_prediction_instruction
        icd_template = get_prediction_instruction(instruction_id=best_icd_template_id)
        if icd_template:
            best_icd_template_name = icd_template['name']
    
    def fmt_pct(val):
        return f"{val * 100:.2f}%" if val is not None else "N/A"
    
    # Calculate detailed metrics and build case-by-case table
    detailed_metrics_html = ""
    case_table_html = ""
    
    if predictions_path and ground_truth_path:
        try:
            from accuracy_utils import read_dataframe, find_column, parse_icd_codes, get_predicted_icd_list
            
            # Read dataframes
            predictions_df = read_dataframe(predictions_path)
            ground_truth_df = read_dataframe(ground_truth_path)
            
            # Find columns
            account_id_col_pred = find_column(predictions_df, ['Account #', 'AccountId', 'Account ID', 'Account', 'ID', 'Acc. #', 'ACC #', 'ACCOUNT #'])
            account_id_col_gt = find_column(ground_truth_df, ['Account #', 'AccountId', 'Account ID', 'Account', 'ID', 'Acc. #', 'ACC #', 'ACCOUNT #'])
            
            cpt_col_pred = find_column(predictions_df, ['CPT', 'Cpt', 'ASA Code']) if enable_cpt else None
            cpt_col_gt = find_column(ground_truth_df, ['CPT', 'Cpt']) if enable_cpt else None
            
            icd_cols_pred = {}
            for col in predictions_df.columns:
                col_upper = col.upper().strip()
                if col_upper in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
                    icd_cols_pred[col_upper] = col
            
            icd_col_gt = find_column(ground_truth_df, ['ICD', 'Icd']) if enable_icd else None
            
            if account_id_col_pred and account_id_col_gt:
                # Build ground truth lookup
                gt_dict = {}
                for idx, row in ground_truth_df.iterrows():
                    account_id = str(row[account_id_col_gt]).strip()
                    if account_id not in gt_dict:
                        gt_dict[account_id] = {
                            'cpt': str(row[cpt_col_gt]).strip() if enable_cpt and cpt_col_gt and pd.notna(row.get(cpt_col_gt, '')) else '',
                            'icd': str(row[icd_col_gt]).strip() if enable_icd and icd_col_gt and pd.notna(row.get(icd_col_gt, '')) else ''
                        }
                
                # Calculate metrics and build case table
                cpt_total = 0
                cpt_correct = 0
                cpt_wrong = 0
                icd_total = 0
                icd_correct = 0
                icd_wrong = 0
                
                case_rows = []
                
                for idx, row in predictions_df.iterrows():
                    account_id = str(row[account_id_col_pred]).strip()
                    
                    if account_id in gt_dict:
                        gt_data = gt_dict[account_id]
                        case_row = {'account_id': account_id}
                        
                        # CPT comparison
                        if enable_cpt and cpt_col_pred:
                            predicted_cpt = str(row[cpt_col_pred]).strip() if pd.notna(row.get(cpt_col_pred, '')) else ''
                            expected_cpt = gt_data['cpt']
                            cpt_total += 1
                            cpt_match = predicted_cpt == expected_cpt
                            
                            if cpt_match:
                                cpt_correct += 1
                                case_row['cpt_status'] = '✅ Correct'
                                case_row['cpt_status_class'] = 'correct'
                            else:
                                cpt_wrong += 1
                                case_row['cpt_status'] = '❌ Wrong'
                                case_row['cpt_status_class'] = 'wrong'
                            
                            case_row['predicted_cpt'] = predicted_cpt if predicted_cpt else '(empty)'
                            case_row['expected_cpt'] = expected_cpt if expected_cpt else '(empty)'
                        
                        # ICD1 comparison
                        if enable_icd and icd_col_gt:
                            predicted_icd_list = get_predicted_icd_list(row, icd_cols_pred)
                            gt_icd_list = parse_icd_codes(gt_data['icd'])
                            
                            if len(predicted_icd_list) > 0 and len(gt_icd_list) > 0:
                                icd_total += 1
                                predicted_icd1 = predicted_icd_list[0] if predicted_icd_list[0] else ''
                                expected_icd1 = gt_icd_list[0] if gt_icd_list[0] else ''
                                icd_match = predicted_icd1 == expected_icd1
                                
                                if icd_match:
                                    icd_correct += 1
                                    case_row['icd_status'] = '✅ Correct'
                                    case_row['icd_status_class'] = 'correct'
                                else:
                                    icd_wrong += 1
                                    case_row['icd_status'] = '❌ Wrong'
                                    case_row['icd_status_class'] = 'wrong'
                                
                                case_row['predicted_icd1'] = predicted_icd1 if predicted_icd1 else '(empty)'
                                case_row['expected_icd1'] = expected_icd1 if expected_icd1 else '(empty)'
                        
                        # PDF path
                        pdf_path = pdf_mapping.get(account_id, 'N/A') if pdf_mapping else 'N/A'
                        case_row['pdf_path'] = Path(pdf_path).name if pdf_path != 'N/A' else 'N/A'
                        
                        case_rows.append(case_row)
                
                # Build detailed metrics HTML
                detailed_metrics_html = f"""
                <div class="section">
                    <h3>Detailed Metrics</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;">
                        {"<div style='background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff;'>" if enable_cpt else ""}
                        {"<h4 style='margin-top: 0; color: #007bff;'>CPT Codes</h4>" if enable_cpt else ""}
                        {"<p style='font-size: 18px; margin: 5px 0;'><strong>Total Cases:</strong> {cpt_total}</p>" if enable_cpt else ""}
                        {"<p style='font-size: 18px; margin: 5px 0; color: #28a745;'><strong>✅ Correct:</strong> {cpt_correct}</p>" if enable_cpt else ""}
                        {"<p style='font-size: 18px; margin: 5px 0; color: #dc3545;'><strong>❌ Wrong:</strong> {cpt_wrong}</p>" if enable_cpt else ""}
                        {"<p style='font-size: 24px; font-weight: bold; margin-top: 10px; color: #007bff;'>Accuracy: {fmt_pct(final_cpt_accuracy)}</p>" if enable_cpt else ""}
                        {"</div>" if enable_cpt else ""}
                        
                        {"<div style='background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;'>" if enable_icd else ""}
                        {"<h4 style='margin-top: 0; color: #28a745;'>ICD1 Codes</h4>" if enable_icd else ""}
                        {"<p style='font-size: 18px; margin: 5px 0;'><strong>Total Cases:</strong> {icd_total}</p>" if enable_icd else ""}
                        {"<p style='font-size: 18px; margin: 5px 0; color: #28a745;'><strong>✅ Correct:</strong> {icd_correct}</p>" if enable_icd else ""}
                        {"<p style='font-size: 18px; margin: 5px 0; color: #dc3545;'><strong>❌ Wrong:</strong> {icd_wrong}</p>" if enable_icd else ""}
                        {"<p style='font-size: 24px; font-weight: bold; margin-top: 10px; color: #28a745;'>Accuracy: {fmt_pct(final_icd1_accuracy)}</p>" if enable_icd else ""}
                        {"</div>" if enable_icd else ""}
                    </div>
                </div>
                """
                
                # Build text blobs for copy-paste
                all_cases_text = "=== ALL CASES (Copy-Paste Ready) ===\n\n"
                errors_only_text = "=== ERRORS ONLY (Copy-Paste Ready) ===\n\n"
                
                for case in case_rows:
                    account_id = case.get('account_id', 'N/A')
                    pdf_name = case.get('pdf_path', 'N/A')
                    
                    # Build case line
                    case_line = f"Account ID: {account_id} | PDF: {pdf_name}"
                    
                    # Add CPT info
                    if enable_cpt and 'predicted_cpt' in case:
                        predicted_cpt = case.get('predicted_cpt', '')
                        expected_cpt = case.get('expected_cpt', '')
                        cpt_status = case.get('cpt_status', '')
                        case_line += f" | CPT: Predicted='{predicted_cpt}' Expected='{expected_cpt}' Status={cpt_status}"
                    
                    # Add ICD info
                    if enable_icd and 'predicted_icd1' in case:
                        predicted_icd1 = case.get('predicted_icd1', '')
                        expected_icd1 = case.get('expected_icd1', '')
                        icd_status = case.get('icd_status', '')
                        case_line += f" | ICD1: Predicted='{predicted_icd1}' Expected='{expected_icd1}' Status={icd_status}"
                    
                    case_line += "\n"
                    
                    # Add to all cases blob
                    all_cases_text += case_line
                    
                    # Add to errors blob if wrong
                    is_error = False
                    if enable_cpt and 'cpt_status' in case and 'wrong' in case.get('cpt_status', '').lower():
                        is_error = True
                    if enable_icd and 'icd_status' in case and 'wrong' in case.get('icd_status', '').lower():
                        is_error = True
                    
                    if is_error:
                        errors_only_text += case_line
                
                # Build case-by-case table HTML
                table_rows = ""
                for case in case_rows:
                    account_id = case.get('account_id', 'N/A')
                    pdf_name = case.get('pdf_path', 'N/A')
                    
                    # CPT columns
                    cpt_cols = ""
                    if enable_cpt and 'predicted_cpt' in case:
                        cpt_status = case.get('cpt_status', '')
                        cpt_status_class = case.get('cpt_status_class', '')
                        predicted_cpt = case.get('predicted_cpt', '')
                        expected_cpt = case.get('expected_cpt', '')
                        cpt_cols = f"""
                        <td class="status-{cpt_status_class}">{cpt_status}</td>
                        <td><strong>{predicted_cpt}</strong></td>
                        <td><strong>{expected_cpt}</strong></td>
                        """
                    
                    # ICD columns
                    icd_cols = ""
                    if enable_icd and 'predicted_icd1' in case:
                        icd_status = case.get('icd_status', '')
                        icd_status_class = case.get('icd_status_class', '')
                        predicted_icd1 = case.get('predicted_icd1', '')
                        expected_icd1 = case.get('expected_icd1', '')
                        icd_cols = f"""
                        <td class="status-{icd_status_class}">{icd_status}</td>
                        <td><strong>{predicted_icd1}</strong></td>
                        <td><strong>{expected_icd1}</strong></td>
                        """
                    
                    table_rows += f"""
                    <tr>
                        <td><strong>{account_id}</strong></td>
                        <td>{pdf_name}</td>
                        {cpt_cols}
                        {icd_cols}
                    </tr>
                    """
                
                # Count total errors for display
                total_errors = 0
                if enable_cpt:
                    total_errors += cpt_wrong
                if enable_icd:
                    total_errors += icd_wrong
                
                # Build table header
                table_header = "<th>Account ID</th><th>PDF File</th>"
                if enable_cpt:
                    table_header += "<th>CPT Status</th><th>Predicted CPT</th><th>Expected CPT</th>"
                if enable_icd:
                    table_header += "<th>ICD1 Status</th><th>Predicted ICD1</th><th>Expected ICD1</th>"
                
                case_table_html = f"""
                <div class="section">
                    <h3>📋 Copy-Paste Ready Text Blobs</h3>
                    <div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0; border: 2px solid #667eea;">
                        <h4 style="margin-top: 0; color: #667eea;">📄 ALL CASES ({len(case_rows)} cases)</h4>
                        <textarea readonly style="width: 100%; min-height: 400px; font-family: 'Courier New', monospace; font-size: 13px; padding: 15px; border: 1px solid #ddd; border-radius: 3px; background: #f9f9f9; line-height: 1.6; white-space: pre-wrap;" onclick="this.select(); document.execCommand('copy');">{all_cases_text}</textarea>
                        <p style="font-size: 12px; color: #666; margin-top: 8px;">💡 Click textarea to select all, then copy-paste to AI</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0; border: 2px solid #dc3545;">
                        <h4 style="margin-top: 0; color: #dc3545;">❌ ERRORS ONLY ({total_errors} errors)</h4>
                        <textarea readonly style="width: 100%; min-height: 300px; font-family: 'Courier New', monospace; font-size: 13px; padding: 15px; border: 1px solid #dc3545; border-radius: 3px; background: #fff5f5; line-height: 1.6; white-space: pre-wrap;" onclick="this.select(); document.execCommand('copy');">{errors_only_text if errors_only_text.strip() else 'No errors found!'}</textarea>
                        <p style="font-size: 12px; color: #666; margin-top: 8px;">💡 Click textarea to select all, then copy-paste to AI</p>
                    </div>
                </div>
                
                <div class="section">
                    <h3>Case-by-Case Results Table ({len(case_rows)} cases)</h3>
                    <div style="overflow-x: auto; margin-top: 15px;">
                        <table class="case-table">
                            <thead>
                                <tr>
                                    {table_header}
                                </tr>
                            </thead>
                            <tbody>
                                {table_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
                """
        except Exception as e:
            logger.error(f"Failed to generate detailed metrics: {e}")
            detailed_metrics_html = f"<div class='section'><p style='color: #dc3545;'>⚠️ Could not generate detailed metrics: {str(e)}</p></div>"
            case_table_html = ""
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .metrics {{ display: flex; justify-content: space-around; margin: 15px 0; }}
            .metric {{ text-align: center; }}
            .metric-value {{ font-size: 32px; font-weight: bold; color: #28a745; }}
            .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
            .case-table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 10px; }}
            .case-table th {{ background: #667eea; color: white; padding: 12px; text-align: left; font-weight: bold; border: 1px solid #555; }}
            .case-table td {{ padding: 10px; border: 1px solid #ddd; }}
            .case-table tr:nth-child(even) {{ background: #f9f9f9; }}
            .case-table tr:hover {{ background: #f0f0f0; }}
            .status-correct {{ color: #28a745; font-weight: bold; }}
            .status-wrong {{ color: #dc3545; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>✅ AI Refinement Complete</h1>
            <p>Job ID: {job_id}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h3>Final Results</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{fmt_pct(final_cpt_accuracy)}</div>
                        <div class="metric-label">Final CPT Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{fmt_pct(final_icd1_accuracy)}</div>
                        <div class="metric-label">Final ICD1 Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{total_iterations}</div>
                        <div class="metric-label">Total Iterations</div>
                    </div>
                </div>
            </div>
            
            {detailed_metrics_html}
            
            {case_table_html}
            
            <div class="section">
                <h3>Best Templates</h3>
                <p><strong>Best CPT Template:</strong> {best_cpt_template_name}</p>
                <p><strong>Best ICD Template:</strong> {best_icd_template_name}</p>
            </div>
            
            {_build_iteration_summary(enable_cpt, enable_icd, cpt_instruction_history, icd_instruction_history, fmt_pct)}
            
            <p style="text-align: center; color: #666; margin-top: 30px;">
                These templates have been saved to the database and can be used for future processing.
            </p>
        </div>
    </body>
    </html>
    """
    
    try:
        params: resend.Emails.SendParams = {
            "from": f"Medical Data Processor <{FROM_EMAIL}>",
            "to": [to_email],
            "subject": f"AI Refinement Complete - Job {job_id}",
            "html": html_content
        }
        
        email = resend.Emails.send(params)
        logger.info(f"Completion email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send completion email: {e}")
        return False

