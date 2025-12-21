#!/usr/bin/env python3
"""
Email utility functions for sending iteration reports via Resend.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

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


def send_completion_report(
    to_email: str,
    job_id: str,
    final_cpt_accuracy: Optional[float],
    final_icd1_accuracy: Optional[float],
    best_cpt_template_id: Optional[int],
    best_icd_template_id: Optional[int],
    total_iterations: int
) -> bool:
    """
    Send final completion email when refinement job finishes.
    
    Args:
        to_email: Recipient email address
        job_id: Job ID
        final_cpt_accuracy: Final CPT accuracy achieved
        final_icd1_accuracy: Final ICD1 accuracy achieved
        best_cpt_template_id: ID of best CPT template
        best_icd_template_id: ID of best ICD template
        total_iterations: Total iterations completed
    
    Returns:
        True if email sent successfully, False otherwise
    """
    if not init_resend():
        logger.error("Failed to initialize Resend")
        return False
    
    def fmt_pct(val):
        return f"{val * 100:.2f}%" if val is not None else "N/A"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; max-width: 800px; margin: 0 auto; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .metrics {{ display: flex; justify-content: space-around; margin: 15px 0; }}
            .metric {{ text-align: center; }}
            .metric-value {{ font-size: 32px; font-weight: bold; color: #28a745; }}
            .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
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
            
            <div class="section">
                <h3>Best Templates</h3>
                <p><strong>Best CPT Template ID:</strong> {best_cpt_template_id or 'N/A'}</p>
                <p><strong>Best ICD Template ID:</strong> {best_icd_template_id or 'N/A'}</p>
            </div>
            
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

