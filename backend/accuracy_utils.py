#!/usr/bin/env python3
"""
Accuracy calculation utilities for comparing predictions vs ground truth.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def read_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read a CSV or Excel file into a DataFrame, handling multiple encodings.
    
    Args:
        file_path: Path to the file
    
    Returns:
        DataFrame with the file contents
    
    Raises:
        Exception: If file cannot be read
    """
    file_path_obj = Path(file_path)
    df = None
    
    if file_path_obj.suffix.lower() in ('.xlsx', '.xls'):
        try:
            if file_path_obj.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path, dtype=str, engine='openpyxl')
            else:
                try:
                    df = pd.read_excel(file_path, dtype=str, engine='xlrd')
                except Exception:
                    df = pd.read_excel(file_path, dtype=str, engine=None)
            logger.info(f"Successfully read file as Excel: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to read as Excel: {e}, trying CSV")
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, dtype=str, encoding=encoding)
                    logger.info(f"Successfully read file as CSV with encoding: {encoding}")
                    break
                except Exception:
                    continue
    else:
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, dtype=str, encoding=encoding)
                logger.info(f"Successfully read file with encoding: {encoding}")
                break
            except Exception:
                continue
    
    if df is None:
        raise Exception(f"Could not read file: {file_path}")
    
    return df


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find a column in DataFrame by case-insensitive matching.
    
    Args:
        df: DataFrame to search
        possible_names: List of possible column names (case-insensitive)
    
    Returns:
        Actual column name if found, None otherwise
    """
    for col in df.columns:
        col_upper = col.upper().strip()
        if col_upper in [name.upper() for name in possible_names]:
            return col
    return None


def parse_icd_codes(icd_string: str) -> List[str]:
    """
    Parse comma-separated ICD codes into a list of normalized codes.
    
    Args:
        icd_string: Comma-separated ICD codes string
    
    Returns:
        List of normalized ICD codes
    """
    if pd.isna(icd_string) or not str(icd_string).strip():
        return []
    codes = [code.strip().upper() for code in str(icd_string).split(',')]
    return [c for c in codes if c]


def get_predicted_icd_list(row: pd.Series, icd_cols: Dict[str, str]) -> List[str]:
    """
    Get predicted ICD codes from ICD1, ICD2, ICD3, ICD4 columns as a list.
    
    Args:
        row: DataFrame row
        icd_cols: Dictionary mapping 'ICD1', 'ICD2', etc. to actual column names
    
    Returns:
        List of ICD codes (empty strings for missing positions)
    """
    icd_list = []
    for icd_num in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
        if icd_num in icd_cols:
            col_name = icd_cols[icd_num]
            value = row[col_name]
            if pd.notna(value) and str(value).strip():
                icd_list.append(str(value).strip().upper())
            else:
                icd_list.append('')
        else:
            icd_list.append('')
    return icd_list


def calculate_accuracy(
    predictions_path: str,
    ground_truth_path: str,
    pdf_mapping: Optional[Dict[str, str]] = None
) -> Tuple[float, float, List[Dict[str, any]]]:
    """
    Calculate CPT and ICD1 accuracy by comparing predictions vs ground truth.
    
    Args:
        predictions_path: Path to predictions CSV/XLSX file
        ground_truth_path: Path to ground truth CSV/XLSX file
        pdf_mapping: Optional mapping from Account ID to PDF file path
    
    Returns:
        Tuple of (cpt_accuracy, icd1_accuracy, error_cases)
        error_cases is a list of dicts with keys: account_id, pdf_path, predicted, expected, error_type
    """
    # Read files
    predictions_df = read_dataframe(predictions_path)
    ground_truth_df = read_dataframe(ground_truth_path)
    
    # Find columns
    # Account ID: prioritize "Account #" format (matches "gos demo.csv" format)
    account_id_col_pred = find_column(predictions_df, ['Account #', 'AccountId', 'Account ID', 'Account', 'ID', 'Acc. #', 'ACC #', 'ACCOUNT #'])
    cpt_col_pred = find_column(predictions_df, ['CPT', 'Cpt', 'ASA Code'])
    
    icd_cols_pred = {}
    for col in predictions_df.columns:
        col_upper = col.upper().strip()
        if col_upper in ['ICD1', 'ICD2', 'ICD3', 'ICD4']:
            icd_cols_pred[col_upper] = col
    
    # Ground truth: prioritize "Account #" format (matches "gos demo.csv" format)
    account_id_col_gt = find_column(ground_truth_df, ['Account #', 'AccountId', 'Account ID', 'Account', 'ID', 'Acc. #', 'ACC #', 'ACCOUNT #'])
    # CPT: support both "CPT" and "Cpt" (case-insensitive matching handles this)
    cpt_col_gt = find_column(ground_truth_df, ['CPT', 'Cpt'])
    # ICD: support both "ICD" and "Icd" (case-insensitive matching handles this)
    icd_col_gt = find_column(ground_truth_df, ['ICD', 'Icd'])
    
    # Validate required columns
    if account_id_col_pred is None:
        raise Exception("Predictions file must have an 'AccountId', 'Account ID', or 'Account #' column")
    if account_id_col_gt is None:
        raise Exception("Ground truth file must have an 'AccountId', 'Account ID', or 'Account #' column")
    
    # CPT columns are optional (may be disabled)
    calculate_cpt = cpt_col_pred is not None
    if calculate_cpt and cpt_col_gt is None:
        raise Exception("Ground truth file must have a 'Cpt' column when CPT predictions are present")
    
    # Build ground truth lookup dictionary (only first occurrence per account ID)
    gt_dict = {}
    for idx, row in ground_truth_df.iterrows():
        account_id = str(row[account_id_col_gt]).strip()
        if account_id not in gt_dict:
            gt_dict[account_id] = {
                'cpt': str(row[cpt_col_gt]).strip() if calculate_cpt and cpt_col_gt and pd.notna(row[cpt_col_gt]) else '',
                'icd': str(row[icd_col_gt]).strip() if icd_col_gt and pd.notna(row[icd_col_gt]) else ''
            }
    
    # Compare predictions with ground truth
    cpt_matches = 0
    cpt_total = 0
    icd1_matches = 0
    icd1_total = 0
    error_cases = []
    
    # Diagnostic counters
    predictions_total = len(predictions_df)
    accounts_not_found = 0
    icd_skipped_no_predicted = 0
    icd_skipped_no_ground_truth = 0
    
    for idx, row in predictions_df.iterrows():
        account_id = str(row[account_id_col_pred]).strip()
        predicted_cpt = str(row[cpt_col_pred]).strip() if calculate_cpt and cpt_col_pred and pd.notna(row[cpt_col_pred]) else ''
        predicted_icd_list = get_predicted_icd_list(row, icd_cols_pred)
        
        if account_id in gt_dict:
            gt_data = gt_dict[account_id]
            
            # Compare CPT (only if CPT columns exist)
            if calculate_cpt:
                gt_cpt = gt_data['cpt']
                cpt_total += 1
                cpt_match = predicted_cpt == gt_cpt
                if cpt_match:
                    cpt_matches += 1
                else:
                    # Log first 5 CPT mismatches for debugging
                    if len(error_cases) < 5:
                        logger.info(f"CPT Mismatch - Account: {account_id}, Predicted: '{predicted_cpt}', Ground Truth: '{gt_cpt}'")
                    pdf_path = pdf_mapping.get(account_id, 'N/A') if pdf_mapping else 'N/A'
                    error_cases.append({
                        'account_id': account_id,
                        'pdf_path': pdf_path,
                        'predicted': predicted_cpt,
                        'expected': gt_cpt,
                        'error_type': 'CPT'
                    })
            
            # Compare ICD1 (only if both predicted and ground truth have codes)
            if icd_col_gt:
                gt_icd_list = parse_icd_codes(gt_data['icd'])
                # Only count cases where BOTH predicted and ground truth have ICD1 codes
                if len(predicted_icd_list) > 0 and len(gt_icd_list) > 0:
                    icd1_total += 1
                    predicted_icd1 = predicted_icd_list[0] if predicted_icd_list[0] else ''
                    gt_icd1 = gt_icd_list[0] if gt_icd_list[0] else ''
                    icd1_match = predicted_icd1 == gt_icd1
                    if icd1_match:
                        icd1_matches += 1
                    else:
                        pdf_path = pdf_mapping.get(account_id, 'N/A') if pdf_mapping else 'N/A'
                        error_cases.append({
                            'account_id': account_id,
                            'pdf_path': pdf_path,
                            'predicted': predicted_icd1,
                            'expected': gt_icd1,
                            'predicted_icd1': predicted_icd_list[0] if len(predicted_icd_list) > 0 else '',
                            'predicted_icd2': predicted_icd_list[1] if len(predicted_icd_list) > 1 else '',
                            'predicted_icd3': predicted_icd_list[2] if len(predicted_icd_list) > 2 else '',
                            'predicted_icd4': predicted_icd_list[3] if len(predicted_icd_list) > 3 else '',
                            'expected_icd1': gt_icd_list[0] if len(gt_icd_list) > 0 else '',
                            'expected_icd2': gt_icd_list[1] if len(gt_icd_list) > 1 else '',
                            'expected_icd3': gt_icd_list[2] if len(gt_icd_list) > 2 else '',
                            'expected_icd4': gt_icd_list[3] if len(gt_icd_list) > 3 else '',
                            'error_type': 'ICD1'
                        })
                elif len(predicted_icd_list) == 0:
                    icd_skipped_no_predicted += 1
                elif len(gt_icd_list) == 0:
                    icd_skipped_no_ground_truth += 1
        else:
            accounts_not_found += 1
    
    # Calculate accuracies (CPT is optional)
    if calculate_cpt:
        cpt_accuracy = cpt_matches / cpt_total if cpt_total > 0 else 0.0
        logger.info(f"CPT Accuracy: {cpt_accuracy:.2%} ({cpt_matches}/{cpt_total})")
        logger.info(f"CPT Diagnostics:")
        logger.info(f"  - CPT total compared: {cpt_total}")
        logger.info(f"  - CPT matches: {cpt_matches}")
        logger.info(f"  - CPT mismatches: {cpt_total - cpt_matches}")
    else:
        cpt_accuracy = None
        logger.info("CPT Accuracy: Not calculated (CPT predictions not present)")
    
    icd1_accuracy = icd1_matches / icd1_total if icd1_total > 0 else 0.0
    logger.info(f"ICD1 Accuracy: {icd1_accuracy:.2%} ({icd1_matches}/{icd1_total})")
    
    # Diagnostic logging
    logger.info(f"Accuracy Diagnostics:")
    logger.info(f"  - Total predictions: {predictions_total}")
    logger.info(f"  - Accounts not found in ground truth: {accounts_not_found}")
    logger.info(f"  - ICD skipped (no predicted code): {icd_skipped_no_predicted}")
    logger.info(f"  - ICD skipped (no ground truth code): {icd_skipped_no_ground_truth}")
    logger.info(f"  - ICD1 total compared: {icd1_total}")
    
    return cpt_accuracy, icd1_accuracy, error_cases


def get_error_cases(
    predictions_path: str,
    ground_truth_path: str,
    error_type: str = 'both',  # 'cpt', 'icd1', or 'both'
    pdf_mapping: Optional[Dict[str, str]] = None,
    limit: int = 10
) -> List[Dict[str, any]]:
    """
    Get error cases (mismatches) between predictions and ground truth.
    
    Args:
        predictions_path: Path to predictions CSV/XLSX file
        ground_truth_path: Path to ground truth CSV/XLSX file
        error_type: Type of errors to return ('cpt', 'icd1', or 'both')
        pdf_mapping: Optional mapping from Account ID to PDF file path
        limit: Maximum number of error cases to return
    
    Returns:
        List of error case dictionaries
    """
    _, _, all_errors = calculate_accuracy(predictions_path, ground_truth_path, pdf_mapping)
    
    # Filter by error type
    if error_type == 'cpt':
        filtered_errors = [e for e in all_errors if e['error_type'] == 'CPT']
    elif error_type == 'icd1':
        filtered_errors = [e for e in all_errors if e['error_type'] == 'ICD1']
    else:
        filtered_errors = all_errors
    
    # Group by error pattern and prioritize diverse errors
    error_patterns = {}
    for error in filtered_errors:
        pattern_key = f"{error['predicted']} -> {error['expected']}"
        if pattern_key not in error_patterns:
            error_patterns[pattern_key] = []
        error_patterns[pattern_key].append(error)
    
    # Select diverse errors (one from each pattern, up to limit)
    selected_errors = []
    
    # If limit is None, return all errors
    if limit is None:
        return filtered_errors
    
    for pattern_errors in error_patterns.values():
        if len(selected_errors) >= limit:
            break
        selected_errors.append(pattern_errors[0])  # Take first from each pattern
    
    # Fill remaining slots with any errors
    remaining = limit - len(selected_errors)
    for error in filtered_errors:
        if len(selected_errors) >= limit:
            break
        if error not in selected_errors:
            selected_errors.append(error)
    
    return selected_errors[:limit]

