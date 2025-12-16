"""
Utility module for exporting data to both CSV and XLSX formats.
This module provides helper functions to save pandas DataFrames to both formats.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_dataframe_dual_format(df, base_path, csv_only=False):
    """
    Save a pandas DataFrame to both CSV and XLSX formats.
    
    Args:
        df: pandas DataFrame to save
        base_path: Base file path (without extension). Files will be saved as:
                   - {base_path}.csv
                   - {base_path}.xlsx
        csv_only: If True, only save CSV format (for backwards compatibility)
    
    Returns:
        tuple: (csv_path, xlsx_path) - paths to the saved files
               xlsx_path will be None if csv_only=True
    """
    try:
        # Ensure base_path is a Path object
        base_path = Path(base_path)
        
        # Remove any existing extension
        if base_path.suffix:
            base_path = base_path.with_suffix('')
        
        # Create parent directory if it doesn't exist
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = base_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to: {csv_path}")
        
        # Save XLSX (unless csv_only is True)
        xlsx_path = None
        if not csv_only:
            xlsx_path = base_path.with_suffix('.xlsx')
            
            # Make a copy to avoid modifying the original dataframe
            df_copy = df.copy()
            
            # Explicitly set ID columns as text to prevent scientific notation in Excel
            id_columns = ['Primary Subsc ID', 'Secondary Subsc ID', 'MRN', 'CSN']
            for col in id_columns:
                if col in df_copy.columns:
                    # Only convert non-empty values to string to avoid 'nan' text
                    df_copy[col] = df_copy[col].apply(lambda x: str(x) if x != '' and pd.notna(x) else '')
            
            # Use openpyxl engine for better compatibility
            df_copy.to_excel(xlsx_path, index=False, engine='openpyxl')
            logger.info(f"Saved XLSX to: {xlsx_path}")
        
        return str(csv_path), str(xlsx_path) if xlsx_path else None
        
    except Exception as e:
        logger.error(f"Error saving dual format: {e}")
        raise


def convert_csv_to_xlsx(csv_path, xlsx_path=None):
    """
    Convert an existing CSV file to XLSX format.
    
    Args:
        csv_path: Path to the CSV file
        xlsx_path: Optional path for XLSX file. If not provided, uses same name with .xlsx extension
    
    Returns:
        str: Path to the created XLSX file
    """
    try:
        csv_path = Path(csv_path)
        
        if xlsx_path is None:
            xlsx_path = csv_path.with_suffix('.xlsx')
        else:
            xlsx_path = Path(xlsx_path)
        
        # Read CSV with dtype=str to preserve data formatting
        df = pd.read_csv(csv_path, dtype=str)
        
        # Explicitly set ID columns as text to prevent scientific notation in Excel
        id_columns = ['Primary Subsc ID', 'Secondary Subsc ID', 'MRN', 'CSN']
        for col in id_columns:
            if col in df.columns:
                # Only convert non-empty values to string to avoid 'nan' text
                df[col] = df[col].apply(lambda x: str(x) if x != '' and pd.notna(x) else '')
        
        # Save as XLSX
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        logger.info(f"Converted {csv_path} to {xlsx_path}")
        
        return str(xlsx_path)
        
    except Exception as e:
        logger.error(f"Error converting CSV to XLSX: {e}")
        raise


def get_output_paths(base_path, format='both'):
    """
    Get output file paths for the requested format(s).
    
    Args:
        base_path: Base file path (can include or exclude extension)
        format: 'csv', 'xlsx', or 'both'
    
    Returns:
        list: List of output file paths for the requested format(s)
    """
    base_path = Path(base_path)
    
    # Remove extension if present
    if base_path.suffix:
        base_path = base_path.with_suffix('')
    
    if format == 'csv':
        return [str(base_path.with_suffix('.csv'))]
    elif format == 'xlsx':
        return [str(base_path.with_suffix('.xlsx'))]
    else:  # 'both'
        return [str(base_path.with_suffix('.csv')), str(base_path.with_suffix('.xlsx'))]

