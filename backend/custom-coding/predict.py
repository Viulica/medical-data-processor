#!/usr/bin/env python3
"""
Prediction Script for TAN-ESC
Uses the trained model to predict ASA/Procedure codes
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def predict_codes_api(input_file, output_file, model_dir=None, confidence_threshold=0.5):
    """
    Predict ASA/Procedure codes for a CSV file (API version)
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save output CSV
        model_dir: Directory containing model files (defaults to custom-coding directory)
        confidence_threshold: Minimum confidence to accept predictions (default 0.5)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("🚀 TAN-ESC Code Prediction")
        
        # Set model directory
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)
        
        # Load model components
        logger.info("📦 Loading trained model...")
        try:
            model = joblib.load(model_dir / 'quick_asa_model.pkl')
            vectorizer = joblib.load(model_dir / 'quick_asa_vectorizer.pkl')
            label_encoder = joblib.load(model_dir / 'quick_asa_label_encoder.pkl')
            logger.info("✅ Model loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"❌ Error: Model files not found! {e}")
            return False
        
        # Load input CSV
        logger.info(f"📊 Loading input file: {input_file}")
        try:
            df = pd.read_csv(input_file, dtype=str)
            logger.info(f"✅ Loaded {len(df)} records")
        except FileNotFoundError:
            logger.error(f"❌ Error: Input file '{input_file}' not found!")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            return False
        
        # Check for required column
        if 'Procedure Description' not in df.columns:
            logger.error(f"❌ Error: 'Procedure Description' column not found!")
            logger.error(f"   Available columns: {list(df.columns)}")
            return False
        
        # Prepare procedure text
        logger.info("🔍 Preparing procedure descriptions...")
        procedures = df['Procedure Description'].fillna('').astype(str)
        
        # Filter out empty procedures
        non_empty_mask = procedures.str.strip() != ''
        non_empty_indices = df[non_empty_mask].index
        non_empty_procedures = procedures[non_empty_mask]
        
        logger.info(f"   Found {len(non_empty_procedures)} non-empty procedures")
        
        if len(non_empty_procedures) == 0:
            logger.warning("⚠️  Warning: No procedures to predict!")
            return False
        
        # Vectorize and predict
        logger.info("🧠 Making predictions...")
        X_vec = vectorizer.transform(non_empty_procedures)
        predictions = model.predict(X_vec)
        probabilities = model.predict_proba(X_vec)
        max_probabilities = np.max(probabilities, axis=1)
        
        # Decode predictions
        predicted_codes = label_encoder.inverse_transform(predictions)
        
        # Format codes as 5-digit strings with leading zeros (e.g., 00142)
        formatted_codes = [f"{int(float(code)):05d}" for code in predicted_codes]
        
        # Find position to insert new columns (after Procedure Description)
        insert_index = df.columns.get_loc("Procedure Description") + 1
        
        # Initialize prediction columns and confidence
        predictions_list = [''] * len(df)
        model_sources = [''] * len(df)
        confidences = [''] * len(df)
        
        # Apply predictions
        accepted_count = 0
        rejected_count = 0
        
        for idx, pred_code, confidence in zip(non_empty_indices, formatted_codes, max_probabilities):
            predictions_list[idx] = pred_code
            confidences[idx] = f"{confidence:.3f}"
            
            if confidence >= confidence_threshold:
                model_sources[idx] = "tan-esc"
                accepted_count += 1
            else:
                model_sources[idx] = "tan-esc-low-confidence"
                rejected_count += 1
        
        # Insert columns in the right position
        df.insert(insert_index, "ASA Code", predictions_list)
        df.insert(insert_index + 1, "Procedure Code", predictions_list)
        df.insert(insert_index + 2, "Model Source", model_sources)
        df.insert(insert_index + 3, "Confidence", confidences)
        
        logger.info(f"✅ Predictions completed:")
        logger.info(f"   ✓ Accepted (confidence ≥ {confidence_threshold:.0%}): {accepted_count}")
        logger.info(f"   ✗ Low confidence (< {confidence_threshold:.0%}): {rejected_count}")
        logger.info(f"   Average confidence: {max_probabilities.mean():.3f}")
        
        # Save output
        logger.info(f"💾 Saving results to: {output_file}")
        df.to_csv(output_file, index=False)
        logger.info("✅ Prediction completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return False

def predict_codes(input_file, output_file=None, confidence_threshold=0.5):
    """
    Predict ASA/Procedure codes for a CSV file (CLI version)
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save output CSV (optional, defaults to input_file with '_predicted' suffix)
        confidence_threshold: Minimum confidence to accept predictions (default 0.5)
    """
    print("🚀 TAN-ESC Code Prediction")
    print("=" * 60)
    
    # Load model components
    print("📦 Loading trained model...")
    try:
        model_dir = Path(__file__).parent
        model = joblib.load(model_dir / 'quick_asa_model.pkl')
        vectorizer = joblib.load(model_dir / 'quick_asa_vectorizer.pkl')
        label_encoder = joblib.load(model_dir / 'quick_asa_label_encoder.pkl')
        print("✅ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found! Please train the model first.")
        print(f"   Missing file: {e}")
        return
    
    # Load input CSV
    print(f"📊 Loading input file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found!")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Check for required column
    if 'Procedure Description' not in df.columns:
        print(f"❌ Error: 'Procedure Description' column not found!")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Prepare procedure text
    print("🔍 Preparing procedure descriptions...")
    procedures = df['Procedure Description'].fillna('').astype(str)
    
    # Filter out empty procedures
    non_empty_mask = procedures.str.strip() != ''
    non_empty_indices = df[non_empty_mask].index
    non_empty_procedures = procedures[non_empty_mask]
    
    print(f"   Found {len(non_empty_procedures)} non-empty procedures")
    
    if len(non_empty_procedures) == 0:
        print("⚠️  Warning: No procedures to predict!")
        return
    
    # Vectorize and predict
    print("🧠 Making predictions...")
    X_vec = vectorizer.transform(non_empty_procedures)
    predictions = model.predict(X_vec)
    probabilities = model.predict_proba(X_vec)
    max_probabilities = np.max(probabilities, axis=1)
    
    # Decode predictions
    predicted_codes = label_encoder.inverse_transform(predictions)
    
    # Format codes as 5-digit strings with leading zeros (e.g., 00142)
    formatted_codes = [f"{int(float(code)):05d}" for code in predicted_codes]
    
    # Initialize columns if they don't exist
    if 'ASA Code' not in df.columns:
        df['ASA Code'] = ''
    if 'Procedure Code' not in df.columns:
        df['Procedure Code'] = ''
    
    # Apply predictions
    accepted_count = 0
    rejected_count = 0
    
    for idx, pred_code, confidence in zip(non_empty_indices, formatted_codes, max_probabilities):
        if confidence >= confidence_threshold:
            df.at[idx, 'ASA Code'] = pred_code
            df.at[idx, 'Procedure Code'] = pred_code
            accepted_count += 1
        else:
            rejected_count += 1
    
    print(f"✅ Predictions completed:")
    print(f"   ✓ Accepted (confidence ≥ {confidence_threshold:.0%}): {accepted_count}")
    print(f"   ✗ Rejected (confidence < {confidence_threshold:.0%}): {rejected_count}")
    
    # Show statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Average confidence: {max_probabilities.mean():.3f}")
    print(f"   Min confidence: {max_probabilities.min():.3f}")
    print(f"   Max confidence: {max_probabilities.max():.3f}")
    
    # Show confidence distribution
    print(f"\n📈 Confidence Distribution:")
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    for i in range(len(thresholds) - 1):
        low = thresholds[i]
        high = thresholds[i + 1]
        count = np.sum((max_probabilities >= low) & (max_probabilities < high))
        pct = count / len(max_probabilities) * 100
        print(f"   {low:.2f} - {high:.2f}: {count:4d} ({pct:5.1f}%)")
    
    # Show predicted code distribution
    print(f"\n🏷️  Predicted Code Distribution (accepted only):")
    accepted_mask = max_probabilities >= confidence_threshold
    if accepted_count > 0:
        accepted_formatted_codes = [code for code, conf in zip(formatted_codes, max_probabilities) if conf >= confidence_threshold]
        code_counts = pd.Series(accepted_formatted_codes).value_counts()
        for code, count in code_counts.head(10).items():
            pct = count / accepted_count * 100
            print(f"   {code}: {count:4d} ({pct:5.1f}%)")
        if len(code_counts) > 10:
            print(f"   ... and {len(code_counts) - 10} more codes")
    
    # Show sample predictions
    print(f"\n🔍 Sample Predictions:")
    sample_indices = non_empty_indices[:5]
    for idx in sample_indices:
        proc = df.at[idx, 'Procedure Description'][:60]
        pred_idx = list(non_empty_indices).index(idx)
        pred_code = formatted_codes[pred_idx]
        confidence = max_probabilities[pred_idx]
        status = "✅" if confidence >= confidence_threshold else "❌"
        print(f"   {status} '{proc}...'")
        print(f"      → Code: {pred_code} (confidence: {confidence:.3f})")
    
    # Save output
    if output_file is None:
        # Generate output filename
        if input_file.endswith('.csv'):
            output_file = input_file.replace('.csv', '_predicted.csv')
        else:
            output_file = input_file + '_predicted.csv'
    
    print(f"\n💾 Saving results to: {output_file}")
    df.to_csv(output_file, index=False)
    print("✅ Prediction completed successfully!")
    
    # Show low confidence predictions for review
    if rejected_count > 0:
        print(f"\n⚠️  Low Confidence Predictions (may need manual review):")
        low_conf_mask = max_probabilities < confidence_threshold
        low_conf_indices = [idx for idx, is_low in zip(non_empty_indices, low_conf_mask) if is_low][:5]
        
        for idx in low_conf_indices:
            proc = df.at[idx, 'Procedure Description'][:60]
            pred_idx = list(non_empty_indices).index(idx)
            pred_code = formatted_codes[pred_idx]
            confidence = max_probabilities[pred_idx]
            print(f"   ⚠️  '{proc}...'")
            print(f"      → Suggested Code: {pred_code} (confidence: {confidence:.3f})")
    
    return df

if __name__ == "__main__":
    import sys
    
    # Default input file
    input_file = "TAN-ESC 09-30-2025#2193.AI.csv"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    confidence_threshold = 0.5  # Accept predictions with 50%+ confidence
    if len(sys.argv) > 2:
        confidence_threshold = float(sys.argv[2])
    
    output_file = None
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    # Run prediction
    predict_codes(input_file, output_file, confidence_threshold)

