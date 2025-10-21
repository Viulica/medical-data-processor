#!/usr/bin/env python3
"""
General ASA Code Prediction using OpenAI API
Integrated for API usage
"""

import pandas as pd
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

def predict_asa_code_general(procedure, preop_diagnosis, postop_diagnosis, cpt_codes_text, model="gpt-5", api_key=None):
    """
    Predict ASA code using OpenAI API with web search
    
    Args:
        procedure: Procedure description
        preop_diagnosis: Pre-operative diagnosis
        postop_diagnosis: Post-operative diagnosis
        cpt_codes_text: Reference text containing all valid CPT codes
        model: Model to use (gpt-5, gpt-4o, etc.)
        api_key: OpenAI API key
    
    Returns:
        tuple: (predicted_code, tokens_used, cost_estimate, error_message)
    """
    # Initialize OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare the prompt
    prompt = f"""You are a medical anesthesia CPT coder.

Your task is to predict the most relevant anesthesia CPT code for anesthesia billing for a certain procedure.

Here is the reference list of valid anesthesia CPT codes:

{cpt_codes_text}

CRITICAL CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) if ANY of these are present:
     * Procedure states "screening"
     * Pre-op diagnosis: [Z12.11] (Encounter for screening colonoscopy)
     * Pre-op diagnosis: [Z80.0] (Family history of colon cancer)
     * Pre-op diagnosis: [Z86.010x] (History of colonic polyps) - this is surveillance screening
     * Pre-op states "Colon cancer screening"
   - Use 00811 (diagnostic colonoscopy) ONLY if:
     * Investigating specific symptoms (bleeding, pain, diarrhea, etc.)
     * NO screening indicators present
   - When uncertain: If ANY screening indicator exists, use 00812

IMPORTANT: The Procedure field is REQUIRED. Pre-operative and Post-operative diagnoses are SUPPLEMENTARY only.
If only the procedure is present and any other information is not available, do your best to determine the correct anesthesia CPT code based on the procedure text alone.

Here is the clinical information:
- Procedure: {procedure}
- Pre-operative diagnosis: {preop_diagnosis}
- Post-operative diagnosis: {postop_diagnosis}

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

Answer with the anesthesia CPT code ONLY, nothing else. For example "00840" - that is your ENTIRE response to me."""

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ],
            text={
                "format": {
                    "type": "text"
                },
                "verbosity": "medium"
            },
            tools=[
                {
                    "type": "web_search",
                    "user_location": {
                        "type": "approximate"
                    },
                    "search_context_size": "medium"
                }
            ],
            store=True,
            include=[
                "web_search_call.action.sources"
            ]
        )
        
        # Extract the predicted code from the response
        predicted_code = response.output_text.strip()
        
        # Handle usage and cost calculation
        tokens = 0
        cost = 0.0
        if hasattr(response, 'usage'):
            usage = response.usage
            total_tokens = getattr(usage, 'total_tokens', 0)
            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            completion_tokens = getattr(usage, 'completion_tokens', 0)
            tokens = total_tokens
            
            # Cost estimation based on model
            if "gpt-5" in model:
                input_cost = prompt_tokens * 0.00015 / 1000
                output_cost = completion_tokens * 0.0006 / 1000
            elif "gpt-4o-mini" in model:
                input_cost = prompt_tokens * 0.00015 / 1000
                output_cost = completion_tokens * 0.0006 / 1000
            elif "gpt-4o" in model:
                input_cost = prompt_tokens * 0.0025 / 1000
                output_cost = completion_tokens * 0.01 / 1000
            elif "gpt-4-turbo" in model:
                input_cost = prompt_tokens * 0.01 / 1000
                output_cost = completion_tokens * 0.03 / 1000
            else:
                input_cost = 0
                output_cost = 0
            
            cost = input_cost + output_cost
        
        return predicted_code, tokens, cost, None
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None, 0, 0.0, str(e)


def load_cpt_codes():
    """Load CPT codes from cpt_codes.txt"""
    try:
        # Try to load from general-coding directory
        cpt_file = os.path.join(os.path.dirname(__file__), 'cpt_codes.txt')
        if not os.path.exists(cpt_file):
            # Try parent directory
            cpt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cpt_codes.txt')
        
        with open(cpt_file, 'r') as f:
            cpt_codes_text = f.read()
        logger.info(f"✅ Loaded CPT codes reference from {cpt_file}")
        return cpt_codes_text
    except FileNotFoundError:
        logger.warning("⚠️  cpt_codes.txt not found, using empty reference")
        return ""


def predict_codes_general_api(input_file, output_file, model="gpt-5", api_key=None, max_workers=5, progress_callback=None):
    """
    Predict ASA codes for a CSV file using OpenAI general model
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        model: OpenAI model to use
        api_key: OpenAI API key
        max_workers: Number of concurrent threads
        progress_callback: Optional callback function(completed, total, message)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Starting general CPT prediction with model: {model}")
        
        # Load CPT codes reference
        cpt_codes_text = load_cpt_codes()
        
        # Read input CSV
        try:
            df = pd.read_csv(input_file, encoding='utf-8', dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding='latin-1', dtype=str)
        
        logger.info(f"Loaded {len(df)} records from {input_file}")
        
        # Verify required columns
        required_cols = ["Procedure Description"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Add optional diagnosis columns if they don't exist
        if "Pre-op diagnosis" not in df.columns:
            df["Pre-op diagnosis"] = ""
            logger.warning("Pre-op diagnosis column not found, using empty values")
        if "Post-op diagnosis" not in df.columns:
            df["Post-op diagnosis"] = ""
            logger.warning("Post-op diagnosis column not found, using empty values")
        
        # Initialize result columns
        predictions = [None] * len(df)
        tokens_list = [0] * len(df)
        costs_list = [0.0] * len(df)
        errors_list = [None] * len(df)
        model_sources = ["openai_general"] * len(df)
        
        if progress_callback:
            progress_callback(0, len(df), "Starting predictions...")
        
        # Process predictions with threading
        def process_row(idx, row):
            procedure = str(row.get('Procedure Description', '')) if pd.notna(row.get('Procedure Description')) else ""
            preop = str(row.get('Pre-op diagnosis', '')) if pd.notna(row.get('Pre-op diagnosis')) else ""
            postop = str(row.get('Post-op diagnosis', '')) if pd.notna(row.get('Post-op diagnosis')) else ""
            
            predicted_code, tokens, cost, error = predict_asa_code_general(
                procedure, preop, postop, cpt_codes_text, model, api_key
            )
            
            return idx, predicted_code, tokens, cost, error
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, idx, row): idx for idx, row in df.iterrows()}
            
            completed = 0
            for future in as_completed(futures):
                idx, predicted_code, tokens, cost, error = future.result()
                predictions[idx] = predicted_code if predicted_code else "ERROR"
                tokens_list[idx] = tokens
                costs_list[idx] = cost
                errors_list[idx] = error
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(df), f"Processed {completed}/{len(df)} procedures...")
                
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(df)} completed")
        
        # Add predictions to dataframe
        insert_index = df.columns.get_loc("Procedure Description") + 1
        df.insert(insert_index, "ASA Code", predictions)
        df.insert(insert_index + 1, "Procedure Code", predictions)
        df.insert(insert_index + 2, "Model Source", model_sources)
        df.insert(insert_index + 3, "Tokens Used", tokens_list)
        df.insert(insert_index + 4, "Cost (USD)", costs_list)
        df.insert(insert_index + 5, "Error Message", errors_list)
        
        # Calculate totals
        total_tokens = sum(tokens_list)
        total_cost = sum(costs_list)
        error_count = sum(1 for e in errors_list if e is not None)
        
        logger.info(f"Total tokens used: {total_tokens:,}")
        logger.info(f"Total estimated cost: ${total_cost:.4f}")
        logger.info(f"Errors encountered: {error_count}")
        
        # Save output
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        if progress_callback:
            progress_callback(len(df), len(df), f"Completed! Total cost: ${total_cost:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in predict_codes_general_api: {e}")
        return False
