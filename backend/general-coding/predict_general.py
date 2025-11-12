#!/usr/bin/env python3
"""
General ASA Code Prediction using OpenAI API
Integrated for API usage
"""

import pandas as pd
import os
import base64
import json
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import requests

logger = logging.getLogger(__name__)

def predict_asa_code_general(procedure, preop_diagnosis, postop_diagnosis, cpt_codes_text, model="gpt5", api_key=None):
    """
    Predict ASA code using OpenAI Responses API
    
    Args:
        procedure: Procedure description
        preop_diagnosis: Pre-operative diagnosis
        postop_diagnosis: Post-operative diagnosis
        cpt_codes_text: Reference text containing all valid CPT codes
        model: Model to use (gpt5, gpt-4o, etc.)
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

    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Extract the predicted code from the response
            predicted_code = response.output_text.strip()
            
            # Handle usage and cost calculation
            tokens = 0
            cost = 0.0
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                total_tokens = getattr(usage, 'total_tokens', 0)
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                tokens = total_tokens
                
                # Cost estimation based on model (OpenAI pricing as of 2024)
                if "gpt5" in model or "gpt-4o-mini" in model:
                    input_cost = prompt_tokens * 0.00015 / 1000
                    output_cost = completion_tokens * 0.0006 / 1000
                elif "gpt-4o" in model:
                    input_cost = prompt_tokens * 0.0025 / 1000
                    output_cost = completion_tokens * 0.01 / 1000
                elif "gpt-4-turbo" in model:
                    input_cost = prompt_tokens * 0.01 / 1000
                    output_cost = completion_tokens * 0.03 / 1000
                else:
                    # Default to gpt5 pricing
                    input_cost = prompt_tokens * 0.00015 / 1000
                    output_cost = completion_tokens * 0.0006 / 1000
                
                cost = input_cost + output_cost
            
            return predicted_code, tokens, cost, None
            
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            # Build descriptive error message
            error_message = f"{error_type}: {error_str}"
            
            # Check if it's a 429 rate limit error
            if "429" in error_str or "rate_limit" in error_str.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit error (429), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached for rate limit error: {error_message}")
                    return None, 0, 0.0, f"Rate limit exceeded after {max_retries} attempts: {error_str}"
            else:
                # For non-429 errors, don't retry
                logger.error(f"Error calling OpenAI API: {error_message}")
                return None, 0, 0.0, error_message
    
    # Should never reach here, but just in case
    return None, 0, 0.0, "Max retries reached"


def predict_asa_code_from_images(image_data_list, cpt_codes_text, model="openai/gpt-5:online", api_key=None):
    """
    Predict ASA code using OpenRouter API from PDF page images
    
    Args:
        image_data_list: List of base64 encoded image strings
        cpt_codes_text: Reference text containing all valid CPT codes
        model: Model to use (default: openai/gpt-5:online). For OpenRouter, must use format "openai/gpt-5" or "openai/gpt-5:online"
        api_key: OpenRouter API key
    
    Returns:
        tuple: (predicted_code, tokens_used, cost_estimate, error_message)
    """
    # Get API key
    if api_key:
        api_key_value = api_key
    else:
        api_key_value = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key_value:
        return None, 0, 0.0, "No API key provided"
    
    # Prepare the prompt
    prompt = f"""You are a medical anesthesia CPT coder.

Your task is to predict the most relevant anesthesia CPT code for anesthesia billing for a certain procedure by analyzing the provided medical document page(s).

Here is the reference list of valid anesthesia CPT codes:

{cpt_codes_text}

CRITICAL CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) if ANY of these are present:
     * If the page/document explicitly states "screening colonoscopy" - USE CODE 00812
     * Procedure states "screening"
     * Pre-op diagnosis: [Z12.11] (Encounter for screening colonoscopy)
     * Pre-op diagnosis: [Z80.0] (Family history of colon cancer)
     * Pre-op diagnosis: [Z86.010x] (History of colonic polyps) - this is surveillance screening
     * Pre-op states "Colon cancer screening"
   - Use 00811 (diagnostic colonoscopy) ONLY if:
     * Investigating specific symptoms (bleeding, pain, diarrhea, etc.)
     * NO screening indicators present
   - When uncertain: If ANY screening indicator exists, use 00812

IMPORTANT: Look at the document images carefully to identify:
- Procedure description
- Pre-operative diagnosis
- Post-operative diagnosis
- Any relevant medical information that can help determine the correct anesthesia CPT code

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

Answer with the anesthesia CPT code ONLY, nothing else. For example "00840" - that is your ENTIRE response to me."""

    # Build content list with text prompt first, then images (OpenRouter format)
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]
    
    # Add images to content (OpenRouter format)
    for img_data in image_data_list:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_data}"
            }
        })
    
    # Prepare messages for OpenRouter API
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key_value}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract the predicted code from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                if content:
                    predicted_code = content.strip()
                else:
                    raise Exception(f"Empty response content. Response: {response_data}")
            else:
                error_info = response_data.get('error', response_data)
                raise Exception(f"Unexpected response format. Missing 'choices' field. Error: {error_info}")
            
            # Handle usage and cost calculation
            tokens = 0
            cost = 0.0
            if "usage" in response_data:
                usage = response_data["usage"]
                total_tokens = usage.get("total_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                tokens = total_tokens
                
                # OpenRouter provides cost in response, but we'll estimate if not available
                # Cost estimation based on model (rough estimates for gpt-5)
                if "openai/gpt-5" in model or "gpt-5" in model or "gpt5:online" in model:
                    # Estimate pricing (adjust based on actual OpenRouter pricing)
                    input_cost = prompt_tokens * 0.01 / 1000  # Estimate
                    output_cost = completion_tokens * 0.03 / 1000  # Estimate
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
                    # Default estimate
                    input_cost = prompt_tokens * 0.01 / 1000
                    output_cost = completion_tokens * 0.03 / 1000
                
                cost = input_cost + output_cost
            
            return predicted_code, tokens, cost, None
            
        except requests.exceptions.HTTPError as e:
            error_str = str(e)
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
            
            # Try to extract detailed error message from response
            error_detail = error_str
            try:
                if hasattr(e, 'response') and e.response:
                    response_data = e.response.json()
                    if isinstance(response_data, dict):
                        error_detail = response_data.get('error', {}).get('message', str(e))
                        if not error_detail or error_detail == str(e):
                            error_detail = response_data.get('error', str(e))
            except:
                pass
            
            # Build descriptive error message
            if status_code:
                error_message = f"HTTP {status_code}: {error_detail}"
            else:
                error_message = f"HTTP Error: {error_detail}"
            
            # Retry on all errors
            if attempt < max_retries - 1:
                # Use exponential backoff for rate limits, shorter delay for other errors
                if status_code == 429 or "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit error (429) on image prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                else:
                    wait_time = min(2 ** attempt, 4)  # Cap at 4 seconds for non-rate-limit errors
                    logger.warning(f"HTTP error on image prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for image prediction: {error_message}")
                return None, 0, 0.0, error_message
                
        except requests.exceptions.RequestException as e:
            error_message = f"Request Error: {str(e)}"
            
            # Retry on request errors
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 4)  # Cap at 4 seconds
                logger.warning(f"Request error on image prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for request error on image prediction: {error_message}")
                return None, 0, 0.0, error_message
                
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            # Build descriptive error message
            error_message = f"{error_type}: {error_str}"
            
            # Retry on all errors
            if attempt < max_retries - 1:
                # Use exponential backoff for rate limits, shorter delay for other errors
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit error on image prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                else:
                    wait_time = min(2 ** attempt, 4)  # Cap at 4 seconds for non-rate-limit errors
                    logger.warning(f"Error on image prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for image prediction: {error_message}")
                return None, 0, 0.0, error_message
    
    # Should never reach here, but just in case
    return None, 0, 0.0, "Max retries reached"


def predict_icd_codes_from_images(image_data_list, model="openai/gpt-5:online", api_key=None):
    """
    Predict ICD codes using OpenRouter API from PDF page images
    
    Args:
        image_data_list: List of base64 encoded image strings
        model: Model to use (default: openai/gpt-5:online). For OpenRouter, must use format "openai/gpt-5" or "openai/gpt-5:online"
        api_key: OpenRouter API key
    
    Returns:
        tuple: (icd_codes_dict, tokens_used, cost_estimate, error_message)
        icd_codes_dict: Dictionary with keys 'ICD1', 'ICD2', 'ICD3', 'ICD4' containing ICD codes
    """
    # Get API key
    if api_key:
        api_key_value = api_key
    else:
        api_key_value = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key_value:
        return None, 0, 0.0, "No API key provided"
    
    # Prepare the prompt
    prompt = """You are a medical coding specialist.

Your task is to analyze the provided medical document page(s) and identify ICD diagnosis codes that are relevant to the procedure being performed.

IMPORTANT INSTRUCTIONS:
1. Analyze the entire PDF document carefully to understand the procedure and patient condition
2. Identify the PRIMARY diagnosis (ICD1) - this should be the main reason for the procedure
3. Identify up to 3 additional ICD codes (ICD2, ICD3, ICD4) sorted by relevance to the procedure
4. Only include ICD codes that are directly relevant to the procedure or patient condition
5. If fewer than 4 relevant ICD codes exist, leave the remaining fields empty
6. Use standard ICD-10 format (e.g., "E11.9", "I10", "Z87.891")
7. CRITICAL: Use web search to verify that all ICD codes you provide are valid and current as of November 2025. Only use the most recent ICD codes that are valid in November 2025. Do not use outdated or invalid codes.

OUTPUT FORMAT:
You must respond with ONLY a JSON object in this exact format:
{
  "ICD1": "primary_diagnosis_code",
  "ICD2": "secondary_diagnosis_code_or_empty",
  "ICD3": "tertiary_diagnosis_code_or_empty",
  "ICD4": "quaternary_diagnosis_code_or_empty"
}

If a code doesn't exist, use an empty string "" for that field.

Example response:
{
  "ICD1": "K63.5",
  "ICD2": "E11.9",
  "ICD3": "I10",
  "ICD4": ""
}

Respond with ONLY the JSON object, nothing else."""

    # Build content list with text prompt first, then images (OpenRouter format)
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]
    
    # Add images to content (OpenRouter format)
    for img_data in image_data_list:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_data}"
            }
        })
    
    # Prepare messages for OpenRouter API
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key_value}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract the predicted codes from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content_text = response_data["choices"][0]["message"]["content"]
                if content_text:
                    # Try to parse JSON from response
                    content_text = content_text.strip()
                    
                    # Try to extract JSON if wrapped in markdown code blocks
                    if "```json" in content_text:
                        content_text = content_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in content_text:
                        content_text = content_text.split("```")[1].split("```")[0].strip()
                    
                    try:
                        icd_codes_dict = json.loads(content_text)
                        # Ensure all required keys exist
                        result = {
                            "ICD1": icd_codes_dict.get("ICD1", ""),
                            "ICD2": icd_codes_dict.get("ICD2", ""),
                            "ICD3": icd_codes_dict.get("ICD3", ""),
                            "ICD4": icd_codes_dict.get("ICD4", "")
                        }
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract codes manually
                        logger.warning(f"Failed to parse JSON, attempting manual extraction. Response: {content_text[:200]}")
                        result = {
                            "ICD1": "",
                            "ICD2": "",
                            "ICD3": "",
                            "ICD4": ""
                        }
                        # Try to find ICD codes in the text (basic pattern matching)
                        icd_pattern = r'[A-Z]\d{2}\.?\d*'
                        found_codes = re.findall(icd_pattern, content_text)
                        for i, code in enumerate(found_codes[:4]):
                            result[f"ICD{i+1}"] = code
                else:
                    raise Exception(f"Empty response content. Response: {response_data}")
            else:
                error_info = response_data.get('error', response_data)
                raise Exception(f"Unexpected response format. Missing 'choices' field. Error: {error_info}")
            
            # Handle usage and cost calculation
            tokens = 0
            cost = 0.0
            if "usage" in response_data:
                usage = response_data["usage"]
                total_tokens = usage.get("total_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                tokens = total_tokens
                
                # Cost estimation based on model
                if "openai/gpt-5" in model or "gpt-5" in model or "gpt5:online" in model:
                    input_cost = prompt_tokens * 0.01 / 1000
                    output_cost = completion_tokens * 0.03 / 1000
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
                    input_cost = prompt_tokens * 0.01 / 1000
                    output_cost = completion_tokens * 0.03 / 1000
                
                cost = input_cost + output_cost
            
            return result, tokens, cost, None
            
        except requests.exceptions.HTTPError as e:
            error_str = str(e)
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
            
            # Try to extract detailed error message from response
            error_detail = error_str
            try:
                if hasattr(e, 'response') and e.response:
                    response_data = e.response.json()
                    if isinstance(response_data, dict):
                        error_detail = response_data.get('error', {}).get('message', str(e))
                        if not error_detail or error_detail == str(e):
                            error_detail = response_data.get('error', str(e))
            except:
                pass
            
            # Build descriptive error message
            if status_code:
                error_message = f"HTTP {status_code}: {error_detail}"
            else:
                error_message = f"HTTP Error: {error_detail}"
            
            # Retry on all errors
            if attempt < max_retries - 1:
                if status_code == 429 or "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit error (429) on ICD prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                else:
                    wait_time = min(2 ** attempt, 4)
                    logger.warning(f"HTTP error on ICD prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for ICD prediction: {error_message}")
                return None, 0, 0.0, error_message
                
        except requests.exceptions.RequestException as e:
            error_message = f"Request Error: {str(e)}"
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 4)
                logger.warning(f"Request error on ICD prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for request error on ICD prediction: {error_message}")
                return None, 0, 0.0, error_message
                
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            error_message = f"{error_type}: {error_str}"
            
            if attempt < max_retries - 1:
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit error on ICD prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                else:
                    wait_time = min(2 ** attempt, 4)
                    logger.warning(f"Error on ICD prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for ICD prediction: {error_message}")
                return None, 0, 0.0, error_message
    
    # Should never reach here, but just in case
    return None, 0, 0.0, "Max retries reached"


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


def predict_codes_general_api(input_file, output_file, model="gpt5", api_key=None, max_workers=3, progress_callback=None):
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
            try:
                procedure = str(row.get('Procedure Description', '')) if pd.notna(row.get('Procedure Description')) else ""
                preop = str(row.get('Pre-op diagnosis', '')) if pd.notna(row.get('Pre-op diagnosis')) else ""
                postop = str(row.get('Post-op diagnosis', '')) if pd.notna(row.get('Post-op diagnosis')) else ""
                
                if not procedure or procedure.strip() == "":
                    return idx, "ERROR: Empty procedure", 0, 0.0, "Procedure Description is empty or missing"
                
                predicted_code, tokens, cost, error = predict_asa_code_general(
                    procedure, preop, postop, cpt_codes_text, model, api_key
                )
                
                # If prediction failed, format error message
                if not predicted_code:
                    if error:
                        error_display = f"ERROR: {error[:50]}" if len(error) > 50 else f"ERROR: {error}"
                        return idx, error_display, tokens, cost, error
                    else:
                        return idx, "ERROR: No prediction returned", tokens, cost, "No prediction returned from API"
                
                return idx, predicted_code, tokens, cost, error
            except Exception as e:
                error_msg = f"Unexpected error processing row {idx}: {str(e)}"
                logger.error(error_msg)
                return idx, f"ERROR: {type(e).__name__}", 0, 0.0, error_msg
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, idx, row): idx for idx, row in df.iterrows()}
            
            completed = 0
            for future in as_completed(futures):
                idx, predicted_code, tokens, cost, error = future.result()
                # Use the returned prediction (which may already contain "ERROR: ..." format)
                predictions[idx] = predicted_code if predicted_code else "ERROR: No prediction"
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


def pdf_pages_to_base64_images(pdf_path, n_pages=1, dpi=150):
    """
    Convert first N pages of a PDF to base64 encoded PNG images
    
    Args:
        pdf_path: Path to PDF file
        n_pages: Number of pages to extract (default 1)
        dpi: DPI for image rendering (default 150)
    
    Returns:
        list: List of base64 encoded image strings
    """
    try:
        import fitz  # PyMuPDF
        from io import BytesIO
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_extract = min(n_pages, total_pages)
        
        base64_images = []
        for page_num in range(pages_to_extract):
            page = doc.load_page(page_num)
            
            # Convert to image
            mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to base64
            base64_img = base64.b64encode(img_data).decode('utf-8')
            base64_images.append(base64_img)
        
        doc.close()
        logger.info(f"Extracted {pages_to_extract} page(s) from {pdf_path}")
        return base64_images
        
    except Exception as e:
        logger.error(f"Error extracting pages from {pdf_path}: {e}")
        return []


def predict_codes_from_pdfs_api(pdf_folder, output_file, n_pages=1, model="openai/gpt-5:online", api_key=None, max_workers=3, progress_callback=None):
    """
    Predict ASA codes from PDF files using OpenRouter vision model
    
    Args:
        pdf_folder: Path to folder containing PDF files
        output_file: Path to output CSV file
        n_pages: Number of pages to extract from each PDF (default 1)
        model: OpenRouter model to use (default openai/gpt-5:online). Must use format "openai/gpt-5" or "openai/gpt-5:online" for OpenRouter
        api_key: OpenRouter API key
        max_workers: Number of concurrent threads
        progress_callback: Optional callback function(completed, total, message)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import glob
        from pathlib import Path
        
        logger.info(f"Starting image-based CPT prediction with model: {model}")
        logger.info(f"Extracting {n_pages} page(s) from each PDF")
        
        # Load CPT codes reference
        cpt_codes_text = load_cpt_codes()
        
        # Find all PDF files in folder (recursively search subdirectories)
        pdf_folder_path = Path(pdf_folder)
        
        # Search recursively for PDF files (case-insensitive)
        pdf_files = []
        for ext in ['*.pdf', '*.PDF']:
            pdf_files.extend(pdf_folder_path.glob(f"**/{ext}"))
        
        # Also check case-insensitive manually
        if not pdf_files:
            all_files = list(pdf_folder_path.rglob("*"))
            pdf_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.pdf']
        
        if not pdf_files:
            # Log what files are actually in the directory for debugging
            all_files = list(pdf_folder_path.rglob("*"))
            files_list = [str(f.relative_to(pdf_folder_path)) for f in all_files if f.is_file()][:20]
            logger.error(f"No PDF files found in {pdf_folder}")
            logger.error(f"Files in directory: {files_list}")  # Show first 20 files
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Sort PDF files to ensure consistent order
        pdf_files = sorted(pdf_files, key=lambda x: x.name)
        
        # Initialize result dictionaries (use index to maintain order)
        results = {}
        
        if progress_callback:
            progress_callback(0, len(pdf_files), "Starting PDF image extraction and predictions...")
        
        # Process each PDF
        def process_pdf(idx, pdf_path):
            filename = pdf_path.name
            
            try:
                # Extract pages as base64 images
                image_data_list = pdf_pages_to_base64_images(str(pdf_path), n_pages=n_pages)
                
                if not image_data_list:
                    error_msg = f"Failed to extract PDF pages from {filename}. File may be corrupted or invalid."
                    return idx, filename, "ERROR", 0, 0.0, error_msg, "openrouter_vision"
                
                # Predict ASA code from images
                predicted_code, tokens, cost, error = predict_asa_code_from_images(
                    image_data_list, cpt_codes_text, model, api_key
                )
                
                # If prediction failed, use error message
                if not predicted_code:
                    if error:
                        # Put error in ASA Code column for visibility, but also keep in Error Message
                        return idx, filename, f"ERROR: {error[:50]}", tokens, cost, error, "openrouter_vision"
                    else:
                        return idx, filename, "ERROR: No prediction returned", tokens, cost, "No prediction returned from API", "openrouter_vision"
                
                return idx, filename, predicted_code, tokens, cost, error, "openrouter_vision"
                
            except Exception as e:
                error_msg = f"Unexpected error processing {filename}: {str(e)}"
                logger.error(error_msg)
                return idx, filename, f"ERROR: {type(e).__name__}", 0, 0.0, error_msg, "openrouter_vision"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pdf, idx, pdf_path): idx for idx, pdf_path in enumerate(pdf_files)}
            
            completed = 0
            for future in as_completed(futures):
                idx, filename, predicted_code, tokens, cost, error, model_source = future.result()
                results[idx] = {
                    'filename': filename,
                    'prediction': predicted_code,
                    'tokens': tokens,
                    'cost': cost,
                    'error': error,
                    'model_source': model_source
                }
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(pdf_files), f"Processed {completed}/{len(pdf_files)} PDFs...")
                
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(pdf_files)} completed")
        
        # Convert results to lists in correct order
        filenames = [results[i]['filename'] for i in range(len(pdf_files))]
        predictions = [results[i]['prediction'] for i in range(len(pdf_files))]
        tokens_list = [results[i]['tokens'] for i in range(len(pdf_files))]
        costs_list = [results[i]['cost'] for i in range(len(pdf_files))]
        errors_list = [results[i]['error'] for i in range(len(pdf_files))]
        model_sources = [results[i]['model_source'] for i in range(len(pdf_files))]
        
        # Create dataframe with results
        df = pd.DataFrame({
            'Patient Filename': filenames,
            'ASA Code': predictions,
            'Procedure Code': predictions,
            'Model Source': model_sources,
            'Tokens Used': tokens_list,
            'Cost (USD)': costs_list,
            'Error Message': errors_list
        })
        
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
            progress_callback(len(pdf_files), len(pdf_files), f"Completed! Total cost: ${total_cost:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in predict_codes_from_pdfs_api: {e}")
        return False


def predict_icd_codes_from_pdfs_api(pdf_folder, output_file, n_pages=1, model="openai/gpt-5:online", api_key=None, max_workers=3, progress_callback=None):
    """
    Predict ICD codes from PDF files using OpenRouter vision model
    
    Args:
        pdf_folder: Path to folder containing PDF files
        output_file: Path to output CSV file
        n_pages: Number of pages to extract from each PDF (default 1)
        model: OpenRouter model to use (default openai/gpt-5:online). Must use format "openai/gpt-5" or "openai/gpt-5:online" for OpenRouter
        api_key: OpenRouter API key
        max_workers: Number of concurrent threads
        progress_callback: Optional callback function(completed, total, message)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from pathlib import Path
        
        logger.info(f"Starting image-based ICD prediction with model: {model}")
        logger.info(f"Extracting {n_pages} page(s) from each PDF")
        
        # Find all PDF files in folder (recursively search subdirectories)
        pdf_folder_path = Path(pdf_folder)
        
        # Search recursively for PDF files (case-insensitive)
        pdf_files = []
        for ext in ['*.pdf', '*.PDF']:
            pdf_files.extend(pdf_folder_path.glob(f"**/{ext}"))
        
        # Also check case-insensitive manually
        if not pdf_files:
            all_files = list(pdf_folder_path.rglob("*"))
            pdf_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.pdf']
        
        if not pdf_files:
            # Log what files are actually in the directory for debugging
            all_files = list(pdf_folder_path.rglob("*"))
            files_list = [str(f.relative_to(pdf_folder_path)) for f in all_files if f.is_file()][:20]
            logger.error(f"No PDF files found in {pdf_folder}")
            logger.error(f"Files in directory: {files_list}")  # Show first 20 files
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Sort PDF files to ensure consistent order
        pdf_files = sorted(pdf_files, key=lambda x: x.name)
        
        # Initialize result dictionaries (use index to maintain order)
        results = {}
        
        if progress_callback:
            progress_callback(0, len(pdf_files), "Starting PDF image extraction and ICD predictions...")
        
        # Process each PDF
        def process_pdf(idx, pdf_path):
            filename = pdf_path.name
            
            try:
                # Extract pages as base64 images
                image_data_list = pdf_pages_to_base64_images(str(pdf_path), n_pages=n_pages)
                
                if not image_data_list:
                    error_msg = f"Failed to extract PDF pages from {filename}. File may be corrupted or invalid."
                    return idx, filename, {"ICD1": "ERROR", "ICD2": "", "ICD3": "", "ICD4": ""}, 0, 0.0, error_msg, "openrouter_vision"
                
                # Predict ICD codes from images
                icd_codes_dict, tokens, cost, error = predict_icd_codes_from_images(
                    image_data_list, model, api_key
                )
                
                # If prediction failed, use error message
                if not icd_codes_dict:
                    if error:
                        error_dict = {
                            "ICD1": f"ERROR: {error[:30]}",
                            "ICD2": "",
                            "ICD3": "",
                            "ICD4": ""
                        }
                        return idx, filename, error_dict, tokens, cost, error, "openrouter_vision"
                    else:
                        error_dict = {
                            "ICD1": "ERROR: No prediction",
                            "ICD2": "",
                            "ICD3": "",
                            "ICD4": ""
                        }
                        return idx, filename, error_dict, tokens, cost, "No prediction returned from API", "openrouter_vision"
                
                return idx, filename, icd_codes_dict, tokens, cost, error, "openrouter_vision"
                
            except Exception as e:
                error_msg = f"Unexpected error processing {filename}: {str(e)}"
                logger.error(error_msg)
                error_dict = {
                    "ICD1": f"ERROR: {type(e).__name__}",
                    "ICD2": "",
                    "ICD3": "",
                    "ICD4": ""
                }
                return idx, filename, error_dict, 0, 0.0, error_msg, "openrouter_vision"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pdf, idx, pdf_path): idx for idx, pdf_path in enumerate(pdf_files)}
            
            completed = 0
            for future in as_completed(futures):
                idx, filename, icd_codes_dict, tokens, cost, error, model_source = future.result()
                results[idx] = {
                    'filename': filename,
                    'icd1': icd_codes_dict.get('ICD1', ''),
                    'icd2': icd_codes_dict.get('ICD2', ''),
                    'icd3': icd_codes_dict.get('ICD3', ''),
                    'icd4': icd_codes_dict.get('ICD4', ''),
                    'tokens': tokens,
                    'cost': cost,
                    'error': error,
                    'model_source': model_source
                }
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(pdf_files), f"Processed {completed}/{len(pdf_files)} PDFs...")
                
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(pdf_files)} completed")
        
        # Convert results to lists in correct order
        filenames = [results[i]['filename'] for i in range(len(pdf_files))]
        icd1_list = [results[i]['icd1'] for i in range(len(pdf_files))]
        icd2_list = [results[i]['icd2'] for i in range(len(pdf_files))]
        icd3_list = [results[i]['icd3'] for i in range(len(pdf_files))]
        icd4_list = [results[i]['icd4'] for i in range(len(pdf_files))]
        tokens_list = [results[i]['tokens'] for i in range(len(pdf_files))]
        costs_list = [results[i]['cost'] for i in range(len(pdf_files))]
        errors_list = [results[i]['error'] for i in range(len(pdf_files))]
        model_sources = [results[i]['model_source'] for i in range(len(pdf_files))]
        
        # Create dataframe with results
        df = pd.DataFrame({
            'Patient Filename': filenames,
            'ICD1': icd1_list,
            'ICD2': icd2_list,
            'ICD3': icd3_list,
            'ICD4': icd4_list,
            'Model Source': model_sources,
            'Tokens Used': tokens_list,
            'Cost (USD)': costs_list,
            'Error Message': errors_list
        })
        
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
            progress_callback(len(pdf_files), len(pdf_files), f"Completed! Total cost: ${total_cost:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in predict_icd_codes_from_pdfs_api: {e}")
        return False
