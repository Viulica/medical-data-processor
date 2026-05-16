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
from pathlib import Path
import requests

# Google GenAI SDK imports
try:
    import google.genai as genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    genai = None
    types = None

logger = logging.getLogger(__name__)

# ICD code corrections: old deprecated codes → current valid codes
ICD_CODE_CORRECTIONS = {
    "Z86.010": "Z86.0100",
}

def correct_icd_codes(result_dict):
    """Apply hard corrections to ICD code predictions."""
    for key in ("ICD1", "ICD2", "ICD3", "ICD4"):
        val = result_dict.get(key, "")
        if val in ICD_CODE_CORRECTIONS:
            result_dict[key] = ICD_CODE_CORRECTIONS[val]
    return result_dict



def is_gemini_model(model_name):
    """Check if model name indicates a Gemini model (not OpenRouter format)"""
    # Remove OpenRouter prefixes/suffixes if present
    clean_model = model_name.replace('google/', '').replace(':online', '')
    return clean_model.startswith('gemini') or 'gemini' in clean_model.lower()


def normalize_gemini_model(model_name):
    """Normalize Gemini model name by removing OpenRouter prefixes/suffixes"""
    # Remove OpenRouter format prefixes/suffixes and models/ prefix
    clean_model = model_name.replace('google/', '').replace(':online', '').replace('models/', '')
    return clean_model


def predict_asa_code_from_images_gemini(image_data_list, cpt_codes_text, model="gemini-3-flash-preview", api_key=None, custom_instructions=None, include_code_list=True, web_search=True):
    """
    Predict ASA code using Google GenAI SDK from PDF page images

    Args:
        image_data_list: List of base64 encoded image strings
        cpt_codes_text: Reference text containing all valid CPT codes (ignored if include_code_list=False)
        model: Gemini model name (e.g., "gemini-3-flash-preview", "gemini-2.5-pro")
        api_key: Google API key
        custom_instructions: Optional custom instructions to append to the prompt
        include_code_list: Whether to include the complete CPT code list in the prompt (default True)
        web_search: Whether to enable web search for code validation (default True)

    Returns:
        tuple: (predicted_code, tokens_used, cost_estimate, error_message)
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return None, 0, 0.0, "Google GenAI SDK not available. Install with: pip install google-genai"
    
    # Normalize model name
    model = normalize_gemini_model(model)
    
    # Get API key
    if api_key:
        api_key_value = api_key
    else:
        api_key_value = os.getenv("GOOGLE_API_KEY")
    
    if not api_key_value:
        return None, 0, 0.0, "No Google API key provided"
    
    # Initialize client
    try:
        client = genai.Client(api_key=api_key_value)
    except Exception as e:
        return None, 0, 0.0, f"Failed to initialize Google GenAI client: {str(e)}"
    
    # Prepare the prompt (same as OpenRouter version)
    if include_code_list:
        prompt = f"""You are a medical anesthesia CPT coder.

Your task is to predict the most relevant anesthesia CPT code for anesthesia billing for a certain procedure by analyzing the provided medical document page(s).

Here is the reference list of valid anesthesia CPT codes:

{cpt_codes_text}

CRITICAL CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) ONLY if:
     * The document explicitly states "screening colonoscopy"
     * Procedure description includes the word "screening"
     * Pre-op diagnosis is Z12.11 (Encounter for screening colonoscopy) with NO symptom or surveillance indication
     * Pre-op states "Colon cancer screening"
     * Patient has ONLY family history (Z80.0) as indication — no personal history, no symptoms
   - Use 00811 (diagnostic colonoscopy) if ANY of these are present:
     * Indication says "polyp surveillance", "surveillance colonoscopy", or "follow-up"
     * Patient has personal history of colon polyps (Z86.010x) or colon cancer — this is SURVEILLANCE, NOT screening
     * Investigating specific symptoms (bleeding, pain, diarrhea, anemia, weight loss, etc.)
     * Any GI diagnosis as indication (GERD, diverticulosis, IBD, etc.)
   - Key distinction: "screening" = routine, no prior findings. "surveillance" = follow-up due to prior polyps/cancer = DIAGNOSTIC = 00811
   - When uncertain between screening and surveillance: if patient has ANY personal history of polyps or cancer, use 00811

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

4. PERCUTANEOUS LUMBAR SPINAL INTERVENTION CODING:
   - If the procedure is a percutaneous lumbar spinal intervention (such as a Medial Branch Block or Facet Injection) -> use 01938

IMPORTANT: Look at the document images carefully to identify:
- Procedure description
- Pre-operative diagnosis
- Post-operative diagnosis
- Any relevant medical information that can help determine the correct anesthesia CPT code

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

You must respond with a JSON object in this exact format:
{{
  "code": "00840",
  "explanation": "Brief explanation of why this code was chosen (1-2 sentences)"
}}

The explanation should briefly describe why this specific CPT code is appropriate for this procedure. Keep it concise (1-2 sentences maximum).

Respond with ONLY the JSON object, nothing else."""
    else:
        prompt = """You are a medical anesthesia CPT coder.

Your task is to predict the most relevant anesthesia CPT code for anesthesia billing for a certain procedure by analyzing the provided medical document page(s).

CRITICAL CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) ONLY if:
     * The document explicitly states "screening colonoscopy"
     * Procedure description includes the word "screening"
     * Pre-op diagnosis is Z12.11 (Encounter for screening colonoscopy) with NO symptom or surveillance indication
     * Pre-op states "Colon cancer screening"
     * Patient has ONLY family history (Z80.0) as indication — no personal history, no symptoms
   - Use 00811 (diagnostic colonoscopy) if ANY of these are present:
     * Indication says "polyp surveillance", "surveillance colonoscopy", or "follow-up"
     * Patient has personal history of colon polyps (Z86.010x) or colon cancer — this is SURVEILLANCE, NOT screening
     * Investigating specific symptoms (bleeding, pain, diarrhea, anemia, weight loss, etc.)
     * Any GI diagnosis as indication (GERD, diverticulosis, IBD, etc.)
   - Key distinction: "screening" = routine, no prior findings. "surveillance" = follow-up due to prior polyps/cancer = DIAGNOSTIC = 00811
   - When uncertain between screening and surveillance: if patient has ANY personal history of polyps or cancer, use 00811

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

4. PERCUTANEOUS LUMBAR SPINAL INTERVENTION CODING:
   - If the procedure is a percutaneous lumbar spinal intervention (such as a Medial Branch Block or Facet Injection) -> use 01938

IMPORTANT: Look at the document images carefully to identify:
- Procedure description
- Pre-operative diagnosis
- Post-operative diagnosis
- Any relevant medical information that can help determine the correct anesthesia CPT code

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

You must respond with a JSON object in this exact format:
{{
  "code": "00840",
  "explanation": "Brief explanation of why this code was chosen (1-2 sentences)"
}}

The explanation should briefly describe why this specific CPT code is appropriate for this procedure. Keep it concise (1-2 sentences maximum).

Respond with ONLY the JSON object, nothing else."""
    
    # Try loading prompt from database (overrides hardcoded if available)
    db_cpt_prompt = _get_cpt_prompt(cpt_codes_text, include_code_list)
    if db_cpt_prompt:
        prompt = db_cpt_prompt

    # Append custom instructions if provided
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{custom_instructions.strip()}"

    # Build content with images
    parts = [types.Part.from_text(text=prompt)]

    # Add images from base64 strings
    for img_data in image_data_list:
        # Decode base64 to bytes
        img_bytes = base64.b64decode(img_data)
        parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
    
    contents = [types.Content(role="user", parts=parts)]
    
    # Configure thinking for Gemini 3 / 3.1 models
    if model in ("gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-pro-preview"):
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)

    # Conditionally enable web search for Gemini models (CPT prediction)
    tools = None
    if web_search:
        tools = [
            types.Tool(googleSearch=types.GoogleSearch()),
        ]

    use_flex = True  # Start with flex tier (50% cheaper)
    FLEX_TIMEOUT = 600  # 10 minutes

    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            service_tier = "flex" if use_flex else "standard"
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                thinking_config=thinking_config,
                tools=tools if tools else None,
                http_options=types.HttpOptions(
                    extra_body={"serviceTier": service_tier},
                    timeout=FLEX_TIMEOUT * 1000 if use_flex else None,
                ),
            )

            # Call Gemini API
            full_response = ""
            request_start = time.time()
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:
                    full_response += chunk.text
                if use_flex and (time.time() - request_start) > FLEX_TIMEOUT:
                    raise TimeoutError("Flex request exceeded 10 minute timeout")

            response_text = full_response.strip()

            if not response_text:
                raise ValueError("Empty response from API")

            # Clean up response (remove markdown code blocks if present)
            cleaned_response = response_text
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parse JSON
            result = json.loads(cleaned_response)
            predicted_code = result.get("code", "").strip()
            explanation = result.get("explanation", "").strip()

            if not predicted_code:
                raise ValueError("No code found in response")

            prompt_chars = len(prompt)
            image_tokens_estimate = len(image_data_list) * 1000
            response_chars = len(response_text)
            total_tokens_estimate = prompt_chars + image_tokens_estimate + response_chars

            if "gemini-flash-lite" in model:
                cost = total_tokens_estimate * 0.00005 / 1000
            elif "gemini-3-flash" in model:
                cost = total_tokens_estimate * 0.000125 / 1000
            elif "gemini-3-pro" in model or "gemini-3.1-pro" in model:
                cost = total_tokens_estimate * 0.00125 / 1000
            elif "gemini-2.5-flash" in model:
                cost = total_tokens_estimate * 0.000075 / 1000
            else:
                cost = total_tokens_estimate * 0.0005 / 1000

            # Halve the cost if flex was used
            if use_flex:
                cost *= 0.5

            return predicted_code, explanation, total_tokens_estimate, cost, None

        except json.JSONDecodeError as e:
            error_message = f"JSON parsing failed: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 4)
                logger.warning(f"JSON parsing error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for JSON parsing: {error_message}")
                code_match = re.search(r'\b0\d{4}\b', response_text)
                if code_match:
                    predicted_code = code_match.group(0)
                    return predicted_code, response_text.replace(predicted_code, "").strip(), 0, 0.0, None
                return None, "", 0, 0.0, error_message

        except (TimeoutError, Exception) as e:
            if use_flex and not isinstance(e, json.JSONDecodeError):
                reason = "timeout" if isinstance(e, TimeoutError) else str(e)[:80]
                logger.warning(f"Flex failed ({reason}), switching to standard tier")
                use_flex = False
                continue  # Retry immediately with standard

            error_message = f"API error: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 4)
                logger.warning(f"API error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached: {error_message}")
                return None, "", 0, 0.0, error_message

    return None, "", 0, 0.0, "Max retries reached"


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

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

4. PERCUTANEOUS LUMBAR SPINAL INTERVENTION CODING:
   - If the procedure is a percutaneous lumbar spinal intervention (such as a Medial Branch Block or Facet Injection) -> use 01938

IMPORTANT: The Procedure field is REQUIRED. Pre-operative and Post-operative diagnoses are SUPPLEMENTARY only.
If only the procedure is present and any other information is not available, do your best to determine the correct anesthesia CPT code based on the procedure text alone.

Here is the clinical information:
- Procedure: {procedure}
- Pre-operative diagnosis: {preop_diagnosis}
- Post-operative diagnosis: {postop_diagnosis}

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

You must respond with a JSON object in this exact format:
{{
  "code": "00840",
  "explanation": "Brief explanation of why this code was chosen (1-2 sentences)"
}}

The explanation should briefly describe why this specific CPT code is appropriate for this procedure. Keep it concise (1-2 sentences maximum).

Respond with ONLY the JSON object, nothing else."""

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
            
            # Extract the predicted code and explanation from the response
            response_text = response.output_text.strip()
            
            # Try to parse JSON response
            predicted_code = None
            explanation = ""
            try:
                # Try to extract JSON if wrapped in markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(response_text)
                predicted_code = result.get("code", "").strip()
                explanation = result.get("explanation", "").strip()
            except (json.JSONDecodeError, AttributeError):
                # Fallback: try to extract code if JSON parsing fails
                # Look for code pattern (5 digits starting with 0)
                code_match = re.search(r'\b0\d{4}\b', response_text)
                if code_match:
                    predicted_code = code_match.group(0)
                    explanation = response_text.replace(predicted_code, "").strip()
                else:
                    # Last resort: use entire response as code
                    predicted_code = response_text
                    explanation = ""
            
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
            
            return predicted_code, explanation, tokens, cost, None
            
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
                    return None, "", 0, 0.0, f"Rate limit exceeded after {max_retries} attempts: {error_str}"
            else:
                # For non-429 errors, don't retry
                logger.error(f"Error calling OpenAI API: {error_message}")
                return None, "", 0, 0.0, error_message
    
    # Should never reach here, but just in case
    return None, "", 0, 0.0, "Max retries reached"


def predict_asa_code_from_images(image_data_list, cpt_codes_text, model="openai/gpt-5.2:online", api_key=None, custom_instructions=None, include_code_list=True, web_search=True):
    """
    Predict ASA code using OpenRouter API or Google GenAI SDK from PDF page images

    Args:
        image_data_list: List of base64 encoded image strings
        cpt_codes_text: Reference text containing all valid CPT codes (ignored if include_code_list=False)
        model: Model to use (default: openai/gpt-5.2:online). For Gemini, use format "gemini-3-flash-preview" or "gemini-2.5-pro"
        api_key: API key (OpenRouter API key for OpenAI models, Google API key for Gemini models)
        custom_instructions: Optional custom instructions to append to the prompt
        include_code_list: Whether to include the complete CPT code list in the prompt (default True)
        web_search: Whether to enable web search for code validation (default True)

    Returns:
        tuple: (predicted_code, tokens_used, cost_estimate, error_message)
    """
    # Check if using Gemini model
    if is_gemini_model(model):
        # Use Google GenAI SDK directly if GOOGLE_API_KEY is available
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if google_api_key and GOOGLE_GENAI_AVAILABLE:
            logger.info(f"Using Google GenAI SDK directly for Gemini model '{model}'")
            return predict_asa_code_from_images_gemini(image_data_list, cpt_codes_text, model, api_key, custom_instructions, include_code_list, web_search)
        else:
            # Fall back to OpenRouter with google/ prefix
            logger.info(f"No GOOGLE_API_KEY found, routing Gemini model '{model}' through OpenRouter")
            model = f"google/{normalize_gemini_model(model)}"
            # Fall through to OpenRouter path below

    # Use OpenRouter for non-Gemini models (or Gemini models without GOOGLE_API_KEY)
    # Get API key
    if api_key:
        api_key_value = api_key
    else:
        api_key_value = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key_value:
        return None, 0, 0.0, "No API key provided"

    # Prepare the prompt
    if include_code_list:
        prompt = f"""You are a medical anesthesia CPT coder.

Your task is to predict the most relevant anesthesia CPT code for anesthesia billing for a certain procedure by analyzing the provided medical document page(s).

Here is the reference list of valid anesthesia CPT codes:

{cpt_codes_text}

CRITICAL CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) ONLY if:
     * The document explicitly states "screening colonoscopy"
     * Procedure description includes the word "screening"
     * Pre-op diagnosis is Z12.11 (Encounter for screening colonoscopy) with NO symptom or surveillance indication
     * Pre-op states "Colon cancer screening"
     * Patient has ONLY family history (Z80.0) as indication — no personal history, no symptoms
   - Use 00811 (diagnostic colonoscopy) if ANY of these are present:
     * Indication says "polyp surveillance", "surveillance colonoscopy", or "follow-up"
     * Patient has personal history of colon polyps (Z86.010x) or colon cancer — this is SURVEILLANCE, NOT screening
     * Investigating specific symptoms (bleeding, pain, diarrhea, anemia, weight loss, etc.)
     * Any GI diagnosis as indication (GERD, diverticulosis, IBD, etc.)
   - Key distinction: "screening" = routine, no prior findings. "surveillance" = follow-up due to prior polyps/cancer = DIAGNOSTIC = 00811
   - When uncertain between screening and surveillance: if patient has ANY personal history of polyps or cancer, use 00811

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

4. PERCUTANEOUS LUMBAR SPINAL INTERVENTION CODING:
   - If the procedure is a percutaneous lumbar spinal intervention (such as a Medial Branch Block or Facet Injection) -> use 01938

IMPORTANT: Look at the document images carefully to identify:
- Procedure description
- Pre-operative diagnosis
- Post-operative diagnosis
- Any relevant medical information that can help determine the correct anesthesia CPT code

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

You must respond with a JSON object in this exact format:
{{
  "code": "00840",
  "explanation": "Brief explanation of why this code was chosen (1-2 sentences)"
}}

The explanation should briefly describe why this specific CPT code is appropriate for this procedure. Keep it concise (1-2 sentences maximum).

Respond with ONLY the JSON object, nothing else."""
    else:
        prompt = """You are a medical anesthesia CPT coder.

Your task is to predict the most relevant anesthesia CPT code for anesthesia billing for a certain procedure by analyzing the provided medical document page(s).

CRITICAL CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) ONLY if:
     * The document explicitly states "screening colonoscopy"
     * Procedure description includes the word "screening"
     * Pre-op diagnosis is Z12.11 (Encounter for screening colonoscopy) with NO symptom or surveillance indication
     * Pre-op states "Colon cancer screening"
     * Patient has ONLY family history (Z80.0) as indication — no personal history, no symptoms
   - Use 00811 (diagnostic colonoscopy) if ANY of these are present:
     * Indication says "polyp surveillance", "surveillance colonoscopy", or "follow-up"
     * Patient has personal history of colon polyps (Z86.010x) or colon cancer — this is SURVEILLANCE, NOT screening
     * Investigating specific symptoms (bleeding, pain, diarrhea, anemia, weight loss, etc.)
     * Any GI diagnosis as indication (GERD, diverticulosis, IBD, etc.)
   - Key distinction: "screening" = routine, no prior findings. "surveillance" = follow-up due to prior polyps/cancer = DIAGNOSTIC = 00811
   - When uncertain between screening and surveillance: if patient has ANY personal history of polyps or cancer, use 00811

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

4. PERCUTANEOUS LUMBAR SPINAL INTERVENTION CODING:
   - If the procedure is a percutaneous lumbar spinal intervention (such as a Medial Branch Block or Facet Injection) -> use 01938

IMPORTANT: Look at the document images carefully to identify:
- Procedure description
- Pre-operative diagnosis
- Post-operative diagnosis
- Any relevant medical information that can help determine the correct anesthesia CPT code

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

You must respond with a JSON object in this exact format:
{{
  "code": "00840",
  "explanation": "Brief explanation of why this code was chosen (1-2 sentences)"
}}

The explanation should briefly describe why this specific CPT code is appropriate for this procedure. Keep it concise (1-2 sentences maximum).

Respond with ONLY the JSON object, nothing else."""

    # Try loading CPT prompt from database (overrides hardcoded if available)
    db_cpt_prompt = _get_cpt_prompt(cpt_codes_text, include_code_list)
    if db_cpt_prompt:
        prompt = db_cpt_prompt
    
    # Append custom instructions if provided
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{custom_instructions.strip()}"

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
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/medical-data-processor",
        "X-Title": "Medical Data Processor"
    }
    
    # Ensure DeepSeek model uses correct OpenRouter format
    openrouter_model = model
    if "deepseek" in model.lower():
        # Map common DeepSeek model names to OpenRouter format
        model_lower = model.lower()
        if "v3.2" in model_lower or "v3" in model_lower:
            # Try deepseek-chat or deepseek-reasoner for v3 models
            openrouter_model = "deepseek/deepseek-chat"  # Most common DeepSeek model on OpenRouter
        elif "reasoner" in model_lower:
            openrouter_model = "deepseek/deepseek-reasoner"
        elif "chat" in model_lower:
            openrouter_model = "deepseek/deepseek-chat"
        elif model.startswith("deepseek/"):
            # Already in correct format, use as-is
            openrouter_model = model
        else:
            # Default to deepseek-chat (most commonly available)
            openrouter_model = "deepseek/deepseek-chat"
        logger.info(f"DeepSeek model - Original: '{model}', Using: '{openrouter_model}'")
    
    payload = {
        "model": openrouter_model,
        "messages": messages
    }

    # Gemini 3 models via OpenRouter: no reasoning, flex tier for cost savings
    if "gemini-3" in openrouter_model:
        payload["provider"] = {"sort": "throughput"}
        # NOTE: serviceTier=flex is set on Gemini 3 to halve the price
        payload["service_tier"] = "flex"

    # Enable web search if requested (for CPT code validation)
    if web_search:
        payload["plugins"] = [{"id": "web"}]
        logger.info(f"Enabled web search plugin for OpenRouter CPT prediction")

    # Log the model being used for debugging
    logger.info(f"OpenRouter API call - URL: {url}, Model: {openrouter_model}, Original model: {model}")

    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            response_data = response.json()

            # Extract the predicted code and explanation from the response
            predicted_code = None
            explanation = ""
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content_text = response_data["choices"][0]["message"]["content"]
                if content_text:
                    content_text = content_text.strip()
                    
                    # Try to parse JSON response
                    try:
                        # Try to extract JSON if wrapped in markdown code blocks
                        if "```json" in content_text:
                            content_text = content_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in content_text:
                            content_text = content_text.split("```")[1].split("```")[0].strip()
                        
                        result = json.loads(content_text)
                        predicted_code = result.get("code", "").strip()
                        explanation = result.get("explanation", "").strip()
                    except (json.JSONDecodeError, AttributeError):
                        # Fallback: try to extract code if JSON parsing fails
                        # Look for code pattern (5 digits starting with 0)
                        code_match = re.search(r'\b0\d{4}\b', content_text)
                        if code_match:
                            predicted_code = code_match.group(0)
                            explanation = content_text.replace(predicted_code, "").strip()
                        else:
                            # Last resort: use entire response as code
                            predicted_code = content_text
                            explanation = ""
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
            
            return predicted_code, explanation, tokens, cost, None
            
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
                if status_code == 404:
                    error_message = f"HTTP 404: Model '{openrouter_model}' not found. {error_detail}. Please check if the model name is correct or if it's available on OpenRouter."
                else:
                    error_message = f"HTTP {status_code}: {error_detail}"
            else:
                error_message = f"HTTP Error: {error_detail}"
            
            # Retry on all errors except 404 (model not found)
            if attempt < max_retries - 1:
                # Don't retry 404 errors - model name is wrong
                if status_code == 404:
                    logger.error(f"Model '{openrouter_model}' not found (404). Stopping retries. Error: {error_detail}")
                    break
                # Use exponential backoff for rate limits, shorter delay for other errors
                elif status_code == 429 or "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit error (429) on image prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                else:
                    wait_time = min(2 ** attempt, 4)  # Cap at 4 seconds for non-rate-limit errors
                    logger.warning(f"HTTP error on image prediction, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for image prediction: {error_message}")
                return None, "", 0, 0.0, error_message
                
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
                return None, "", 0, 0.0, error_message
                
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
                return None, "", 0, 0.0, error_message
    
    # Should never reach here, but just in case
    return None, "", 0, 0.0, "Max retries reached"


def predict_icd_codes_from_images_gemini(image_data_list, model="gemini-3-flash-preview", api_key=None, custom_instructions=None, predicted_cpt=None):
    """
    Predict ICD codes using Google GenAI SDK from PDF page images

    Args:
        image_data_list: List of base64 encoded image strings
        model: Gemini model name (e.g., "gemini-3-flash-preview", "gemini-2.5-pro")
        api_key: Google API key
        custom_instructions: Optional custom instructions to append to the prompt
        predicted_cpt: Optional predicted CPT code for this PDF (from prior CPT step). When provided,
            is injected into the ICD prompt as guidance so the model can apply CPT-specific ICD rules.

    Returns:
        tuple: (icd_codes_dict, tokens_used, cost_estimate, error_message)
        icd_codes_dict: Dictionary with keys 'ICD1', 'ICD1_Reasoning', 'ICD2', 'ICD2_Reasoning', 'ICD3', 'ICD3_Reasoning', 'ICD4', 'ICD4_Reasoning' containing ICD codes and reasoning
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return None, 0, 0.0, "Google GenAI SDK not available. Install with: pip install google-genai"
    
    # Normalize model name
    model = normalize_gemini_model(model)
    
    # Get API key
    if api_key:
        api_key_value = api_key
    else:
        api_key_value = os.getenv("GOOGLE_API_KEY")
    
    if not api_key_value:
        return None, 0, 0.0, "No Google API key provided"
    
    # Initialize client
    try:
        client = genai.Client(api_key=api_key_value)
    except Exception as e:
        return None, 0, 0.0, f"Failed to initialize Google GenAI client: {str(e)}"
    
    # Prepare the prompt (same as OpenRouter version)
    prompt = """You are a medical coding specialist.

Your task is to analyze the provided medical document page(s) and identify ICD diagnosis codes that are relevant to the procedure being performed.

IMPORTANT INSTRUCTIONS:
1. Analyze the entire PDF document carefully to understand the procedure and patient condition
2. Identify the PRIMARY diagnosis (ICD1) - this should be the main reason for the procedure
3. CRITICAL: Never code both the pre operative and post operative diagnoses, instead code only post operative diagnosis + supportive diagnoses (often there are supportive diagnoses of patient listed later in the pdf, these are diagnoses that support the necessity for the procedure). There might be many additional diagnoses listed, but choose the ones that are most supportive of necessity for the procedure. Think like an anesthesia medical coder and follow the rules an anesthesia medical coder would.
4. CRITICAL COLONOSCOPY CODING RULES:
   - General screening colonoscopy with NO findings: Code Z12.11 in ICD1. If patient has family history of GI malignancy, add Z80.0 in ICD2. If patient has personal history of colonic polyps, add Z86.0100 in ICD2 or ICD3. Leave remaining slots empty.
   - Screening colonoscopy with polypectomy/polyp found: Code Z12.11 in ICD1, K63.5 in ICD2 (or D12.x site-specific benign neoplasm if pathology specifies location: D12.0 cecum, D12.2 ascending, D12.3 transverse, D12.5 sigmoid, D12.6 descending). Add K57.30 (diverticulosis) in ICD3 if documented. Add K64.0/K64.9 (hemorrhoids) in ICD4 if documented.
   - Non-screening colonoscopy (diagnostic, with symptoms/findings): Use the actual finding or indication as ICD1 (e.g., K63.5 for polyp, K21.9 for GERD, K57.30 for diverticulosis, K44.9 for hernia). Put Z12.11 in ICD2 only if screening was also part of the reason. Add history codes (Z80.0, Z86.0100) and supportive findings (K57.30, K64.0) in remaining slots.
   - ORDERING PRIORITY for colonoscopy ICD2-4: History codes (Z80.0, Z86.0100) > additional findings (K57.30, K63.5, K64.0) > comorbidities (I10, E11.9)
5. CRITICAL VAGINAL DELIVERY / C-SECTION CODING RULES:
   - If the procedure is a vaginal delivery (CPT 01967) or C-section (CPT 01961) with NO documented complications: Use O80 in ICD1.
   - EXCEPTION: If the record documents complications such as tobacco/substance use in pregnancy (O99.214, O99.314), preeclampsia (O14.x), fetal distress (O77.x), gestational diabetes (O24.x), or other obstetric complications, use the complication code as ICD1 instead of O80.
   - For C-sections: Check for prior uterine scar codes (O34.21x). Use O34.219 (unspecified laterality) unless the document explicitly states which side the prior scar is on.
   - Do NOT use Z37.0 (single live birth outcome) as ICD1 — it is an outcome code, not a diagnosis.
6. CRITICAL CATARACT EXTRACTION CODING RULE:
   - If the procedure is cataract extraction surgery: Use the appropriate cataract ICD-10 diagnosis code as ICD1 (e.g., H25.811 for right eye, H25.812 for left eye). Do NOT use CPT codes (like 00142) as ICD codes.
   - EXCEPTION: If the procedure includes BOTH cataract extraction AND a glaucoma procedure (e.g., iStent, goniotomy, trabectome, MIGS), use the glaucoma diagnosis (H40.x) as ICD1, NOT the cataract code.
7. CRITICAL COLONOSCOPY/EGD DISTINCTION:
   - Z12.11 (Encounter for screening colonoscopy) must ONLY be used for colonoscopy procedures. NEVER use Z12.11 for EGD/upper endoscopy/esophagogastroduodenoscopy.
   - ⚠️ EGD (00731/00813) ICD1 RULE: NEVER use K31.7 (polyp of stomach/duodenum) as ICD1 unless the post-op findings explicitly document a stomach or duodenal polyp. K31.7 is wrong in 90%+ of cases. Instead use the actual indication/finding from the post-op diagnosis: K29.70 (gastritis), K44.9 (hiatal hernia), K21.9 (GERD), K22.2 (esophageal obstruction), R13.10 (dysphagia), K92.0 (hematemesis), K25.9 (gastric ulcer), K22.89 (other esophageal disease). For combined EGD + colonoscopy (00813), the upper GI indication goes in ICD1 (not Z12.11 screening and not K63.5 colon polyp — those are colonoscopy codes, not EGD codes).
   - For colonoscopy with polypectomy where a polyp is found: If the post-op diagnosis mentions polyp of colon, use K63.5 (polyp of colon) as ICD1. Do NOT confuse K62.5 (hemorrhage of anus/rectum) with K63.5 (polyp of colon).
   - For EGD supportive codes in ICD2-4: Add K29.70 (gastritis), K44.9 (hiatal hernia), K20.90 (esophagitis), K21.9 (GERD) ONLY if documented. Do NOT add K31.7 as a default.
8. CRITICAL PROSTATE BIOPSY CODING RULE:
   - If the procedure is a prostate biopsy and the indication is elevated PSA or abnormal PSA: Use R97.20 (Elevated prostate specific antigen [PSA]) as ICD1. Do NOT default to N20.0 (calculus of kidney) or N40.1 (BPH) unless those are the actual documented reason for the biopsy.
9. CRITICAL ENCOUNTER TYPE RULE (7th character):
   - For injury/fracture codes (S-codes, T-codes): Use "A" (initial encounter) unless the document explicitly states this is a follow-up or subsequent visit for a previously treated injury. Anesthesia for surgery is almost always an initial encounter (A), NOT subsequent (D) or sequela (S).
   - Example: S42.391A (initial) NOT S42.391D (subsequent).
10. CRITICAL LATERALITY AND SPECIFICITY RULE:
   - Carefully read the surgical site (left, right, bilateral) from the operative note and match the ICD code laterality accordingly.
   - When the document does not specify laterality, use the "unspecified" variant of the code (typically ending in 9). Do NOT guess laterality.
   - Always prefer the most specific ICD code supported by the documentation. For example, if the document says "open-angle glaucoma of left eye," use H40.1122 (specific) rather than H40.9 (unspecified).
   - ⚠️ POST-OP DIAGNOSIS ALWAYS TAKES PRIORITY: Always read and use the Post-Operative Diagnosis for ICD1, NOT the Pre-Op Diagnosis. The post-op is what actually happened during the procedure and is the definitive diagnosis for coding. If the post-op diagnosis is different from or more specific than the pre-op (e.g., pre-op says "gallstones" K80.00 but post-op says "acute cholecystitis" K81.0, or pre-op says "osteomyelitis" but post-op says "diabetic complication"), use the post-op. Only fall back to pre-op if no post-op diagnosis is documented.
11. CRITICAL ICD2/ICD3/ICD4 SECONDARY DIAGNOSIS RULES:
   - Only include secondary diagnoses that are directly relevant to the anesthesia encounter. Do NOT fill ICD2-4 slots just because space is available.
   - DO NOT USE these as routine filler codes: E66.9/E66.01 (obesity), Z68.x (BMI codes), R-codes (symptoms like R19.7, R10.13, R14.0, R11.2) — unless they are the primary reason for or directly complicate the procedure. NEVER use Z68.x BMI codes for anesthesia billing.
   - COMORBIDITY RULE BASED ON ASA STATUS: If the patient's ASA status is 1 or 2, do NOT add comorbidities from the medical history — only use ICD codes directly related to the procedure/diagnosis. If ASA is 3 or higher, you MAY add relevant comorbidities (I10 hypertension, E11.9 diabetes, G47.33 sleep apnea, I25.10 CAD, E78.5 hyperlipidemia, J44.9 COPD) in ICD2-4 since these contribute to anesthesia risk. ASA 1-2 = procedure diagnosis only. ASA 3+ = include significant comorbidities.
   - When BOTH depression (F32.9) AND anxiety (F41.9) are documented, combine them into F41.8 (mixed anxiety and depressive disorder). Do not use F32.9 and F41.9 separately.
   - For eye/ophthalmic cases: Use I10 (hypertension) as ICD2 if documented, NOT H40.9 (unspecified glaucoma) unless glaucoma is actually relevant to the procedure. Do not add multiple eye-specific codes (H52.x refractive errors, H43.x vitreous disorders, H35.x retinal disorders) as secondary diagnoses unless they are the reason for the procedure.
   - For GI endoscopy cases: Add K57.30 (diverticulosis), K64.0/K64.9 (hemorrhoids), K29.70 (gastritis) as supportive ICD3/4 ONLY if documented in the procedure findings.
   - PREFERRED comorbidity ordering when multiple are documented (ASA 3+ only): I10 (hypertension) > E78.5 (hyperlipidemia) > J44.9 (COPD) > G47.33 (sleep apnea) > E11.9 (diabetes) > E03.9 (hypothyroidism). Do NOT use E66.9/E66.01 (obesity), F41.9 (anxiety), K21.9 (GERD), or F17.210 (nicotine) as filler comorbidities.
   - If the document lists many comorbidities, pick at most 2-3 that are most relevant to anesthesia risk (cardiovascular, metabolic, respiratory). Leave slots empty rather than adding marginally relevant codes.
   - COLONOSCOPY/EGD ICD ORDERING: For colonoscopy and EGD cases, the primary finding or indication goes in ICD1 (e.g., K63.5 polyp, K44.9 hernia, R10.11 pain, K21.9 GERD). Put Z12.11 (screening) in ICD2, NOT ICD1. The procedure finding/indication always takes priority over the screening code.
12. CRITICAL POLYP SPECIFICITY RULE:
   - When the endoscopy report specifies the anatomic location of a polyp, use site-specific D12.x benign neoplasm codes instead of generic K63.5:
     D12.0 (cecum), D12.2 (ascending colon), D12.3 (transverse colon), D12.4 (descending colon), D12.5 (sigmoid colon), D12.6 (colon, unspecified), D12.8 (rectum).
   - Use K63.5 (polyp of colon) only when the specific location is not documented or when coding the polyp as a secondary finding.
13. LATERALITY AND SPECIFICITY RULE: When the procedure description specifies a side (Right, Left, Bilateral), you MUST use the laterality-specific ICD code, NOT the bilateral or unspecified variant.
   - "Right Knee Arthroplasty" → M17.11 (right), NOT M17.0 (bilateral)
   - "Left Knee Arthroplasty" → M17.12 (left), NOT M17.0 (bilateral)
   - "Right Carpal Tunnel Release" → G56.01 (right), NOT G56.03 (bilateral)
   - "Left eye" procedure → use the left-eye specific code, not unspecified
   - For encounter type: Use 7th character "A" (initial encounter) unless the record explicitly says follow-up/subsequent, in which case use "D" (subsequent) or "S" (sequela).
   - General rule: Always match the ICD laterality to the procedure laterality. If the procedure says one side, never code bilateral. If the procedure says bilateral, use the bilateral code.
14. Identify up to 3 additional ICD codes (ICD2, ICD3, ICD4) sorted by relevance to the procedure
14. Only include ICD codes that are directly relevant to the procedure or patient condition
15. If fewer than 4 relevant ICD codes exist, leave the remaining fields empty
16. Use standard ICD-10 format (e.g., "E11.9", "I10", "Z87.891")
17. CRITICAL: Use web search to verify that all ICD codes you provide are valid and current as of November 2025. Only use the most recent ICD codes that are valid in November 2025. Do not use outdated or invalid codes.
18. CRITICAL: If there is outdated ICD codes listed on record try to find on google valid icd codes updated as of december 2025, so basically take the diagnosis code and update it accordingly with google
19. CRITICAL: Always make sure to not just pick the main diagnosis but to also look at secondary diagnoses further in the record IF available, they will often not be listed clearly as codes but instead as small snippets of text, there might be many of them listed like obesity and diabetes and such... make sure to convert those small texts to diagnosis codes, but also make sure to pick the ones that are MOST related to the main procedure and diagnosis itself, also make sure to use UPDATED december 2025 codes with google search
20. CRITICAL: Check for Excludes1 Conflicts: Before finalizing the JSON, verify if the selected codes have "Excludes1" notes in the ICD-10 manual that prevent them from being billed together.
21. Conflict Resolution: If two codes conflict (e.g., J35.1 and J35.01), prioritize the Post-Operative or more specific diagnosis.
22. AFIB CODING: If the record mentions "AFIB" or "atrial fibrillation" without specifying the type (paroxysmal, persistent, chronic, permanent), use I48.91 (unspecified atrial fibrillation). Do NOT default to I48.0 (paroxysmal) unless "paroxysmal" is explicitly documented.

OUTPUT FORMAT:
You must respond with ONLY a JSON object in this exact format:
{
  "ICD1": "primary_diagnosis_code",
  "ICD1_Reasoning": "brief explanation of why this code was chosen (1-2 sentences)",
  "ICD2": "secondary_diagnosis_code_or_empty",
  "ICD2_Reasoning": "brief explanation of why this code was chosen or empty string if no code",
  "ICD3": "tertiary_diagnosis_code_or_empty",
  "ICD3_Reasoning": "brief explanation of why this code was chosen or empty string if no code",
  "ICD4": "quaternary_diagnosis_code_or_empty",
  "ICD4_Reasoning": "brief explanation of why this code was chosen or empty string if no code"
}

If a code doesn't exist, use an empty string "" for both the code and its reasoning field.

Example response:
{
  "ICD1": "K63.5",
  "ICD1_Reasoning": "Polyp of colon identified during colonoscopy procedure",
  "ICD2": "E11.9",
  "ICD2_Reasoning": "Type 2 diabetes mellitus documented in patient history",
  "ICD3": "I10",
  "ICD3_Reasoning": "Essential hypertension noted in medical record",
  "ICD4": "",
  "ICD4_Reasoning": ""
}

IMPORTANT: For each ICD code you provide, you MUST include a brief reasoning explanation (1-2 sentences) explaining why that specific code was chosen based on the document content. This helps verify the accuracy of your coding decisions.

Respond with ONLY the JSON object, nothing else."""
    
    # Try loading ICD prompt from database (overrides hardcoded if available)
    db_icd_prompt = _get_icd_prompt()
    if db_icd_prompt:
        prompt = db_icd_prompt

    # Append custom instructions if provided
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{custom_instructions.strip()}"

    # Append predicted CPT guidance if provided
    if predicted_cpt:
        prompt += _build_cpt_guidance_block(predicted_cpt)

    # Build content with images
    parts = [types.Part.from_text(text=prompt)]

    # Add images from base64 strings
    for img_data in image_data_list:
        # Decode base64 to bytes
        img_bytes = base64.b64decode(img_data)
        parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
    
    contents = [types.Content(role="user", parts=parts)]
    
    # Configure thinking for Gemini 3 / 3.1 models
    if model in ("gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-pro-preview"):
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)

    # Enable web search for Gemini models (required for ICD code validation per prompt)
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    
    use_flex = True  # Start with flex tier (50% cheaper)
    FLEX_TIMEOUT = 600  # 10 minutes

    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            service_tier = "flex" if use_flex else "standard"
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                thinking_config=thinking_config,
                tools=tools,
                http_options=types.HttpOptions(
                    extra_body={"serviceTier": service_tier},
                    timeout=FLEX_TIMEOUT * 1000 if use_flex else None,
                ),
            )

            # Call Gemini API
            full_response = ""
            request_start = time.time()
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:
                    full_response += chunk.text
                if use_flex and (time.time() - request_start) > FLEX_TIMEOUT:
                    raise TimeoutError("Flex request exceeded 10 minute timeout")

            response_text = full_response.strip()

            if not response_text:
                raise ValueError("Empty response from API")

            cleaned_response = response_text
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            icd_codes_dict = json.loads(cleaned_response)

            result = {
                "ICD1": icd_codes_dict.get("ICD1", ""),
                "ICD1_Reasoning": icd_codes_dict.get("ICD1_Reasoning", ""),
                "ICD2": icd_codes_dict.get("ICD2", ""),
                "ICD2_Reasoning": icd_codes_dict.get("ICD2_Reasoning", ""),
                "ICD3": icd_codes_dict.get("ICD3", ""),
                "ICD3_Reasoning": icd_codes_dict.get("ICD3_Reasoning", ""),
                "ICD4": icd_codes_dict.get("ICD4", ""),
                "ICD4_Reasoning": icd_codes_dict.get("ICD4_Reasoning", "")
            }
            correct_icd_codes(result)

            prompt_chars = len(prompt)
            image_tokens_estimate = len(image_data_list) * 1000
            response_chars = len(response_text)
            total_tokens_estimate = prompt_chars + image_tokens_estimate + response_chars

            if "gemini-flash-lite" in model:
                cost = total_tokens_estimate * 0.00005 / 1000
            elif "gemini-3-flash" in model:
                cost = total_tokens_estimate * 0.000125 / 1000
            elif "gemini-3-pro" in model or "gemini-3.1-pro" in model:
                cost = total_tokens_estimate * 0.00125 / 1000
            elif "gemini-2.5-flash" in model:
                cost = total_tokens_estimate * 0.000075 / 1000
            else:
                cost = total_tokens_estimate * 0.0005 / 1000

            if use_flex:
                cost *= 0.5

            return result, total_tokens_estimate, cost, None

        except json.JSONDecodeError as e:
            error_message = f"JSON parsing failed: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 4)
                logger.warning(f"JSON parsing error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached for JSON parsing: {error_message}")
                result = {
                    "ICD1": "", "ICD1_Reasoning": "",
                    "ICD2": "", "ICD2_Reasoning": "",
                    "ICD3": "", "ICD3_Reasoning": "",
                    "ICD4": "", "ICD4_Reasoning": ""
                }
                icd_pattern = r'[A-Z]\d{2}\.?\d*'
                found_codes = re.findall(icd_pattern, response_text)
                for i, code in enumerate(found_codes[:4]):
                    result[f"ICD{i+1}"] = code
                    result[f"ICD{i+1}_Reasoning"] = "Code extracted from response (reasoning not available)"
                return result, 0, 0.0, error_message

        except (TimeoutError, Exception) as e:
            if use_flex and not isinstance(e, json.JSONDecodeError):
                reason = "timeout" if isinstance(e, TimeoutError) else str(e)[:80]
                logger.warning(f"Flex failed on ICD prediction ({reason}), switching to standard tier")
                use_flex = False
                continue

            error_message = f"API error: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 4)
                logger.warning(f"API error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {error_message}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Max retries reached: {error_message}")
                return None, 0, 0.0, error_message

    return None, 0, 0.0, "Max retries reached"


def predict_icd_codes_from_images(image_data_list, model="openai/gpt-5.2:online", api_key=None, custom_instructions=None, predicted_cpt=None):
    """
    Predict ICD codes using OpenRouter API or Google GenAI SDK from PDF page images

    Args:
        image_data_list: List of base64 encoded image strings
        model: Model to use (default: openai/gpt-5.2:online). For Gemini, use format "gemini-3-flash-preview" or "gemini-2.5-pro"
        api_key: API key (OpenRouter API key for OpenAI models, Google API key for Gemini models)
        custom_instructions: Optional custom instructions to append to the prompt
        predicted_cpt: Optional predicted CPT code for this PDF (from prior CPT step). When provided,
            is injected into the ICD prompt as guidance so the model can apply CPT-specific ICD rules.

    Returns:
        tuple: (icd_codes_dict, tokens_used, cost_estimate, error_message)
        icd_codes_dict: Dictionary with keys 'ICD1', 'ICD1_Reasoning', 'ICD2', 'ICD2_Reasoning', 'ICD3', 'ICD3_Reasoning', 'ICD4', 'ICD4_Reasoning' containing ICD codes and reasoning
    """
    # Check if using Gemini model
    if is_gemini_model(model):
        # Use Google GenAI SDK directly if GOOGLE_API_KEY is available
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if google_api_key and GOOGLE_GENAI_AVAILABLE:
            logger.info(f"Using Google GenAI SDK directly for Gemini model '{model}'")
            return predict_icd_codes_from_images_gemini(image_data_list, model, api_key, custom_instructions, predicted_cpt=predicted_cpt)
        else:
            # Fall back to OpenRouter with google/ prefix
            logger.info(f"No GOOGLE_API_KEY found, routing Gemini model '{model}' through OpenRouter")
            model = f"google/{normalize_gemini_model(model)}"
            # Fall through to OpenRouter path below

    # Use OpenRouter for non-Gemini models (or Gemini models without GOOGLE_API_KEY)
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
3. CRITICAL: Never code both the pre operative and post operative diagnoses, instead code only post operative diagnosis + supportive diagnoses (often there are supportive diagnoses of patient listed later in the pdf, these are diagnoses that support the necessity for the procedure). There might be many additional diagnoses listed, but choose the ones that are most supportive of necessity for the procedure. Think like an anesthesia medical coder and follow the rules an anesthesia medical coder would.
4. CRITICAL COLONOSCOPY CODING RULES:
   - General screening colonoscopy with NO findings: Code Z12.11 in ICD1. If patient has family history of GI malignancy, add Z80.0 in ICD2. If patient has personal history of colonic polyps, add Z86.0100 in ICD2 or ICD3. Leave remaining slots empty.
   - Screening colonoscopy with polypectomy/polyp found: Code Z12.11 in ICD1, K63.5 in ICD2 (or D12.x site-specific benign neoplasm if pathology specifies location: D12.0 cecum, D12.2 ascending, D12.3 transverse, D12.5 sigmoid, D12.6 descending). Add K57.30 (diverticulosis) in ICD3 if documented. Add K64.0/K64.9 (hemorrhoids) in ICD4 if documented.
   - Non-screening colonoscopy (diagnostic, with symptoms/findings): Use the actual finding or indication as ICD1 (e.g., K63.5 for polyp, K21.9 for GERD, K57.30 for diverticulosis, K44.9 for hernia). Put Z12.11 in ICD2 only if screening was also part of the reason. Add history codes (Z80.0, Z86.0100) and supportive findings (K57.30, K64.0) in remaining slots.
   - ORDERING PRIORITY for colonoscopy ICD2-4: History codes (Z80.0, Z86.0100) > additional findings (K57.30, K63.5, K64.0) > comorbidities (I10, E11.9)
5. CRITICAL VAGINAL DELIVERY / C-SECTION CODING RULES:
   - If the procedure is a vaginal delivery (CPT 01967) or C-section (CPT 01961) with NO documented complications: Use O80 in ICD1.
   - EXCEPTION: If the record documents complications such as tobacco/substance use in pregnancy (O99.214, O99.314), preeclampsia (O14.x), fetal distress (O77.x), gestational diabetes (O24.x), or other obstetric complications, use the complication code as ICD1 instead of O80.
   - For C-sections: Check for prior uterine scar codes (O34.21x). Use O34.219 (unspecified laterality) unless the document explicitly states which side the prior scar is on.
   - Do NOT use Z37.0 (single live birth outcome) as ICD1 — it is an outcome code, not a diagnosis.
6. CRITICAL CATARACT EXTRACTION CODING RULE:
   - If the procedure is cataract extraction surgery: Use the appropriate cataract ICD-10 diagnosis code as ICD1 (e.g., H25.811 for right eye, H25.812 for left eye). Do NOT use CPT codes (like 00142) as ICD codes.
   - EXCEPTION: If the procedure includes BOTH cataract extraction AND a glaucoma procedure (e.g., iStent, goniotomy, trabectome, MIGS), use the glaucoma diagnosis (H40.x) as ICD1, NOT the cataract code.
7. CRITICAL COLONOSCOPY/EGD DISTINCTION:
   - Z12.11 (Encounter for screening colonoscopy) must ONLY be used for colonoscopy procedures. NEVER use Z12.11 for EGD/upper endoscopy/esophagogastroduodenoscopy.
   - ⚠️ EGD (00731/00813) ICD1 RULE: NEVER use K31.7 (polyp of stomach/duodenum) as ICD1 unless the post-op findings explicitly document a stomach or duodenal polyp. K31.7 is wrong in 90%+ of cases. Instead use the actual indication/finding from the post-op diagnosis: K29.70 (gastritis), K44.9 (hiatal hernia), K21.9 (GERD), K22.2 (esophageal obstruction), R13.10 (dysphagia), K92.0 (hematemesis), K25.9 (gastric ulcer), K22.89 (other esophageal disease). For combined EGD + colonoscopy (00813), the upper GI indication goes in ICD1 (not Z12.11 screening and not K63.5 colon polyp — those are colonoscopy codes, not EGD codes).
   - For colonoscopy with polypectomy where a polyp is found: If the post-op diagnosis mentions polyp of colon, use K63.5 (polyp of colon) as ICD1. Do NOT confuse K62.5 (hemorrhage of anus/rectum) with K63.5 (polyp of colon).
   - For EGD supportive codes in ICD2-4: Add K29.70 (gastritis), K44.9 (hiatal hernia), K20.90 (esophagitis), K21.9 (GERD) ONLY if documented. Do NOT add K31.7 as a default.
8. CRITICAL PROSTATE BIOPSY CODING RULE:
   - If the procedure is a prostate biopsy and the indication is elevated PSA or abnormal PSA: Use R97.20 (Elevated prostate specific antigen [PSA]) as ICD1. Do NOT default to N20.0 (calculus of kidney) or N40.1 (BPH) unless those are the actual documented reason for the biopsy.
9. CRITICAL ENCOUNTER TYPE RULE (7th character):
   - For injury/fracture codes (S-codes, T-codes): Use "A" (initial encounter) unless the document explicitly states this is a follow-up or subsequent visit for a previously treated injury. Anesthesia for surgery is almost always an initial encounter (A), NOT subsequent (D) or sequela (S).
   - Example: S42.391A (initial) NOT S42.391D (subsequent).
10. CRITICAL LATERALITY AND SPECIFICITY RULE:
   - Carefully read the surgical site (left, right, bilateral) from the operative note and match the ICD code laterality accordingly.
   - When the document does not specify laterality, use the "unspecified" variant of the code (typically ending in 9). Do NOT guess laterality.
   - Always prefer the most specific ICD code supported by the documentation. For example, if the document says "open-angle glaucoma of left eye," use H40.1122 (specific) rather than H40.9 (unspecified).
   - ⚠️ POST-OP DIAGNOSIS ALWAYS TAKES PRIORITY: Always read and use the Post-Operative Diagnosis for ICD1, NOT the Pre-Op Diagnosis. The post-op is what actually happened during the procedure and is the definitive diagnosis for coding. If the post-op diagnosis is different from or more specific than the pre-op (e.g., pre-op says "gallstones" K80.00 but post-op says "acute cholecystitis" K81.0, or pre-op says "osteomyelitis" but post-op says "diabetic complication"), use the post-op. Only fall back to pre-op if no post-op diagnosis is documented.
11. CRITICAL ICD2/ICD3/ICD4 SECONDARY DIAGNOSIS RULES:
   - Only include secondary diagnoses that are directly relevant to the anesthesia encounter. Do NOT fill ICD2-4 slots just because space is available.
   - DO NOT USE these as routine filler codes: E66.9/E66.01 (obesity), Z68.x (BMI codes), R-codes (symptoms like R19.7, R10.13, R14.0, R11.2) — unless they are the primary reason for or directly complicate the procedure. NEVER use Z68.x BMI codes for anesthesia billing.
   - COMORBIDITY RULE BASED ON ASA STATUS: If the patient's ASA status is 1 or 2, do NOT add comorbidities from the medical history — only use ICD codes directly related to the procedure/diagnosis. If ASA is 3 or higher, you MAY add relevant comorbidities (I10 hypertension, E11.9 diabetes, G47.33 sleep apnea, I25.10 CAD, E78.5 hyperlipidemia, J44.9 COPD) in ICD2-4 since these contribute to anesthesia risk. ASA 1-2 = procedure diagnosis only. ASA 3+ = include significant comorbidities.
   - When BOTH depression (F32.9) AND anxiety (F41.9) are documented, combine them into F41.8 (mixed anxiety and depressive disorder). Do not use F32.9 and F41.9 separately.
   - For eye/ophthalmic cases: Use I10 (hypertension) as ICD2 if documented, NOT H40.9 (unspecified glaucoma) unless glaucoma is actually relevant to the procedure. Do not add multiple eye-specific codes (H52.x refractive errors, H43.x vitreous disorders, H35.x retinal disorders) as secondary diagnoses unless they are the reason for the procedure.
   - For GI endoscopy cases: Add K57.30 (diverticulosis), K64.0/K64.9 (hemorrhoids), K29.70 (gastritis) as supportive ICD3/4 ONLY if documented in the procedure findings.
   - PREFERRED comorbidity ordering when multiple are documented (ASA 3+ only): I10 (hypertension) > E78.5 (hyperlipidemia) > J44.9 (COPD) > G47.33 (sleep apnea) > E11.9 (diabetes) > E03.9 (hypothyroidism). Do NOT use E66.9/E66.01 (obesity), F41.9 (anxiety), K21.9 (GERD), or F17.210 (nicotine) as filler comorbidities.
   - If the document lists many comorbidities, pick at most 2-3 that are most relevant to anesthesia risk (cardiovascular, metabolic, respiratory). Leave slots empty rather than adding marginally relevant codes.
   - COLONOSCOPY/EGD ICD ORDERING: For colonoscopy and EGD cases, the primary finding or indication goes in ICD1 (e.g., K63.5 polyp, K44.9 hernia, R10.11 pain, K21.9 GERD). Put Z12.11 (screening) in ICD2, NOT ICD1. The procedure finding/indication always takes priority over the screening code.
12. CRITICAL POLYP SPECIFICITY RULE:
   - When the endoscopy report specifies the anatomic location of a polyp, use site-specific D12.x benign neoplasm codes instead of generic K63.5:
     D12.0 (cecum), D12.2 (ascending colon), D12.3 (transverse colon), D12.4 (descending colon), D12.5 (sigmoid colon), D12.6 (colon, unspecified), D12.8 (rectum).
   - Use K63.5 (polyp of colon) only when the specific location is not documented or when coding the polyp as a secondary finding.
13. LATERALITY AND SPECIFICITY RULE: When the procedure description specifies a side (Right, Left, Bilateral), you MUST use the laterality-specific ICD code, NOT the bilateral or unspecified variant.
   - "Right Knee Arthroplasty" → M17.11 (right), NOT M17.0 (bilateral)
   - "Left Knee Arthroplasty" → M17.12 (left), NOT M17.0 (bilateral)
   - "Right Carpal Tunnel Release" → G56.01 (right), NOT G56.03 (bilateral)
   - "Left eye" procedure → use the left-eye specific code, not unspecified
   - For encounter type: Use 7th character "A" (initial encounter) unless the record explicitly says follow-up/subsequent, in which case use "D" (subsequent) or "S" (sequela).
   - General rule: Always match the ICD laterality to the procedure laterality. If the procedure says one side, never code bilateral. If the procedure says bilateral, use the bilateral code.
14. Identify up to 3 additional ICD codes (ICD2, ICD3, ICD4) sorted by relevance to the procedure
14. Only include ICD codes that are directly relevant to the procedure or patient condition
15. If fewer than 4 relevant ICD codes exist, leave the remaining fields empty
16. Use standard ICD-10 format (e.g., "E11.9", "I10", "Z87.891")
17. CRITICAL: Use web search to verify that all ICD codes you provide are valid and current as of November 2025. Only use the most recent ICD codes that are valid in November 2025. Do not use outdated or invalid codes.
18. CRITICAL: If there is outdated ICD codes listed on record try to find on google valid icd codes updated as of december 2025, so basically take the diagnosis code and update it accordingly with google
19. CRITICAL: Always make sure to not just pick the main diagnosis but to also look at secondary diagnoses further in the record IF available, they will often not be listed clearly as codes but instead as small snippets of text, there might be many of them listed like obesity and diabetes and such... make sure to convert those small texts to diagnosis codes, but also make sure to pick the ones that are MOST related to the main procedure and diagnosis itself, also make sure to use UPDATED december 2025 codes with google search
20. CRITICAL: Check for Excludes1 Conflicts: Before finalizing the JSON, verify if the selected codes have "Excludes1" notes in the ICD-10 manual that prevent them from being billed together.
21. Conflict Resolution: If two codes conflict (e.g., J35.1 and J35.01), prioritize the Post-Operative or more specific diagnosis.
22. AFIB CODING: If the record mentions "AFIB" or "atrial fibrillation" without specifying the type (paroxysmal, persistent, chronic, permanent), use I48.91 (unspecified atrial fibrillation). Do NOT default to I48.0 (paroxysmal) unless "paroxysmal" is explicitly documented.

OUTPUT FORMAT:
You must respond with ONLY a JSON object in this exact format:
{
  "ICD1": "primary_diagnosis_code",
  "ICD1_Reasoning": "brief explanation of why this code was chosen (1-2 sentences)",
  "ICD2": "secondary_diagnosis_code_or_empty",
  "ICD2_Reasoning": "brief explanation of why this code was chosen or empty string if no code",
  "ICD3": "tertiary_diagnosis_code_or_empty",
  "ICD3_Reasoning": "brief explanation of why this code was chosen or empty string if no code",
  "ICD4": "quaternary_diagnosis_code_or_empty",
  "ICD4_Reasoning": "brief explanation of why this code was chosen or empty string if no code"
}

If a code doesn't exist, use an empty string "" for both the code and its reasoning field.

Example response:
{
  "ICD1": "K63.5",
  "ICD1_Reasoning": "Polyp of colon identified during colonoscopy procedure",
  "ICD2": "E11.9",
  "ICD2_Reasoning": "Type 2 diabetes mellitus documented in patient history",
  "ICD3": "I10",
  "ICD3_Reasoning": "Essential hypertension noted in medical record",
  "ICD4": "",
  "ICD4_Reasoning": ""
}

IMPORTANT: For each ICD code you provide, you MUST include a brief reasoning explanation (1-2 sentences) explaining why that specific code was chosen based on the document content. This helps verify the accuracy of your coding decisions.

Respond with ONLY the JSON object, nothing else."""
    
        # Try loading ICD prompt from database (overrides hardcoded if available)
    db_icd_prompt = _get_icd_prompt()
    if db_icd_prompt:
        prompt = db_icd_prompt

    # Append custom instructions if provided
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{custom_instructions.strip()}"

    # Append predicted CPT guidance if provided
    if predicted_cpt:
        prompt += _build_cpt_guidance_block(predicted_cpt)

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
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/medical-data-processor",
        "X-Title": "Medical Data Processor"
    }
    
    # Ensure DeepSeek model uses correct OpenRouter format
    openrouter_model = model
    if "deepseek" in model.lower():
        # Map common DeepSeek model names to OpenRouter format
        model_lower = model.lower()
        if "v3.2" in model_lower or "v3" in model_lower:
            # Try deepseek-chat or deepseek-reasoner for v3 models
            openrouter_model = "deepseek/deepseek-chat"  # Most common DeepSeek model on OpenRouter
        elif "reasoner" in model_lower:
            openrouter_model = "deepseek/deepseek-reasoner"
        elif "chat" in model_lower:
            openrouter_model = "deepseek/deepseek-chat"
        elif model.startswith("deepseek/"):
            # Already in correct format, use as-is
            openrouter_model = model
        else:
            # Default to deepseek-chat (most commonly available)
            openrouter_model = "deepseek/deepseek-chat"
        logger.info(f"DeepSeek model - Original: '{model}', Using: '{openrouter_model}'")
    
    payload = {
        "model": openrouter_model,
        "messages": messages
    }

    # Enable reasoning for Gemini 3 models via OpenRouter
    if "gemini-3" in openrouter_model:
        payload["reasoning"] = {"effort": "high"}

    # Enable web search for ICD code validation
    payload["plugins"] = [{"id": "web"}]
    logger.info(f"Enabled web search plugin for OpenRouter ICD prediction")

    # Log the model being used for debugging
    logger.info(f"OpenRouter API call - URL: {url}, Model: {openrouter_model}, Original model: {model}")

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
                            "ICD1_Reasoning": icd_codes_dict.get("ICD1_Reasoning", ""),
                            "ICD2": icd_codes_dict.get("ICD2", ""),
                            "ICD2_Reasoning": icd_codes_dict.get("ICD2_Reasoning", ""),
                            "ICD3": icd_codes_dict.get("ICD3", ""),
                            "ICD3_Reasoning": icd_codes_dict.get("ICD3_Reasoning", ""),
                            "ICD4": icd_codes_dict.get("ICD4", ""),
                            "ICD4_Reasoning": icd_codes_dict.get("ICD4_Reasoning", "")
                        }
                        correct_icd_codes(result)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract codes manually
                        logger.warning(f"Failed to parse JSON, attempting manual extraction. Response: {content_text[:200]}")
                        result = {
                            "ICD1": "",
                            "ICD1_Reasoning": "",
                            "ICD2": "",
                            "ICD2_Reasoning": "",
                            "ICD3": "",
                            "ICD3_Reasoning": "",
                            "ICD4": "",
                            "ICD4_Reasoning": ""
                        }
                        # Try to find ICD codes in the text (basic pattern matching)
                        icd_pattern = r'[A-Z]\d{2}\.?\d*'
                        found_codes = re.findall(icd_pattern, content_text)
                        for i, code in enumerate(found_codes[:4]):
                            result[f"ICD{i+1}"] = code
                            result[f"ICD{i+1}_Reasoning"] = "Code extracted from response (reasoning not available)"
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
                if "openai/gpt-5" in model or "gpt-5" in model or "gpt5:online" in model or "gpt-5.2" in model:
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
                    logger.error(f"OpenRouter error response: {response_data}")
                    if isinstance(response_data, dict):
                        error_detail = response_data.get('error', {}).get('message', str(e))
                        if not error_detail or error_detail == str(e):
                            error_detail = response_data.get('error', str(e))
            except Exception as parse_error:
                logger.error(f"Failed to parse error response: {parse_error}")
                if hasattr(e, 'response') and e.response:
                    logger.error(f"Raw error response text: {e.response.text}")
            
            # Build descriptive error message
            if status_code:
                error_message = f"HTTP {status_code}: {error_detail}"
            else:
                error_message = f"HTTP Error: {error_detail}"
            
            # Log the model that was used for debugging
            logger.error(f"Failed OpenRouter request - Model: {openrouter_model}, URL: {url}")
            
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


def _get_cpt_prompt(cpt_codes_text, include_code_list):
    """Get the CPT base prompt, trying database first, falling back to hardcoded."""
    db_prompt = load_base_prompt('base_cpt_prompt')
    if db_prompt:
        # DB prompt may contain {cpt_codes_text} placeholder or may be the raw text
        if include_code_list and cpt_codes_text:
            code_section = f"\n\nHere is the reference list of valid anesthesia CPT codes:\n\n{cpt_codes_text}\n"
        else:
            code_section = ""
        return f"You are a medical anesthesia CPT coder.\n\nYour task is to predict the most relevant anesthesia CPT code for anesthesia billing for a certain procedure by analyzing the provided medical document page(s).\n{code_section}\n{db_prompt}\n\nYou must respond with a JSON object in this exact format:\n{{\n  \"code\": \"00840\",\n  \"explanation\": \"Brief explanation of why this code was chosen (1-2 sentences)\"\n}}\n\nThe explanation should briefly describe why this specific CPT code is appropriate for this procedure. Keep it concise (1-2 sentences maximum).\n\nRespond with ONLY the JSON object, nothing else."
    return None  # Caller uses hardcoded fallback


def _build_cpt_guidance_block(predicted_cpt):
    """Build a high-priority CPT-guidance block to append to the ICD prompt.

    Used when the ICD prediction step runs *after* CPT prediction and receives
    the predicted CPT code as input. Placed at the very end of the prompt so it
    is the last instruction the model sees before producing output.
    """
    code = str(predicted_cpt).strip() if predicted_cpt is not None else ""
    if not code:
        return ""
    return (
        "\n\n"
        "================================================================================\n"
        "PREDICTED CPT CODE FOR THIS CASE (from prior CPT prediction step)\n"
        "================================================================================\n"
        f"CPT = {code}\n"
        "\n"
        "A separate CPT prediction step has already analyzed this document and predicted\n"
        "the CPT above. Treat this as strong guidance — the CPT code tightly constrains\n"
        "which ICD1 is appropriate. Apply any facility-specific CPT->ICD rules in the\n"
        "prompt above using this CPT. If no facility-specific rule applies for this CPT,\n"
        "ensure your ICD1 is clinically consistent with the procedure this CPT represents\n"
        "(e.g., do NOT return a GI ICD when CPT is for an ortho procedure).\n"
    )


def _get_icd_prompt():
    """Get the ICD base prompt, trying database first, falling back to hardcoded."""
    db_prompt = load_base_prompt('base_icd_prompt')
    if db_prompt:
        return f"You are a medical coding specialist.\n\nYour task is to analyze the provided medical document page(s) and identify ICD diagnosis codes that are relevant to the procedure being performed.\n\n{db_prompt}\n\nOUTPUT FORMAT:\nYou must respond with ONLY a JSON object in this exact format:\n{{\n  \"ICD1\": \"primary_diagnosis_code\",\n  \"ICD1_Reasoning\": \"brief explanation of why this code was chosen (1-2 sentences)\",\n  \"ICD2\": \"secondary_diagnosis_code_or_empty\",\n  \"ICD2_Reasoning\": \"brief explanation of why this code was chosen or empty string if no code\",\n  \"ICD3\": \"tertiary_diagnosis_code_or_empty\",\n  \"ICD3_Reasoning\": \"brief explanation of why this code was chosen or empty string if no code\",\n  \"ICD4\": \"quaternary_diagnosis_code_or_empty\",\n  \"ICD4_Reasoning\": \"brief explanation of why this code was chosen or empty string if no code\"\n}}\n\nIf a code doesn't exist, use an empty string \"\" for both the code and its reasoning field.\n\nIMPORTANT: For each ICD code you provide, you MUST include a brief reasoning explanation (1-2 sentences) explaining why that specific code was chosen based on the document content.\n\nRespond with ONLY the JSON object, nothing else."
    return None  # Caller uses hardcoded fallback


def load_base_prompt(name):
    """Load a base prompt from the database. Returns None if not found."""
    try:
        import sys
        backend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        from db_utils import get_base_prompt
        result = get_base_prompt(name)
        if result and result.get('content'):
            logger.info(f"✅ Loaded base prompt '{name}' from database ({len(result['content'])} chars)")
            return result['content']
        return None
    except Exception as e:
        logger.warning(f"⚠️  Could not load base prompt '{name}' from database: {e}")
        return None


def load_cpt_codes():
    """Load CPT codes from database, falling back to cpt_codes.txt"""
    # Try database first
    db_codes = load_base_prompt('cpt_codes_list')
    if db_codes:
        return db_codes

    # Fallback to file
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
        explanations = [""] * len(df)
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
                    return idx, "ERROR: Empty procedure", "", 0, 0.0, "Procedure Description is empty or missing"
                
                predicted_code, explanation, tokens, cost, error = predict_asa_code_general(
                    procedure, preop, postop, cpt_codes_text, model, api_key
                )
                
                # If prediction failed, format error message
                if not predicted_code:
                    if error:
                        error_display = f"ERROR: {error[:50]}" if len(error) > 50 else f"ERROR: {error}"
                        return idx, error_display, "", tokens, cost, error
                    else:
                        return idx, "ERROR: No prediction returned", "", tokens, cost, "No prediction returned from API"
                
                return idx, predicted_code, explanation, tokens, cost, error
            except Exception as e:
                error_msg = f"Unexpected error processing row {idx}: {str(e)}"
                logger.error(error_msg)
                return idx, f"ERROR: {type(e).__name__}", 0, 0.0, error_msg
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, idx, row): idx for idx, row in df.iterrows()}
            
            completed = 0
            for future in as_completed(futures):
                idx, predicted_code, explanation, tokens, cost, error = future.result()
                # Use the returned prediction (which may already contain "ERROR: ..." format)
                predictions[idx] = predicted_code if predicted_code else "ERROR: No prediction"
                explanations[idx] = explanation if explanation else ""
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
        df.insert(insert_index + 2, "Code Explanation", explanations)
        df.insert(insert_index + 3, "Model Source", model_sources)
        df.insert(insert_index + 4, "Error Message", errors_list)
        
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


def pdf_pages_to_base64_images(pdf_path, n_pages=1, dpi=200):
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


def predict_codes_from_pdfs_api(pdf_folder, output_file, n_pages=1, model="openai/gpt-5.2:online", api_key=None, max_workers=3, progress_callback=None, custom_instructions=None, include_code_list=True, image_cache=None, web_search=True):
    """
    Predict ASA codes from PDF files using OpenRouter vision model

    Args:
        pdf_folder: Path to folder containing PDF files
        output_file: Path to output CSV file
        n_pages: Number of pages to extract from each PDF (default 1)
        model: OpenRouter model to use (default openai/gpt-5.2:online). Must use format "openai/gpt-5.2" or "openai/gpt-5.2:online" for OpenRouter
        api_key: OpenRouter API key
        max_workers: Number of concurrent threads
        progress_callback: Optional callback function(completed, total, message)
        custom_instructions: Optional custom instructions to append to the prompt
        include_code_list: Whether to include the complete CPT code list in the prompt (default True)
        image_cache: Optional dict mapping PDF filename to list of base64 images (for speed optimization)
        web_search: Whether to enable web search for code validation (default True)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import glob
        from pathlib import Path
        
        logger.info(f"Starting image-based CPT prediction with model: {model}")
        logger.info(f"Extracting {n_pages} page(s) from each PDF")
        logger.info(f"Include CPT code list: {include_code_list}")
        
        # Load CPT codes reference only if needed
        cpt_codes_text = load_cpt_codes() if include_code_list else ""
        
        # Find all PDF files in folder (recursively search subdirectories)
        pdf_folder_path = Path(pdf_folder)
        
        # Search recursively for PDF files (case-insensitive)
        pdf_files = []
        for ext in ['*.pdf', '*.PDF']:
            pdf_files.extend(pdf_folder_path.glob(f"**/{ext}"))

        # Deduplicate PDF files (on case-insensitive filesystems like macOS, *.pdf and *.PDF match the same files)
        pdf_files = list(set(pdf_files))

        # Filter out __MACOSX metadata files
        pdf_files = [f for f in pdf_files if '__MACOSX' not in str(f)]

        # Also check case-insensitive manually
        if not pdf_files:
            all_files = list(pdf_folder_path.rglob("*"))
            pdf_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.pdf' and '__MACOSX' not in str(f)]

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
                    return idx, filename, "ERROR", "", 0, 0.0, error_msg, "openrouter_vision"
                
                # Predict ASA code from images
                predicted_code, explanation, tokens, cost, error = predict_asa_code_from_images(
                    image_data_list, cpt_codes_text, model, api_key, custom_instructions, include_code_list, web_search
                )
                
                # Determine model source
                model_source = "gemini_vision" if is_gemini_model(model) else "openrouter_vision"
                
                # If prediction failed, use error message
                if not predicted_code:
                    if error:
                        # Put error in ASA Code column for visibility, but also keep in Error Message
                        return idx, filename, f"ERROR: {error[:50]}", "", tokens, cost, error, model_source
                    else:
                        return idx, filename, "ERROR: No prediction returned", "", tokens, cost, "No prediction returned from API", model_source
                
                return idx, filename, predicted_code, explanation, tokens, cost, error, model_source
                
            except Exception as e:
                error_msg = f"Unexpected error processing {filename}: {str(e)}"
                logger.error(error_msg)
                return idx, filename, f"ERROR: {type(e).__name__}", "", 0, 0.0, error_msg, "openrouter_vision"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pdf, idx, pdf_path): idx for idx, pdf_path in enumerate(pdf_files)}
            
            completed = 0
            for future in as_completed(futures):
                idx, filename, predicted_code, explanation, tokens, cost, error, model_source = future.result()
                results[idx] = {
                    'filename': filename,
                    'prediction': predicted_code,
                    'explanation': explanation,
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
        explanations = [results[i]['explanation'] for i in range(len(pdf_files))]
        tokens_list = [results[i]['tokens'] for i in range(len(pdf_files))]
        costs_list = [results[i]['cost'] for i in range(len(pdf_files))]
        errors_list = [results[i]['error'] for i in range(len(pdf_files))]
        model_sources = [results[i]['model_source'] for i in range(len(pdf_files))]
        
        # Create dataframe with results
        df = pd.DataFrame({
            'Patient Filename': filenames,
            'ASA Code': predictions,
            'Procedure Code': predictions,
            'Code Explanation': explanations,
            'Model Source': model_sources,
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


def predict_icd_codes_from_pdfs_api(pdf_folder, output_file, n_pages=1, model="openai/gpt-5.2:online", api_key=None, max_workers=3, progress_callback=None, custom_instructions=None, image_cache=None, cpt_lookup=None):
    """
    Predict ICD codes from PDF files using OpenRouter vision model

    Args:
        pdf_folder: Path to folder containing PDF files
        output_file: Path to output CSV file
        n_pages: Number of pages to extract from each PDF (default 1)
        model: OpenRouter model to use (default openai/gpt-5.2:online). Must use format "openai/gpt-5.2" or "openai/gpt-5.2:online" for OpenRouter
        api_key: OpenRouter API key
        max_workers: Number of concurrent threads
        progress_callback: Optional callback function(completed, total, message)
        custom_instructions: Optional custom instructions to append to the prompt
        image_cache: Optional dict mapping PDF filename to list of base64 images (for speed optimization)
        cpt_lookup: Optional dict mapping PDF filename -> predicted CPT code. When provided, each
            per-PDF ICD call receives the CPT as guidance injected into the prompt.

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

        # Deduplicate PDF files (on case-insensitive filesystems like macOS, *.pdf and *.PDF match the same files)
        pdf_files = list(set(pdf_files))

        # Filter out __MACOSX metadata files
        pdf_files = [f for f in pdf_files if '__MACOSX' not in str(f)]

        # Also check case-insensitive manually
        if not pdf_files:
            all_files = list(pdf_folder_path.rglob("*"))
            pdf_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.pdf' and '__MACOSX' not in str(f)]

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
                # Use cached images if available, otherwise extract
                if image_cache and filename in image_cache:
                    image_data_list = image_cache[filename]
                    logger.debug(f"Using cached images for {filename}")
                else:
                    # Extract pages as base64 images
                    image_data_list = pdf_pages_to_base64_images(str(pdf_path), n_pages=n_pages)
                    # Cache for potential reuse
                    if image_cache is not None:
                        image_cache[filename] = image_data_list
                
                if not image_data_list:
                    error_msg = f"Failed to extract PDF pages from {filename}. File may be corrupted or invalid."
                    model_source = "gemini_vision" if is_gemini_model(model) else "openrouter_vision"
                    error_dict = {
                        "ICD1": "ERROR",
                        "ICD1_Reasoning": "",
                        "ICD2": "",
                        "ICD2_Reasoning": "",
                        "ICD3": "",
                        "ICD3_Reasoning": "",
                        "ICD4": "",
                        "ICD4_Reasoning": ""
                    }
                    return idx, filename, error_dict, 0, 0.0, error_msg, model_source
                
                # Predict ICD codes from images (with optional CPT guidance from prior step)
                per_pdf_cpt = cpt_lookup.get(filename) if cpt_lookup else None
                icd_codes_dict, tokens, cost, error = predict_icd_codes_from_images(
                    image_data_list, model, api_key, custom_instructions, predicted_cpt=per_pdf_cpt
                )
                
                # Determine model source
                model_source = "gemini_vision" if is_gemini_model(model) else "openrouter_vision"
                
                # If prediction failed, use error message
                if not icd_codes_dict:
                    if error:
                        error_dict = {
                            "ICD1": f"ERROR: {error[:30]}",
                            "ICD1_Reasoning": "",
                            "ICD2": "",
                            "ICD2_Reasoning": "",
                            "ICD3": "",
                            "ICD3_Reasoning": "",
                            "ICD4": "",
                            "ICD4_Reasoning": ""
                        }
                        return idx, filename, error_dict, tokens, cost, error, model_source
                    else:
                        error_dict = {
                            "ICD1": "ERROR: No prediction",
                            "ICD1_Reasoning": "",
                            "ICD2": "",
                            "ICD2_Reasoning": "",
                            "ICD3": "",
                            "ICD3_Reasoning": "",
                            "ICD4": "",
                            "ICD4_Reasoning": ""
                        }
                        return idx, filename, error_dict, tokens, cost, "No prediction returned from API", model_source
                
                correct_icd_codes(icd_codes_dict)
                return idx, filename, icd_codes_dict, tokens, cost, error, model_source

            except Exception as e:
                error_msg = f"Unexpected error processing {filename}: {str(e)}"
                logger.error(error_msg)
                model_source = "gemini_vision" if is_gemini_model(model) else "openrouter_vision"
                error_dict = {
                    "ICD1": f"ERROR: {type(e).__name__}",
                    "ICD1_Reasoning": "",
                    "ICD2": "",
                    "ICD2_Reasoning": "",
                    "ICD3": "",
                    "ICD3_Reasoning": "",
                    "ICD4": "",
                    "ICD4_Reasoning": ""
                }
                return idx, filename, error_dict, 0, 0.0, error_msg, model_source
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pdf, idx, pdf_path): idx for idx, pdf_path in enumerate(pdf_files)}
            
            completed = 0
            for future in as_completed(futures):
                idx, filename, icd_codes_dict, tokens, cost, error, model_source = future.result()
                results[idx] = {
                    'filename': filename,
                    'icd1': icd_codes_dict.get('ICD1', ''),
                    'icd1_reasoning': icd_codes_dict.get('ICD1_Reasoning', ''),
                    'icd2': icd_codes_dict.get('ICD2', ''),
                    'icd2_reasoning': icd_codes_dict.get('ICD2_Reasoning', ''),
                    'icd3': icd_codes_dict.get('ICD3', ''),
                    'icd3_reasoning': icd_codes_dict.get('ICD3_Reasoning', ''),
                    'icd4': icd_codes_dict.get('ICD4', ''),
                    'icd4_reasoning': icd_codes_dict.get('ICD4_Reasoning', ''),
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
        icd1_reasoning_list = [results[i]['icd1_reasoning'] for i in range(len(pdf_files))]
        icd2_list = [results[i]['icd2'] for i in range(len(pdf_files))]
        icd2_reasoning_list = [results[i]['icd2_reasoning'] for i in range(len(pdf_files))]
        icd3_list = [results[i]['icd3'] for i in range(len(pdf_files))]
        icd3_reasoning_list = [results[i]['icd3_reasoning'] for i in range(len(pdf_files))]
        icd4_list = [results[i]['icd4'] for i in range(len(pdf_files))]
        icd4_reasoning_list = [results[i]['icd4_reasoning'] for i in range(len(pdf_files))]
        tokens_list = [results[i]['tokens'] for i in range(len(pdf_files))]
        costs_list = [results[i]['cost'] for i in range(len(pdf_files))]
        errors_list = [results[i]['error'] for i in range(len(pdf_files))]
        model_sources = [results[i]['model_source'] for i in range(len(pdf_files))]
        
        # Create dataframe with results
        df = pd.DataFrame({
            'Patient Filename': filenames,
            'ICD1': icd1_list,
            'ICD1 Reasoning': icd1_reasoning_list,
            'ICD2': icd2_list,
            'ICD2 Reasoning': icd2_reasoning_list,
            'ICD3': icd3_list,
            'ICD3 Reasoning': icd3_reasoning_list,
            'ICD4': icd4_list,
            'ICD4 Reasoning': icd4_reasoning_list,
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


# ==============================================================================
# COMBINED CPT + ICD PREDICTION (single AI call)
# ==============================================================================

def predict_cpt_and_icd_from_images_gemini(image_data_list, cpt_codes_text, model="gemini-3-flash-preview", api_key=None, cpt_custom_instructions=None, icd_custom_instructions=None, include_code_list=True):
    """
    Predict both CPT and ICD codes in a single Gemini API call.

    Returns:
        tuple: (cpt_code, cpt_explanation, icd_dict, tokens_used, cost_estimate, error_message)
        icd_dict has keys ICD1..ICD4 and ICD1_Reasoning..ICD4_Reasoning
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return None, "", None, 0, 0.0, "Google GenAI SDK not available"

    model = normalize_gemini_model(model)

    api_key_value = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key_value:
        return None, "", None, 0, 0.0, "No Google API key provided"

    try:
        client = genai.Client(api_key=api_key_value)
    except Exception as e:
        return None, "", None, 0, 0.0, f"Failed to initialize Google GenAI client: {str(e)}"

    cpt_code_section = f"""
Here is the reference list of valid anesthesia CPT codes:

{cpt_codes_text}
""" if include_code_list and cpt_codes_text else ""

    prompt = f"""You are a medical coding specialist with expertise in both anesthesia CPT coding and ICD-10 diagnosis coding.

Your task is to analyze the provided medical document page(s) and return BOTH the anesthesia CPT code AND the ICD diagnosis codes in a single response.

=== PART 1: CPT CODE ===
{cpt_code_section}
CRITICAL CPT CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) ONLY if:
     * The document explicitly states "screening colonoscopy"
     * Procedure description includes the word "screening"
     * Pre-op diagnosis is Z12.11 (Encounter for screening colonoscopy) with NO symptom or surveillance indication
     * Pre-op states "Colon cancer screening"
     * Patient has ONLY family history (Z80.0) as indication — no personal history, no symptoms
   - Use 00811 (diagnostic colonoscopy) if ANY of these are present:
     * Indication says "polyp surveillance", "surveillance colonoscopy", or "follow-up"
     * Patient has personal history of colon polyps (Z86.010x) or colon cancer — this is SURVEILLANCE, NOT screening
     * Investigating specific symptoms (bleeding, pain, diarrhea, anemia, weight loss, etc.)
     * Any GI diagnosis as indication (GERD, diverticulosis, IBD, etc.)
   - Key distinction: "screening" = routine, no prior findings. "surveillance" = follow-up due to prior polyps/cancer = DIAGNOSTIC = 00811
   - When uncertain between screening and surveillance: if patient has ANY personal history of polyps or cancer, use 00811

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

4. PERCUTANEOUS LUMBAR SPINAL INTERVENTION CODING:
   - If the procedure is a percutaneous lumbar spinal intervention (such as a Medial Branch Block or Facet Injection) -> use 01938

5. SACRAL NEUROMODULATION / INTERSTIM CODING:
   - InterStim implant/placement (Stage 1 or Stage 2), sacral nerve stimulator, or sacral neuromodulation procedures -> use 00400 (integumentary system), NOT 00860
   - These are subcutaneous implant procedures, not intraperitoneal

6. TRANSRECTAL / TRANSPERINEAL PROSTATE PROCEDURES:
   - Transrectal ultrasound (TRUS) biopsy of prostate -> use 00902 (anorectal), NOT 00860
   - Transperineal prostate procedures (barrigel injection, fiducial marker placement, SpaceOAR, biopsy) -> use 00902 (anorectal/perineal), NOT 00860
   - Do NOT code prostate biopsies or prostate marker placements as 00860 (intraperitoneal)

7. INTRAMEDULLARY FEMORAL NAILING / GAMMA NAILING:
   - Intramedullary nailing of the femur (gamma nailing, femoral nailing, IM nail/rod) for hip fractures (intertrochanteric, femoral neck, subtrochanteric) -> use 01230 (upper 2/3 of femur), NOT 01210 (hip joint)
   - Even though the diagnosis is a "hip fracture," the surgical procedure is on the femur (nail/rod insertion), so use 01230
   - 01210 is for procedures directly on the hip joint (e.g., hip pinning with screws only, hip hemiarthroplasty without nailing)

IMPORTANT: Look at the document images carefully to identify:
- Procedure description
- Pre-operative diagnosis
- Post-operative diagnosis
- Any relevant medical information that can help determine the correct anesthesia CPT code

Give me the most relevant anesthesia CPT code for anesthesia billing for this certain procedure.

=== PART 2: ICD DIAGNOSIS CODES ===

CRITICAL ICD CODING RULES:

1. Analyze the entire PDF document carefully to understand the procedure and patient condition
2. Identify the PRIMARY diagnosis (ICD1) - this should be the main reason for the procedure
3. CRITICAL: Never code both the pre operative and post operative diagnoses, instead code only post operative diagnosis + supportive diagnoses. There might be many additional diagnoses listed, but choose the ones that are most supportive of necessity for the procedure.
4. CRITICAL COLONOSCOPY CODING RULES:
   - General screening colonoscopy with NO findings: Code Z12.11 in ICD1. If patient has family history of GI malignancy, add Z80.0 in ICD2. If patient has personal history of colonic polyps, add Z86.0100 in ICD2 or ICD3. Leave remaining slots empty.
   - Screening colonoscopy with polypectomy/polyp found: Code Z12.11 in ICD1, K63.5 in ICD2 (or D12.x site-specific benign neoplasm if pathology specifies location). Add K57.30 (diverticulosis) in ICD3 if documented. Add K64.0/K64.9 (hemorrhoids) in ICD4 if documented.
   - Non-screening colonoscopy (diagnostic, with symptoms/findings): Use the actual finding or indication as ICD1 (e.g., K63.5 for polyp, K21.9 for GERD, K57.30 for diverticulosis, K44.9 for hernia). Put Z12.11 in ICD2 only if screening was also part of the reason.
   - ORDERING PRIORITY for colonoscopy ICD2-4: History codes (Z80.0, Z86.0100) > additional findings (K57.30, K63.5, K64.0) > comorbidities (I10, E11.9)
5. CRITICAL VAGINAL DELIVERY / C-SECTION CODING RULES:
   - If the procedure is a vaginal delivery (CPT 01967) or C-section (CPT 01961) with NO documented complications: Use O80 in ICD1.
   - EXCEPTION: If the record documents complications such as tobacco/substance use in pregnancy (O99.214, O99.314), preeclampsia (O14.x), fetal distress (O77.x), gestational diabetes (O24.x), or other obstetric complications, use the complication code as ICD1 instead of O80.
   - For C-sections: Check for prior uterine scar codes (O34.21x). Use O34.219 (unspecified laterality) unless the document explicitly states which side.
6. CRITICAL CATARACT EXTRACTION CODING RULE:
   - If the procedure is cataract extraction surgery: Use the appropriate cataract ICD-10 diagnosis code as ICD1 (e.g., H25.811 for right eye, H25.812 for left eye). Do NOT use CPT codes (like 00142) as ICD codes.
   - EXCEPTION: If the procedure includes BOTH cataract extraction AND a glaucoma procedure (e.g., iStent, goniotomy), use the glaucoma diagnosis (H40.x) as ICD1.
7. CRITICAL COLONOSCOPY/EGD DISTINCTION:
   - Z12.11 must ONLY be used for colonoscopy procedures. NEVER use Z12.11 for EGD/upper endoscopy.
   - ⚠️ EGD (00731/00813) ICD1 RULE: NEVER use K31.7 (polyp of stomach/duodenum) as ICD1 unless the post-op findings explicitly document a stomach or duodenal polyp. K31.7 is wrong in 90%+ of cases. Instead use the actual indication/finding from the post-op diagnosis: K29.70 (gastritis), K44.9 (hiatal hernia), K21.9 (GERD), K22.2 (esophageal obstruction), R13.10 (dysphagia), K92.0 (hematemesis), K25.9 (gastric ulcer), K22.89 (other esophageal disease). For combined EGD + colonoscopy (00813), the upper GI indication goes in ICD1.
8. ⚠️ POST-OP DIAGNOSIS ALWAYS TAKES PRIORITY: Always read and use the Post-Operative Diagnosis for ICD1, NOT the Pre-Op Diagnosis. The post-op is the definitive diagnosis for coding. Only fall back to pre-op if no post-op diagnosis is documented.
9. COMORBIDITY RULE BASED ON ASA STATUS: If the patient's ASA status is 1 or 2, do NOT add comorbidities from the medical history — only use ICD codes directly related to the procedure/diagnosis. If ASA is 3 or higher, you MAY add relevant comorbidities (I10, E11.9, G47.33, I25.10, E78.5, J44.9) in ICD2-4. ASA 1-2 = procedure diagnosis only. ASA 3+ = include significant comorbidities.
10. DO NOT USE these as routine filler codes: E66.9/E66.01 (obesity), Z68.x (BMI codes), F41.9 (anxiety), K21.9 (GERD), F17.210 (nicotine), R-codes (symptoms). NEVER use Z68.x BMI codes for anesthesia billing.
11. PREFERRED comorbidity ordering (ASA 3+ only): I10 > E78.5 > J44.9 > G47.33 > E11.9 > E03.9.
12. LATERALITY AND SPECIFICITY RULE: When the procedure specifies a side (Right, Left, Bilateral), use the laterality-specific ICD code, NOT the bilateral or unspecified variant. Match ICD laterality to procedure laterality. Use 7th character "A" (initial encounter) unless explicitly stated otherwise.
13. Use standard ICD-10 format (e.g., "E11.9", "I10", "Z87.891")
14. Check for Excludes1 conflicts before finalizing codes
15. Use web search to verify all ICD codes are current as of 2025
16. Leave unused ICD fields as empty strings

=== OUTPUT FORMAT ===

Respond with ONLY a JSON object in this exact format:
{{
  "CPT": "00840",
  "CPT_Explanation": "Brief explanation of CPT code selection (1-2 sentences)",
  "ICD1": "primary_diagnosis_code",
  "ICD1_Reasoning": "Brief explanation (1-2 sentences)",
  "ICD2": "secondary_code_or_empty",
  "ICD2_Reasoning": "Brief explanation or empty string",
  "ICD3": "tertiary_code_or_empty",
  "ICD3_Reasoning": "Brief explanation or empty string",
  "ICD4": "quaternary_code_or_empty",
  "ICD4_Reasoning": "Brief explanation or empty string"
}}

Respond with ONLY the JSON object, nothing else."""

    if cpt_custom_instructions and cpt_custom_instructions.strip():
        prompt += f"\n\nCUSTOM CPT INSTRUCTIONS:\n{cpt_custom_instructions.strip()}"
    if icd_custom_instructions and icd_custom_instructions.strip():
        prompt += f"\n\nCUSTOM ICD INSTRUCTIONS:\n{icd_custom_instructions.strip()}"

    parts = [types.Part.from_text(text=prompt)]
    for img_data in image_data_list:
        img_bytes = base64.b64decode(img_data)
        parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))

    contents = [types.Content(role="user", parts=parts)]

    if model in ("gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-pro-preview"):
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)

    tools = [types.Tool(googleSearch=types.GoogleSearch())]
    use_flex = True
    FLEX_TIMEOUT = 600

    max_retries = 5
    for attempt in range(max_retries):
        try:
            service_tier = "flex" if use_flex else "standard"
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                thinking_config=thinking_config,
                tools=tools,
                http_options=types.HttpOptions(
                    extra_body={"serviceTier": service_tier},
                    timeout=FLEX_TIMEOUT * 1000 if use_flex else None,
                ),
            )

            full_response = ""
            request_start = time.time()
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:
                    full_response += chunk.text
                if use_flex and (time.time() - request_start) > FLEX_TIMEOUT:
                    raise TimeoutError("Flex request exceeded 10 minute timeout")

            response_text = full_response.strip()
            if not response_text:
                raise ValueError("Empty response from API")

            cleaned = response_text
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result_dict = json.loads(cleaned)

            cpt_code = result_dict.get("CPT", "")
            cpt_explanation = result_dict.get("CPT_Explanation", "")
            icd_dict = {
                "ICD1": result_dict.get("ICD1", ""),
                "ICD1_Reasoning": result_dict.get("ICD1_Reasoning", ""),
                "ICD2": result_dict.get("ICD2", ""),
                "ICD2_Reasoning": result_dict.get("ICD2_Reasoning", ""),
                "ICD3": result_dict.get("ICD3", ""),
                "ICD3_Reasoning": result_dict.get("ICD3_Reasoning", ""),
                "ICD4": result_dict.get("ICD4", ""),
                "ICD4_Reasoning": result_dict.get("ICD4_Reasoning", ""),
            }
            correct_icd_codes(icd_dict)

            logger.info(f"Combined prediction: CPT={cpt_code}, ICD1={icd_dict['ICD1']}")
            return cpt_code, cpt_explanation, icd_dict, 0, 0.0, None

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None, "", None, 0, 0.0, f"JSON parse error after {max_retries} attempts: {str(e)}"
            time.sleep(2 ** attempt)

        except (TimeoutError, Exception) as e:
            if use_flex and not isinstance(e, json.JSONDecodeError):
                reason = "timeout" if isinstance(e, TimeoutError) else str(e)[:80]
                logger.warning(f"Flex failed on combined prediction ({reason}), switching to standard tier")
                use_flex = False
                continue

            logger.warning(f"API error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None, "", None, 0, 0.0, str(e)
            time.sleep(2 ** attempt)

    return None, "", None, 0, 0.0, "All retries failed"


def predict_cpt_and_icd_from_images(image_data_list, cpt_codes_text, model="openai/gpt-5.2:online", api_key=None, cpt_custom_instructions=None, icd_custom_instructions=None, include_code_list=True):
    """
    Route combined CPT+ICD prediction to Gemini or OpenRouter.

    Returns:
        tuple: (cpt_code, cpt_explanation, icd_dict, tokens_used, cost_estimate, error_message)
    """
    if is_gemini_model(model):
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if google_api_key and GOOGLE_GENAI_AVAILABLE:
            return predict_cpt_and_icd_from_images_gemini(
                image_data_list, cpt_codes_text, model, google_api_key,
                cpt_custom_instructions, icd_custom_instructions, include_code_list
            )
        else:
            # Fall through to OpenRouter with google/ prefix
            model = f"google/{normalize_gemini_model(model)}"

    # OpenRouter path
    api_key_value = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key_value:
        return None, "", None, 0, 0.0, "No API key provided"

    cpt_codes_text_loaded = load_cpt_codes() if include_code_list and not cpt_codes_text else cpt_codes_text

    cpt_code_section = f"""
Here is the reference list of valid anesthesia CPT codes:

{cpt_codes_text_loaded}
""" if include_code_list and cpt_codes_text_loaded else ""

    prompt = f"""You are a medical coding specialist with expertise in both anesthesia CPT coding and ICD-10 diagnosis coding.

Your task is to analyze the provided medical document page(s) and return BOTH the anesthesia CPT code AND the ICD diagnosis codes in a single response.

=== PART 1: CPT CODE ==={cpt_code_section}
CRITICAL CPT CODING RULES (FOLLOW THESE EXACTLY):

1. COLONOSCOPY CODING (Most Common Errors):
   - Use 00812 (screening colonoscopy) ONLY if:
     * The document explicitly states "screening colonoscopy"
     * Procedure description includes the word "screening"
     * Pre-op diagnosis is Z12.11 with NO symptom or surveillance indication
     * Patient has ONLY family history (Z80.0) as indication
   - Use 00811 (diagnostic colonoscopy) if ANY of these are present:
     * Indication says "polyp surveillance", "surveillance colonoscopy", or "follow-up"
     * Patient has personal history of colon polyps (Z86.010x) or colon cancer — this is SURVEILLANCE = DIAGNOSTIC = 00811
     * Investigating specific symptoms (bleeding, pain, diarrhea, anemia, weight loss)
   - Key distinction: "screening" = routine. "surveillance" = follow-up = DIAGNOSTIC = 00811

2. MRI/CT SCAN: use 01922
3. TEE: use 01922
4. PERCUTANEOUS LUMBAR SPINAL INTERVENTION (Medial Branch Block, Facet Injection): use 01938
5. SACRAL NEUROMODULATION / INTERSTIM: use 00400 (integumentary), NOT 00860
6. TRANSRECTAL / TRANSPERINEAL PROSTATE PROCEDURES: use 00902 (anorectal), NOT 00860
7. INTRAMEDULLARY FEMORAL NAILING: use 01230 (upper 2/3 femur), NOT 01210 (hip joint)

IMPORTANT: Look at procedure description, pre-op diagnosis, post-op diagnosis to determine the correct CPT code.

=== PART 2: ICD DIAGNOSIS CODES ===
CRITICAL ICD CODING RULES:
1. ICD1 = primary diagnosis. Use POST-OPERATIVE diagnosis over pre-op when available.
2. COLONOSCOPY: Screening with no findings: Z12.11 in ICD1. With polypectomy: Z12.11 in ICD1, K63.5 in ICD2. Diagnostic: actual finding in ICD1, Z12.11 in ICD2 only if screening was also part of the reason.
3. EGD (00731/00813): NEVER use K31.7 as ICD1 unless stomach polyp explicitly documented. Use K29.70 (gastritis), K44.9 (hernia), K21.9 (GERD), R13.10 (dysphagia), K92.0 (hematemesis) instead. For combined EGD+colonoscopy, upper GI indication goes in ICD1.
4. Vaginal delivery (01967) with no complications: O80 in ICD1. With complications: use complication code.
5. Cataract extraction: Use cataract ICD code (H25.x) as ICD1, NOT CPT code 00142.
6. COMORBIDITY RULE: ASA 1-2 = procedure diagnosis only, no filler comorbidities. ASA 3+ = may include I10, E78.5, J44.9, G47.33, E11.9. NEVER use E66.9 (obesity), Z68.x (BMI), F41.9 (anxiety), F17.210 (nicotine) as fillers.
7. LATERALITY: Match ICD laterality to procedure side. Right procedure = right-specific code, not bilateral.
8. Standard ICD-10 format. Check Excludes1 conflicts. Leave unused fields empty.

=== OUTPUT FORMAT ===
Respond with ONLY a JSON object:
{{
  "CPT": "00840",
  "CPT_Explanation": "Brief explanation (1-2 sentences)",
  "ICD1": "primary_code",
  "ICD1_Reasoning": "Brief explanation",
  "ICD2": "",
  "ICD2_Reasoning": "",
  "ICD3": "",
  "ICD3_Reasoning": "",
  "ICD4": "",
  "ICD4_Reasoning": ""
}}"""

    if cpt_custom_instructions and cpt_custom_instructions.strip():
        prompt += f"\n\nCUSTOM CPT INSTRUCTIONS:\n{cpt_custom_instructions.strip()}"
    if icd_custom_instructions and icd_custom_instructions.strip():
        prompt += f"\n\nCUSTOM ICD INSTRUCTIONS:\n{icd_custom_instructions.strip()}"

    headers = {
        "Authorization": f"Bearer {api_key_value}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://medical-data-processor.com",
        "X-Title": "Medical Data Processor",
    }

    messages_content = [{"type": "text", "text": prompt}]
    for img_data in image_data_list:
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_data}"},
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": messages_content}],
        "temperature": 0,
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            content_text = data["choices"][0]["message"]["content"].strip()

            cleaned = content_text
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result_dict = json.loads(cleaned)
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            cost = tokens * 0.000015

            cpt_code = result_dict.get("CPT", "")
            cpt_explanation = result_dict.get("CPT_Explanation", "")
            icd_dict = {
                "ICD1": result_dict.get("ICD1", ""),
                "ICD1_Reasoning": result_dict.get("ICD1_Reasoning", ""),
                "ICD2": result_dict.get("ICD2", ""),
                "ICD2_Reasoning": result_dict.get("ICD2_Reasoning", ""),
                "ICD3": result_dict.get("ICD3", ""),
                "ICD3_Reasoning": result_dict.get("ICD3_Reasoning", ""),
                "ICD4": result_dict.get("ICD4", ""),
                "ICD4_Reasoning": result_dict.get("ICD4_Reasoning", ""),
            }
            correct_icd_codes(icd_dict)
            return cpt_code, cpt_explanation, icd_dict, tokens, cost, None

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None, "", None, 0, 0.0, f"JSON parse error: {str(e)}"
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"API error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None, "", None, 0, 0.0, str(e)
            time.sleep(2 ** attempt)

    return None, "", None, 0, 0.0, "All retries failed"


def predict_cpt_and_icd_from_pdfs_api(
    pdf_folder,
    output_file,
    n_pages=1,
    model="openai/gpt-5.2:online",
    api_key=None,
    max_workers=50,
    progress_callback=None,
    cpt_custom_instructions=None,
    icd_custom_instructions=None,
    include_code_list=True,
    image_cache=None,
):
    """
    Predict both CPT and ICD codes from PDFs in a single AI call per PDF.

    Output CSV columns: Patient Filename, ASA Code, Procedure Code, Code Explanation,
    ICD1, ICD1 Reasoning, ICD2, ICD2 Reasoning, ICD3, ICD3 Reasoning, ICD4, ICD4 Reasoning,
    Model Source, Tokens Used, Cost (USD), Error Message
    """
    try:
        logger.info(f"Starting combined CPT+ICD prediction with model: {model}")

        cpt_codes_text = load_cpt_codes() if include_code_list else ""

        pdf_folder_path = Path(pdf_folder)
        pdf_files = []
        for ext in ["*.pdf", "*.PDF"]:
            pdf_files.extend(pdf_folder_path.glob(f"**/{ext}"))
        pdf_files = list(set(pdf_files))
        pdf_files = [f for f in pdf_files if "__MACOSX" not in str(f)]
        if not pdf_files:
            all_files = list(pdf_folder_path.rglob("*"))
            pdf_files = [f for f in all_files if f.is_file() and f.suffix.lower() == ".pdf" and "__MACOSX" not in str(f)]
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_folder}")
            return False

        pdf_files = sorted(pdf_files, key=lambda x: x.name)
        logger.info(f"Found {len(pdf_files)} PDF files to process (combined CPT+ICD)")

        results = {}

        if progress_callback:
            progress_callback(0, len(pdf_files), "Starting combined CPT+ICD predictions...")

        def process_pdf(idx, pdf_path):
            filename = pdf_path.name
            try:
                if image_cache and filename in image_cache:
                    image_data_list = image_cache[filename]
                else:
                    image_data_list = pdf_pages_to_base64_images(str(pdf_path), n_pages=n_pages)
                    if image_cache is not None:
                        image_cache[filename] = image_data_list

                if not image_data_list:
                    empty_icd = {k: "" for k in ["ICD1", "ICD1_Reasoning", "ICD2", "ICD2_Reasoning", "ICD3", "ICD3_Reasoning", "ICD4", "ICD4_Reasoning"]}
                    return idx, filename, "ERROR", "", empty_icd, 0, 0.0, "Failed to extract PDF pages", "combined_vision"

                cpt_code, cpt_explanation, icd_dict, tokens, cost, error = predict_cpt_and_icd_from_images(
                    image_data_list, cpt_codes_text, model, api_key,
                    cpt_custom_instructions, icd_custom_instructions, include_code_list
                )
                model_source = "gemini_vision" if is_gemini_model(model) else "openrouter_vision"

                if not cpt_code and not icd_dict:
                    empty_icd = {k: "" for k in ["ICD1", "ICD1_Reasoning", "ICD2", "ICD2_Reasoning", "ICD3", "ICD3_Reasoning", "ICD4", "ICD4_Reasoning"]}
                    return idx, filename, f"ERROR: {error[:50] if error else 'No prediction'}", "", empty_icd, tokens, cost, error, model_source

                return idx, filename, cpt_code or "", cpt_explanation or "", icd_dict or {k: "" for k in ["ICD1", "ICD1_Reasoning", "ICD2", "ICD2_Reasoning", "ICD3", "ICD3_Reasoning", "ICD4", "ICD4_Reasoning"]}, tokens, cost, error, model_source

            except Exception as e:
                error_msg = f"Unexpected error processing {filename}: {str(e)}"
                logger.error(error_msg)
                empty_icd = {k: "" for k in ["ICD1", "ICD1_Reasoning", "ICD2", "ICD2_Reasoning", "ICD3", "ICD3_Reasoning", "ICD4", "ICD4_Reasoning"]}
                return idx, filename, f"ERROR: {type(e).__name__}", "", empty_icd, 0, 0.0, error_msg, "combined_vision"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pdf, idx, pdf_path): idx for idx, pdf_path in enumerate(pdf_files)}
            completed = 0
            for future in as_completed(futures):
                idx, filename, cpt_code, cpt_explanation, icd_dict, tokens, cost, error, model_source = future.result()
                results[idx] = {
                    "filename": filename,
                    "cpt_code": cpt_code,
                    "cpt_explanation": cpt_explanation,
                    "icd1": icd_dict.get("ICD1", ""),
                    "icd1_reasoning": icd_dict.get("ICD1_Reasoning", ""),
                    "icd2": icd_dict.get("ICD2", ""),
                    "icd2_reasoning": icd_dict.get("ICD2_Reasoning", ""),
                    "icd3": icd_dict.get("ICD3", ""),
                    "icd3_reasoning": icd_dict.get("ICD3_Reasoning", ""),
                    "icd4": icd_dict.get("ICD4", ""),
                    "icd4_reasoning": icd_dict.get("ICD4_Reasoning", ""),
                    "tokens": tokens,
                    "cost": cost,
                    "error": error,
                    "model_source": model_source,
                }
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(pdf_files), f"Processed {completed}/{len(pdf_files)} PDFs (CPT+ICD combined)...")

        n = len(pdf_files)
        df = pd.DataFrame({
            "Patient Filename": [results[i]["filename"] for i in range(n)],
            "ASA Code": [results[i]["cpt_code"] for i in range(n)],
            "Procedure Code": [results[i]["cpt_code"] for i in range(n)],
            "Code Explanation": [results[i]["cpt_explanation"] for i in range(n)],
            "ICD1": [results[i]["icd1"] for i in range(n)],
            "ICD1 Reasoning": [results[i]["icd1_reasoning"] for i in range(n)],
            "ICD2": [results[i]["icd2"] for i in range(n)],
            "ICD2 Reasoning": [results[i]["icd2_reasoning"] for i in range(n)],
            "ICD3": [results[i]["icd3"] for i in range(n)],
            "ICD3 Reasoning": [results[i]["icd3_reasoning"] for i in range(n)],
            "ICD4": [results[i]["icd4"] for i in range(n)],
            "ICD4 Reasoning": [results[i]["icd4_reasoning"] for i in range(n)],
            "Model Source": [results[i]["model_source"] for i in range(n)],
            "Tokens Used": [results[i]["tokens"] for i in range(n)],
            "Cost (USD)": [results[i]["cost"] for i in range(n)],
            "Error Message": [results[i]["error"] for i in range(n)],
        })

        total_tokens = sum(results[i]["tokens"] for i in range(n))
        total_cost = sum(results[i]["cost"] for i in range(n))
        logger.info(f"Combined CPT+ICD complete. Tokens: {total_tokens:,}, Cost: ${total_cost:.4f}")

        df.to_csv(output_file, index=False)
        logger.info(f"Saved combined results to {output_file}")

        if progress_callback:
            progress_callback(n, n, f"Combined CPT+ICD completed! Total cost: ${total_cost:.4f}")

        return True

    except Exception as e:
        logger.error(f"Error in predict_cpt_and_icd_from_pdfs_api: {e}")
        return False
