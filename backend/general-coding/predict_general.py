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


def predict_asa_code_from_images_gemini(image_data_list, cpt_codes_text, model="gemini-3-flash-preview", api_key=None, custom_instructions=None, include_code_list=True):
    """
    Predict ASA code using Google GenAI SDK from PDF page images
    
    Args:
        image_data_list: List of base64 encoded image strings
        cpt_codes_text: Reference text containing all valid CPT codes (ignored if include_code_list=False)
        model: Gemini model name (e.g., "gemini-3-flash-preview", "gemini-2.5-pro")
        api_key: Google API key
        custom_instructions: Optional custom instructions to append to the prompt
        include_code_list: Whether to include the complete CPT code list in the prompt (default True)
    
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

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

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

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

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
    
    # Configure thinking for Gemini 3 models
    if model == "gemini-3-pro-preview":
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    elif model == "gemini-3-flash-preview":
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
    
    # Enable web search for Gemini models
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        thinking_config=thinking_config,
        tools=tools
    )
    
    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Call Gemini API
            full_response = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:
                    full_response += chunk.text
            
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
            
            # Estimate tokens and cost (Gemini pricing estimates)
            # Rough estimates: assume ~4 tokens per character for images, ~1 token per character for text
            prompt_chars = len(prompt)
            image_tokens_estimate = len(image_data_list) * 1000  # Rough estimate per image
            response_chars = len(response_text)
            total_tokens_estimate = prompt_chars + image_tokens_estimate + response_chars
            
            # Cost estimation for Gemini models (rough estimates)
            if "gemini-flash-lite" in model:
                cost = total_tokens_estimate * 0.00005 / 1000  # Rough estimate - cheapest option
            elif "gemini-3-flash" in model:
                cost = total_tokens_estimate * 0.000125 / 1000  # Rough estimate
            elif "gemini-3-pro" in model:
                cost = total_tokens_estimate * 0.00125 / 1000  # Rough estimate
            elif "gemini-2.5-flash" in model:
                cost = total_tokens_estimate * 0.000075 / 1000  # Rough estimate
            else:
                cost = total_tokens_estimate * 0.0005 / 1000  # Default estimate
            
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
                # Try to extract code if JSON parsing fails
                code_match = re.search(r'\b0\d{4}\b', response_text)
                if code_match:
                    predicted_code = code_match.group(0)
                    return predicted_code, response_text.replace(predicted_code, "").strip(), 0, 0.0, None
                return None, "", 0, 0.0, error_message
                
        except Exception as e:
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


def predict_asa_code_from_images(image_data_list, cpt_codes_text, model="openai/gpt-5.2:online", api_key=None, custom_instructions=None, include_code_list=True):
    """
    Predict ASA code using OpenRouter API or Google GenAI SDK from PDF page images
    
    Args:
        image_data_list: List of base64 encoded image strings
        cpt_codes_text: Reference text containing all valid CPT codes (ignored if include_code_list=False)
        model: Model to use (default: openai/gpt-5.2:online). For Gemini, use format "gemini-3-flash-preview" or "gemini-2.5-pro"
        api_key: API key (OpenRouter API key for OpenAI models, Google API key for Gemini models)
        custom_instructions: Optional custom instructions to append to the prompt
        include_code_list: Whether to include the complete CPT code list in the prompt (default True)
    
    Returns:
        tuple: (predicted_code, tokens_used, cost_estimate, error_message)
    """
    # Check if using Gemini model
    if is_gemini_model(model):
        return predict_asa_code_from_images_gemini(image_data_list, cpt_codes_text, model, api_key, custom_instructions, include_code_list)
    
    # Use OpenRouter for non-Gemini models
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

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

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

2. MRI/CT SCAN CODING:
   - If the procedure is an MRI or CT scan -> use 01922

3. TEE (TRANSESOPHAGEAL ECHOCARDIOGRAM) CODING:
   - If the main procedure was TEE administration (TRANSESOPHAGEAL ECHO (TEE)) or similarly worded -> use 01922

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


def predict_icd_codes_from_images_gemini(image_data_list, model="gemini-3-flash-preview", api_key=None, custom_instructions=None):
    """
    Predict ICD codes using Google GenAI SDK from PDF page images
    
    Args:
        image_data_list: List of base64 encoded image strings
        model: Gemini model name (e.g., "gemini-3-flash-preview", "gemini-2.5-pro")
        api_key: Google API key
        custom_instructions: Optional custom instructions to append to the prompt
    
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
3. CRITICAL: If there is both pre-operative and post-operative diagnosis listed, always put in ICD1 the code for the post-operative diagnosis!
4. CRITICAL COLONOSCOPY CODING RULES:
   - General screening colonoscopy: Code Z12.11 ONLY in ICD1 and leave ICD2, ICD3, ICD4 empty (nothing else)
   - Screening colonoscopy with polypectomy/polyp removal: Code Z12.11 in ICD1 and K63.5 in ICD2, leave ICD3 and ICD4 empty
5. CRITICAL VAGINAL DELIVERY CODING RULE:
   - If the procedure is a planned vaginal delivery labor (CPT code 01967): Use O80 ONLY in ICD1 and leave ICD2, ICD3, ICD4 empty (nothing else)
6. Identify up to 3 additional ICD codes (ICD2, ICD3, ICD4) sorted by relevance to the procedure
7. Only include ICD codes that are directly relevant to the procedure or patient condition
8. If fewer than 4 relevant ICD codes exist, leave the remaining fields empty
9. Use standard ICD-10 format (e.g., "E11.9", "I10", "Z87.891")
10. CRITICAL: Use web search to verify that all ICD codes you provide are valid and current as of November 2025. Only use the most recent ICD codes that are valid in November 2025. Do not use outdated or invalid codes.
11. CRITICAL: If there is outdated ICD codes listed on record try to find on google valid icd codes updated as of december 2025, so basically take the diagnosis code and update it accordingly with google
12. CRITICAL: Always make sure to not just pick the main diagnosis but to also look at secondary diagnoses further in the record IF available, they will often not be listed clearly as codes but instead as small snippets of text, there might be many of them listed like obesity and diabetes and such... make sure to convert those small texts to diagnosis codes, but also make sure to pick the ones that are MOST related to the main procedure and diagnosis itself, also make sure to use UPDATED december 2025 codes with google search
13. CRITICAL: Check for Excludes1 Conflicts: Before finalizing the JSON, verify if the selected codes have "Excludes1" notes in the ICD-10 manual that prevent them from being billed together.
14. Conflict Resolution: If two codes conflict (e.g., J35.1 and J35.01), prioritize the Post-Operative or more specific diagnosis.

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
    
    # Configure thinking for Gemini 3 models
    if model == "gemini-3-pro-preview":
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    elif model == "gemini-3-flash-preview":
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
    
    # Enable web search for Gemini models (required for ICD code validation per prompt)
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        thinking_config=thinking_config,
        tools=tools
    )
    
    # Retry mechanism with exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Call Gemini API
            full_response = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:
                    full_response += chunk.text
            
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
            icd_codes_dict = json.loads(cleaned_response)
            
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
            
            # Estimate tokens and cost (Gemini pricing estimates)
            prompt_chars = len(prompt)
            image_tokens_estimate = len(image_data_list) * 1000  # Rough estimate per image
            response_chars = len(response_text)
            total_tokens_estimate = prompt_chars + image_tokens_estimate + response_chars
            
            # Cost estimation for Gemini models (rough estimates)
            if "gemini-flash-lite" in model:
                cost = total_tokens_estimate * 0.00005 / 1000  # Rough estimate - cheapest option
            elif "gemini-3-flash" in model:
                cost = total_tokens_estimate * 0.000125 / 1000  # Rough estimate
            elif "gemini-3-pro" in model:
                cost = total_tokens_estimate * 0.00125 / 1000  # Rough estimate
            elif "gemini-2.5-flash" in model:
                cost = total_tokens_estimate * 0.000075 / 1000  # Rough estimate
            else:
                cost = total_tokens_estimate * 0.0005 / 1000  # Default estimate
            
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
                # Try to extract codes manually if JSON parsing fails
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
                icd_pattern = r'[A-Z]\d{2}\.?\d*'
                found_codes = re.findall(icd_pattern, response_text)
                for i, code in enumerate(found_codes[:4]):
                    result[f"ICD{i+1}"] = code
                    result[f"ICD{i+1}_Reasoning"] = "Code extracted from response (reasoning not available)"
                return result, 0, 0.0, error_message
                
        except Exception as e:
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


def predict_icd_codes_from_images(image_data_list, model="openai/gpt-5.2:online", api_key=None, custom_instructions=None):
    """
    Predict ICD codes using OpenRouter API or Google GenAI SDK from PDF page images
    
    Args:
        image_data_list: List of base64 encoded image strings
        model: Model to use (default: openai/gpt-5.2:online). For Gemini, use format "gemini-3-flash-preview" or "gemini-2.5-pro"
        api_key: API key (OpenRouter API key for OpenAI models, Google API key for Gemini models)
        custom_instructions: Optional custom instructions to append to the prompt
    
    Returns:
        tuple: (icd_codes_dict, tokens_used, cost_estimate, error_message)
        icd_codes_dict: Dictionary with keys 'ICD1', 'ICD1_Reasoning', 'ICD2', 'ICD2_Reasoning', 'ICD3', 'ICD3_Reasoning', 'ICD4', 'ICD4_Reasoning' containing ICD codes and reasoning
    """
    # Check if using Gemini model
    if is_gemini_model(model):
        return predict_icd_codes_from_images_gemini(image_data_list, model, api_key, custom_instructions)
    
    # Use OpenRouter for non-Gemini models
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
3. CRITICAL: If there is both pre-operative and post-operative diagnosis listed, always put in ICD1 the code for the post-operative diagnosis!
4. CRITICAL COLONOSCOPY CODING RULES:
   - General screening colonoscopy: Code Z12.11 ONLY in ICD1 and leave ICD2, ICD3, ICD4 empty (nothing else)
   - Screening colonoscopy with polypectomy/polyp removal: Code Z12.11 in ICD1 and K63.5 in ICD2, leave ICD3 and ICD4 empty
5. CRITICAL VAGINAL DELIVERY CODING RULE:
   - If the procedure is a planned vaginal delivery labor (CPT code 01967): Use O80 ONLY in ICD1 and leave ICD2, ICD3, ICD4 empty (nothing else)
6. Identify up to 3 additional ICD codes (ICD2, ICD3, ICD4) sorted by relevance to the procedure
7. Only include ICD codes that are directly relevant to the procedure or patient condition
8. If fewer than 4 relevant ICD codes exist, leave the remaining fields empty
9. Use standard ICD-10 format (e.g., "E11.9", "I10", "Z87.891")
10. CRITICAL: Use web search to verify that all ICD codes you provide are valid and current as of November 2025. Only use the most recent ICD codes that are valid in November 2025. Do not use outdated or invalid codes.
11. CRITICAL: If there is outdated ICD codes listed on record try to find on google valid icd codes updated as of december 2025, so basically take the diagnosis code and update it accordingly with google
12. CRITICAL: Always make sure to not just pick the main diagnosis but to also look at secondary diagnoses further in the record IF available, they will often not be listed clearly as codes but instead as small snippets of text, there might be many of them listed like obesity and diabetes and such... make sure to convert those small texts to diagnosis codes, but also make sure to pick the ones that are MOST related to the main procedure and diagnosis itself, also make sure to use UPDATED december 2025 codes with google search
13. CRITICAL: Check for Excludes1 Conflicts: Before finalizing the JSON, verify if the selected codes have "Excludes1" notes in the ICD-10 manual that prevent them from being billed together.
14. Conflict Resolution: If two codes conflict (e.g., J35.1 and J35.01), prioritize the Post-Operative or more specific diagnosis.

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


def predict_codes_from_pdfs_api(pdf_folder, output_file, n_pages=1, model="openai/gpt-5.2:online", api_key=None, max_workers=3, progress_callback=None, custom_instructions=None, include_code_list=True, image_cache=None):
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
                    return idx, filename, "ERROR", "", 0, 0.0, error_msg, "openrouter_vision"
                
                # Predict ASA code from images
                predicted_code, explanation, tokens, cost, error = predict_asa_code_from_images(
                    image_data_list, cpt_codes_text, model, api_key, custom_instructions, include_code_list
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


def predict_icd_codes_from_pdfs_api(pdf_folder, output_file, n_pages=1, model="openai/gpt-5.2:online", api_key=None, max_workers=3, progress_callback=None, custom_instructions=None, image_cache=None):
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
                
                # Predict ICD codes from images
                icd_codes_dict, tokens, cost, error = predict_icd_codes_from_images(
                    image_data_list, model, api_key, custom_instructions
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
