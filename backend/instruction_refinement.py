#!/usr/bin/env python3
"""
Gemini-powered instruction refinement module.
Uses Gemini 3 Flash to analyze errors and improve CPT/ICD instruction templates.
"""

import os
import base64
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import google.genai as genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    genai = None
    types = None

logger = logging.getLogger(__name__)


def normalize_gemini_model(model_name: str) -> str:
    """Normalize Gemini model name by removing OpenRouter prefixes/suffixes."""
    clean_model = model_name.replace('google/', '').replace(':online', '')
    return clean_model


def load_pdf_as_images(pdf_path: str, max_pages: int = 2) -> List[str]:
    """
    Load PDF pages as base64-encoded PNG images.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to load
    
    Returns:
        List of base64-encoded image strings
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            images.append(img_base64)
        
        doc.close()
        return images
    except Exception as e:
        logger.error(f"Failed to load PDF as images: {e}")
        return []


def refine_cpt_instructions(
    current_instructions: str,
    error_cases: List[Dict[str, Any]],
    pdf_mapping: Dict[str, str],
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Use Gemini 3 Flash to refine CPT instructions based on error cases.
    
    Args:
        current_instructions: Current instruction text
        error_cases: List of error dictionaries with account_id, pdf_path, predicted, expected
        pdf_mapping: Dictionary mapping account_id to PDF file path
        model: Gemini model to use
        api_key: Google API key (optional, uses env var if not provided)
    
    Returns:
        Tuple of (improved_instructions, reasoning) or (None, error_message)
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return None, "Google GenAI SDK not available"
    
    model = normalize_gemini_model(model)
    
    # Get API key
    if api_key:
        api_key_value = api_key
    else:
        api_key_value = os.getenv("GOOGLE_API_KEY")
    
    if not api_key_value:
        return None, "No Google API key provided"
    
    # Initialize client
    try:
        client = genai.Client(api_key=api_key_value)
    except Exception as e:
        return None, f"Failed to initialize Google GenAI client: {str(e)}"
    
    # Select up to 5-10 diverse error cases with PDF context
    selected_errors = error_cases[:10]
    
    # Build error examples text
    error_examples = []
    pdf_images = []
    
    for idx, error in enumerate(selected_errors):
        account_id = error.get('account_id', 'N/A')
        pdf_path = pdf_mapping.get(account_id, '')
        predicted = error.get('predicted', 'N/A')
        expected = error.get('expected', 'N/A')
        
        error_examples.append(f"""
Error Case {idx + 1}:
- Account ID: {account_id}
- Predicted CPT: {predicted}
- Expected CPT: {expected}
- PDF: {Path(pdf_path).name if pdf_path else 'N/A'}
""")
        
        # Load PDF images for this error case (if available)
        if pdf_path and os.path.exists(pdf_path):
            images = load_pdf_as_images(pdf_path, max_pages=2)
            pdf_images.extend(images)
    
    error_examples_text = "\n".join(error_examples)
    
    # Build prompt
    prompt = f"""You are an expert medical coding instruction optimizer.

TASK:
Analyze the following errors and improve the CPT coding instructions to prevent these mistakes.

CURRENT INSTRUCTIONS:
{current_instructions}

ERROR ANALYSIS:
The AI made the following mistakes when predicting CPT codes:

{error_examples_text}

REQUIREMENTS:
1. Analyze the patterns in these errors - what common mistakes is the AI making?
2. Improve the instructions to prevent these specific errors
3. Make instructions GENERALIZABLE - focus on pattern recognition, not hardcoded rules
4. Use if-else logic ONLY when absolutely necessary for specific edge cases
5. Build upon the existing instructions - don't start from scratch
6. Keep instructions clear, concise, and actionable
7. Prioritize rules that will generalize to similar cases

OUTPUT FORMAT:
Respond with ONLY a JSON object in this exact format:
{{
  "improved_instructions": "Your improved instruction text here...",
  "reasoning": "Brief explanation of what changed and why (2-3 sentences)"
}}

The improved_instructions should be the complete, refined instruction text that replaces the current instructions.
The reasoning should explain the key changes made.

Respond with ONLY the JSON object, nothing else."""

    # Build content with images if available
    parts = [types.Part.from_text(text=prompt)]
    
    # Add PDF images (limit to avoid token limits)
    for img_data in pdf_images[:10]:  # Max 10 images
        try:
            img_bytes = base64.b64decode(img_data)
            parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
        except Exception as e:
            logger.warning(f"Failed to add image: {e}")
    
    contents = [types.Content(role="user", parts=parts)]
    
    # Configure thinking for Gemini 3 models
    if model in ["gemini-3-pro-preview", "gemini-3-flash-preview"]:
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
    
    # Enable web search
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        thinking_config=thinking_config,
        tools=tools
    )
    
    # Retry mechanism
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config
            )
            
            response_text = response.text.strip()
            
            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                improved_instructions = result.get("improved_instructions", "")
                reasoning = result.get("reasoning", "")
                
                if improved_instructions:
                    return improved_instructions, reasoning
                else:
                    return None, "Gemini did not return improved_instructions"
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                logger.warning(f"Response text: {response_text[:500]}")
                if attempt < max_retries - 1:
                    continue
                else:
                    # Try to extract instructions from plain text
                    if "improved_instructions" in response_text.lower():
                        # Fallback: return the response as-is
                        return response_text, "Parsed from plain text response"
                    return None, f"Failed to parse JSON response: {str(e)}"
        
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return None, f"Max retries reached: {str(e)}"
    
    return None, "Max retries reached"


def refine_icd_instructions(
    current_instructions: str,
    error_cases: List[Dict[str, Any]],
    pdf_mapping: Dict[str, str],
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Use Gemini 3 Flash to refine ICD instructions based on error cases.
    
    Args:
        current_instructions: Current instruction text
        error_cases: List of error dictionaries with account_id, pdf_path, predicted, expected
        pdf_mapping: Dictionary mapping account_id to PDF file path
        model: Gemini model to use
        api_key: Google API key (optional, uses env var if not provided)
    
    Returns:
        Tuple of (improved_instructions, reasoning) or (None, error_message)
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return None, "Google GenAI SDK not available"
    
    model = normalize_gemini_model(model)
    
    # Get API key
    if api_key:
        api_key_value = api_key
    else:
        api_key_value = os.getenv("GOOGLE_API_KEY")
    
    if not api_key_value:
        return None, "No Google API key provided"
    
    # Initialize client
    try:
        client = genai.Client(api_key=api_key_value)
    except Exception as e:
        return None, f"Failed to initialize Google GenAI client: {str(e)}"
    
    # Select up to 5-10 diverse error cases with PDF context
    selected_errors = error_cases[:10]
    
    # Build error examples text
    error_examples = []
    pdf_images = []
    
    for idx, error in enumerate(selected_errors):
        account_id = error.get('account_id', 'N/A')
        pdf_path = pdf_mapping.get(account_id, '')
        predicted = error.get('predicted', 'N/A')
        expected = error.get('expected', 'N/A')
        
        error_examples.append(f"""
Error Case {idx + 1}:
- Account ID: {account_id}
- Predicted ICD1: {predicted}
- Expected ICD1: {expected}
- PDF: {Path(pdf_path).name if pdf_path else 'N/A'}
""")
        
        # Load PDF images for this error case (if available)
        if pdf_path and os.path.exists(pdf_path):
            images = load_pdf_as_images(pdf_path, max_pages=2)
            pdf_images.extend(images)
    
    error_examples_text = "\n".join(error_examples)
    
    # Build prompt
    prompt = f"""You are an expert medical coding instruction optimizer.

TASK:
Analyze the following errors and improve the ICD coding instructions to prevent these mistakes.

CURRENT INSTRUCTIONS:
{current_instructions}

ERROR ANALYSIS:
The AI made the following mistakes when predicting ICD1 (primary diagnosis) codes:

{error_examples_text}

REQUIREMENTS:
1. Analyze the patterns in these errors - what common mistakes is the AI making?
2. Improve the instructions to prevent these specific errors
3. Make instructions GENERALIZABLE - focus on pattern recognition, not hardcoded rules
4. Use if-else logic ONLY when absolutely necessary for specific edge cases
5. Build upon the existing instructions - don't start from scratch
6. Keep instructions clear, concise, and actionable
7. Prioritize rules that will generalize to similar cases
8. Remember: ICD1 is the PRIMARY diagnosis - the main reason for the procedure

OUTPUT FORMAT:
Respond with ONLY a JSON object in this exact format:
{{
  "improved_instructions": "Your improved instruction text here...",
  "reasoning": "Brief explanation of what changed and why (2-3 sentences)"
}}

The improved_instructions should be the complete, refined instruction text that replaces the current instructions.
The reasoning should explain the key changes made.

Respond with ONLY the JSON object, nothing else."""

    # Build content with images if available
    parts = [types.Part.from_text(text=prompt)]
    
    # Add PDF images (limit to avoid token limits)
    for img_data in pdf_images[:10]:  # Max 10 images
        try:
            img_bytes = base64.b64decode(img_data)
            parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
        except Exception as e:
            logger.warning(f"Failed to add image: {e}")
    
    contents = [types.Content(role="user", parts=parts)]
    
    # Configure thinking for Gemini 3 models
    if model in ["gemini-3-pro-preview", "gemini-3-flash-preview"]:
        thinking_config = types.ThinkingConfig(thinking_level="HIGH")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
    
    # Enable web search
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        thinking_config=thinking_config,
        tools=tools
    )
    
    # Retry mechanism
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config
            )
            
            response_text = response.text.strip()
            
            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                improved_instructions = result.get("improved_instructions", "")
                reasoning = result.get("reasoning", "")
                
                if improved_instructions:
                    return improved_instructions, reasoning
                else:
                    return None, "Gemini did not return improved_instructions"
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                logger.warning(f"Response text: {response_text[:500]}")
                if attempt < max_retries - 1:
                    continue
                else:
                    # Try to extract instructions from plain text
                    if "improved_instructions" in response_text.lower():
                        # Fallback: return the response as-is
                        return response_text, "Parsed from plain text response"
                    return None, f"Failed to parse JSON response: {str(e)}"
        
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return None, f"Max retries reached: {str(e)}"
    
    return None, "Max retries reached"

