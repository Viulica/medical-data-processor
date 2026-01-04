#!/usr/bin/env python3
"""
Gemini-powered instruction refinement module.
Uses Gemini 3 Flash to analyze errors and improve CPT/ICD instruction templates.
"""

import os
import base64
import json
import logging
import requests
import time
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

# Default guidance prompt for AI refinement
DEFAULT_REFINEMENT_GUIDANCE = """Think general, make the ai make not the same mistake again, i need 100% coding accuracy, think hard about why a coder would have put that code and ai put another (read from pdf and dissect why) and understand yourself and try to find patterns

order is important so sometimes the ai will make mistakes of putting a code into ICD2 slot instead of ICD1, our job is to guide it such that taking the procedure type / diagnosises listed / other details listed 

the ai will know what to pick for PRIMARY ICD1 diagnosis

if you are the cpt coded prediction analyer, then ignore icd specific instructions"""


def is_gemini_model(model_name: str) -> bool:
    """Check if model name indicates a Gemini model (not OpenRouter format)"""
    clean_model = model_name.replace('google/', '').replace(':online', '').replace('models/', '')
    return clean_model.startswith('gemini') or 'gemini' in clean_model.lower()


def normalize_gemini_model(model_name: str) -> str:
    """Normalize Gemini model name by removing OpenRouter prefixes/suffixes."""
    clean_model = model_name.replace('google/', '').replace(':online', '').replace('models/', '')
    return clean_model


# Global PDF image cache
_pdf_image_cache: Dict[str, List[str]] = {}

def load_pdf_as_images(pdf_path: str, max_pages: Optional[int] = None, use_cache: bool = True) -> List[str]:
    """
    Load PDF pages as base64-encoded PNG images.
    Uses caching to avoid reloading the same PDF multiple times.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to load (None = all pages)
        use_cache: Whether to use cached images if available
    
    Returns:
        List of base64-encoded image strings
    """
    # Check cache first
    cache_key = f"{pdf_path}:{max_pages}"
    if use_cache and cache_key in _pdf_image_cache:
        logger.debug(f"Using cached images for {Path(pdf_path).name}")
        return _pdf_image_cache[cache_key]
    
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        images = []
        
        # Load all pages if max_pages is None, otherwise limit
        page_range = len(doc) if max_pages is None else min(len(doc), max_pages)
        
        for page_num in range(page_range):
            page = doc[page_num]
            # Use same DPI as standard prediction (150 DPI) for consistency
            mat = fitz.Matrix(150/72, 150/72)  # Scale factor for 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            images.append(img_base64)
        
        doc.close()
        
        # Cache the result
        if use_cache:
            _pdf_image_cache[cache_key] = images
        
        return images
    except Exception as e:
        logger.error(f"Failed to load PDF as images: {e}")
        return []


def preload_pdf_images(pdf_mapping: Dict[str, str], max_pages: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Pre-load all PDF images into cache for faster access during refinement.
    
    Args:
        pdf_mapping: Dictionary mapping account_id to PDF file path
        max_pages: Maximum number of pages to load per PDF (None = all pages)
    
    Returns:
        Dictionary mapping account_id to list of base64-encoded images
    """
    logger.info(f"Pre-loading PDF images for {len(pdf_mapping)} PDFs...")
    cached_images = {}
    
    for account_id, pdf_path in pdf_mapping.items():
        if pdf_path and os.path.exists(pdf_path):
            images = load_pdf_as_images(pdf_path, max_pages=max_pages, use_cache=True)
            cached_images[account_id] = images
            logger.debug(f"Pre-loaded {len(images)} pages from {Path(pdf_path).name}")
    
    logger.info(f"Pre-loaded images for {len(cached_images)} PDFs")
    return cached_images


def clear_pdf_image_cache():
    """Clear the PDF image cache to free memory."""
    global _pdf_image_cache
    _pdf_image_cache.clear()
    logger.info("PDF image cache cleared")


def refine_cpt_instructions(
    current_instructions: str,
    error_cases: List[Dict[str, Any]],
    pdf_mapping: Dict[str, str],
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    user_guidance: Optional[str] = None,
    pdf_image_cache: Optional[Dict[str, List[str]]] = None,
    instruction_history: Optional[List[Dict[str, Any]]] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Use AI model to refine CPT instructions based on error cases.
    
    Args:
        current_instructions: Current instruction text
        error_cases: List of error dictionaries with account_id, pdf_path, predicted, expected
        pdf_mapping: Dictionary mapping account_id to PDF file path
        model: Model to use (Gemini or OpenRouter format like "deepseek/deepseek-v3.2")
        api_key: API key (optional, uses env var if not provided)
        user_guidance: Optional user-provided guidance/prompt for the refinement agent
        pdf_image_cache: Optional pre-loaded PDF image cache for faster access
        instruction_history: Optional list of previous instruction attempts with accuracies.
                           Each dict should have: {"instructions": str, "accuracy": float, "iteration": int}
    
    Returns:
        Tuple of (improved_instructions, reasoning) or (None, error_message)
    """
    # Check if using Gemini or OpenRouter model
    use_openrouter = not is_gemini_model(model)
    
    if use_openrouter:
        # Use OpenRouter for non-Gemini models
        if api_key:
            api_key_value = api_key
        else:
            api_key_value = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not api_key_value:
            return None, "No OpenRouter API key provided"
    else:
        # Use Gemini
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
    
    # Select up to 10 diverse error cases with PDF context
    selected_errors = error_cases[:10]
    
    # Build prompt
    # Use default guidance if user_guidance is not provided
    effective_guidance = user_guidance if user_guidance and user_guidance.strip() else DEFAULT_REFINEMENT_GUIDANCE
    
    user_guidance_section = f"""

USER GUIDANCE:
The user has provided the following guidance for this refinement process. Please incorporate these instructions and priorities into your refinement:
{effective_guidance}

"""
    
    # Build instruction history section if provided
    history_section = ""
    if instruction_history and len(instruction_history) > 0:
        history_lines = ["\nPREVIOUS REFINEMENT ATTEMPTS (Learn from what was tried before):"]
        for hist_item in instruction_history:
            iteration = hist_item.get('iteration', '?')
            accuracy = hist_item.get('accuracy', None)
            instructions = hist_item.get('instructions', '')
            
            if accuracy is not None:
                history_lines.append(f"\n--- Iteration {iteration} (Accuracy: {accuracy:.2%}) ---")
            else:
                history_lines.append(f"\n--- Iteration {iteration} (Original) ---")
            
            # Truncate very long instructions for readability
            if len(instructions) > 2000:
                instructions_preview = instructions[:2000] + "\n... [truncated for length]"
            else:
                instructions_preview = instructions
            
            history_lines.append(instructions_preview)
        
        history_lines.append("\n\nIMPORTANT: Analyze what worked and what didn't in previous attempts. Avoid repeating approaches that didn't improve accuracy significantly. Build on what worked, and try different strategies for what didn't.")
        history_section = "\n".join(history_lines) + "\n"
    
    prompt = f"""You are an expert medical coding instruction optimizer.

TASK:
Analyze the following errors and improve the CPT coding instructions to prevent these mistakes.

{history_section}CURRENT INSTRUCTIONS (to improve):
{current_instructions}

{user_guidance_section}ERROR ANALYSIS WITH PDF CONTEXT:
Below are the error cases. Each error is followed immediately by its COMPLETE PDF document (all pages) for easy mapping.

IMPORTANT: Each error includes the model's reasoning for why it chose the predicted code. Use this reasoning to understand the model's thought process and identify where it went wrong. This will help you create better instructions that address the root cause of the mistake.

REQUIREMENTS:
1. Analyze the patterns in these errors - what common mistakes is the AI making?
2. Pay special attention to the model's reasoning - understand WHY it made the wrong choice
3. Improve the instructions to prevent these specific errors by addressing the flawed reasoning patterns
4. Make instructions GENERALIZABLE - focus on pattern recognition, not hardcoded rules
5. Use if-else logic ONLY when absolutely necessary for specific edge cases
6. Build upon the existing instructions - don't start from scratch
7. Keep instructions clear, concise, and actionable
8. Prioritize rules that will generalize to similar cases
{("9. Follow the user guidance provided above" if user_guidance_section else "")}

OUTPUT FORMAT:
Respond with ONLY a JSON object in this exact format:
{{
  "improved_instructions": "Your improved instruction text here...",
  "reasoning": "Brief explanation of what changed and why (2-3 sentences)"
}}

The improved_instructions should be the complete, refined instruction text that replaces the current instructions.
The reasoning should explain the key changes made.

Respond with ONLY the JSON object, nothing else."""

    # Build content with interleaved errors and PDFs for easy mapping
    # Load all PDF images first
    all_error_images = []
    for idx, error in enumerate(selected_errors):
        account_id = error.get('account_id', 'N/A')
        pdf_path = pdf_mapping.get(account_id, error.get('pdf_path', ''))
        
        # Try to get images from cache first
        images = None
        if pdf_image_cache and account_id in pdf_image_cache:
            images = pdf_image_cache[account_id]
            logger.debug(f"Using cached images for {Path(pdf_path).name} (error {idx + 1})")
        
        if not images and pdf_path and os.path.exists(pdf_path):
            images = load_pdf_as_images(pdf_path, max_pages=None, use_cache=True)  # Load all pages, cache for next time
            logger.info(f"Loaded {len(images)} pages from PDF: {Path(pdf_path).name} for error {idx + 1}")
            # Add to cache for next time
            if pdf_image_cache is not None:
                pdf_image_cache[account_id] = images
        
        all_error_images.append(images if images else [])
    
    # Build final instructions text
    final_instructions = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    
    if use_openrouter:
        # Build OpenRouter format content
        content = [{"type": "text", "text": prompt}]
        
        # Add each error with its PDF images
        for idx, error in enumerate(selected_errors):
            predicted = error.get('predicted', 'N/A')
            expected = error.get('expected', 'N/A')
            predicted_reasoning = error.get('predicted_reasoning', '')
            
            # Build error text with reasoning if available
            error_text = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nERROR {idx + 1}:\n- Predicted: '{predicted}'\n- Expected: '{expected}'"
            if predicted_reasoning:
                error_text += f"\n- Model's Reasoning for '{predicted}': {predicted_reasoning}"
            error_text += f"\n\nPDF DOCUMENT FOR ERROR {idx + 1}:\n"
            content.append({"type": "text", "text": error_text})
            
            # Add images in OpenRouter format
            images = all_error_images[idx]
            if images:
                for img_data in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_data}"
                        }
                    })
            else:
                content.append({"type": "text", "text": "(PDF not available for this error)\n"})
        
        content.append({"type": "text", "text": final_instructions})
        
        messages = [{"role": "user", "content": content}]
        
        # OpenRouter API call
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key_value}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Retry mechanism
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_text = response_data["choices"][0]["message"]["content"].strip()
                else:
                    return None, "OpenRouter did not return a valid response"
                
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
                        return None, "Model did not return improved_instructions"
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    logger.warning(f"Response text: {response_text[:500]}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        # Try to extract instructions from plain text
                        if "improved_instructions" in response_text.lower():
                            return response_text, "Parsed from plain text response"
                        return None, f"Failed to parse JSON response: {str(e)}"
            
            except Exception as e:
                logger.error(f"OpenRouter API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None, f"Max retries reached: {str(e)}"
        
        return None, "Max retries reached"
    else:
        # Build Gemini format content
        parts = [types.Part.from_text(text=prompt)]
        
        # Add each error with its PDF images immediately after
        for idx, error in enumerate(selected_errors):
            predicted = error.get('predicted', 'N/A')
            expected = error.get('expected', 'N/A')
            predicted_reasoning = error.get('predicted_reasoning', '')
            
            # Build error text with reasoning if available
            error_text = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nERROR {idx + 1}:\n- Predicted: '{predicted}'\n- Expected: '{expected}'"
            if predicted_reasoning:
                error_text += f"\n- Model's Reasoning for '{predicted}': {predicted_reasoning}"
            error_text += f"\n\nPDF DOCUMENT FOR ERROR {idx + 1}:\n"
            parts.append(types.Part.from_text(text=error_text))
            
            # Add images
            images = all_error_images[idx]
            if images:
                for img_data in images:
                    try:
                        img_bytes = base64.b64decode(img_data)
                        parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
                    except Exception as e:
                        logger.warning(f"Failed to add image: {e}")
            else:
                parts.append(types.Part.from_text(text="(PDF not available for this error)\n"))
        
        parts.append(types.Part.from_text(text=final_instructions))
        
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
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None, f"Max retries reached: {str(e)}"
        
        return None, "Max retries reached"


def refine_instructions_focused_mode(
    current_instructions: str,
    single_error_case: Dict[str, Any],
    pdf_path: str,
    instruction_type: str,  # "cpt" or "icd"
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    user_guidance: Optional[str] = None
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Focused refinement mode: Process ONE error at a time with deep analysis.
    AI intelligently decides whether to add general rules or hardcoded rules.
    
    Args:
        current_instructions: Current instruction text
        single_error_case: Single error dictionary with account_id, predicted, expected
        pdf_path: Path to the PDF for this specific error
        instruction_type: "cpt" or "icd"
        model: Model to use (Gemini or OpenRouter format like "deepseek/deepseek-v3.2")
        api_key: API key (optional, uses env var if not provided)
        user_guidance: Optional user-provided guidance/prompt for the refinement agent
    
    Returns:
        Tuple of (improved_instructions, reasoning, rule_type) where rule_type is "general" or "hardcoded"
    """
    # Check if using Gemini or OpenRouter model
    use_openrouter = not is_gemini_model(model)
    
    if use_openrouter:
        # Use OpenRouter for non-Gemini models
        if api_key:
            api_key_value = api_key
        else:
            api_key_value = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not api_key_value:
            return None, "No OpenRouter API key provided", None
    else:
        # Use Gemini
        if not GOOGLE_GENAI_AVAILABLE:
            return None, "Google GenAI SDK not available", None
        
        model = normalize_gemini_model(model)
        
        # Get API key
        if api_key:
            api_key_value = api_key
        else:
            api_key_value = os.getenv("GOOGLE_API_KEY")
        
        if not api_key_value:
            return None, "No Google API key provided", None
        
        # Initialize client
        try:
            client = genai.Client(api_key=api_key_value)
        except Exception as e:
            return None, f"Failed to initialize Google GenAI client: {str(e)}", None
    
    # Load ONLY this PDF (all pages for context) - use cache if available
    pdf_images = []
    if pdf_path and os.path.exists(pdf_path):
        images = load_pdf_as_images(pdf_path, max_pages=None, use_cache=True)  # Load all pages, cache for next time
        pdf_images.extend(images)
        logger.info(f"Loaded {len(images)} pages from PDF: {Path(pdf_path).name}")
    else:
        logger.warning(f"PDF not found: {pdf_path}")
    
    # Build error description (no Account ID - not relevant for refinement)
    predicted = single_error_case.get('predicted', '')
    expected = single_error_case.get('expected', '')
    error_type = single_error_case.get('error_type', instruction_type.upper())
    
    # Extract reasoning if available
    predicted_reasoning = single_error_case.get('predicted_reasoning', '')
    
    # For ICD, also get individual code reasoning
    predicted_icd1_reasoning = single_error_case.get('predicted_icd1_reasoning', '')
    predicted_icd2_reasoning = single_error_case.get('predicted_icd2_reasoning', '')
    predicted_icd3_reasoning = single_error_case.get('predicted_icd3_reasoning', '')
    predicted_icd4_reasoning = single_error_case.get('predicted_icd4_reasoning', '')
    
    error_description = f"""
ERROR CASE:
- Error Type: {error_type}
- Predicted: {predicted}
- Expected: {expected}
- The AI predicted "{predicted}" but the correct answer is "{expected}"
"""
    
    # Add reasoning to error description
    if predicted_reasoning:
        error_description += f"\n- Model's Reasoning for '{predicted}': {predicted_reasoning}"
    
    # For ICD errors, add reasoning for each code if available
    if error_type == 'ICD1' and predicted_icd1_reasoning:
        error_description += f"\n- Model's Reasoning for ICD1 '{predicted}': {predicted_icd1_reasoning}"
    if predicted_icd2_reasoning:
        predicted_icd2 = single_error_case.get('predicted_icd2', '')
        if predicted_icd2:
            error_description += f"\n- Model's Reasoning for ICD2 '{predicted_icd2}': {predicted_icd2_reasoning}"
    if predicted_icd3_reasoning:
        predicted_icd3 = single_error_case.get('predicted_icd3', '')
        if predicted_icd3:
            error_description += f"\n- Model's Reasoning for ICD3 '{predicted_icd3}': {predicted_icd3_reasoning}"
    if predicted_icd4_reasoning:
        predicted_icd4 = single_error_case.get('predicted_icd4', '')
        if predicted_icd4:
            error_description += f"\n- Model's Reasoning for ICD4 '{predicted_icd4}': {predicted_icd4_reasoning}"
    
    # Use default guidance if not provided
    effective_guidance = user_guidance if user_guidance and user_guidance.strip() else DEFAULT_REFINEMENT_GUIDANCE
    
    # Build intelligent prompt that distinguishes between general vs hardcoded rules
    prompt = f"""You are an expert medical coding instruction optimizer working in FOCUSED MODE.

TASK:
Analyze this SINGLE error case deeply and improve the {instruction_type.upper()} coding instructions to prevent this specific mistake.

CURRENT INSTRUCTIONS:
{current_instructions}

{error_description}

PDF CONTEXT:
Below is the COMPLETE PDF document (all pages) for this error case.
- Carefully examine the PDF to understand WHY the coder chose "{expected}" instead of "{predicted}"
- Look for patterns, context clues, procedure types, diagnoses, and other details that would guide the correct coding decision
- Pay attention to the model's reasoning (if provided) - understand WHY it made the wrong choice and what flawed logic led to the mistake

USER GUIDANCE:
{effective_guidance}

CRITICAL DECISION: GENERAL vs HARDCODED RULES

You must intelligently decide whether this error requires:
1. GENERAL RULE: A pattern-based rule that applies to many similar cases
   - Use when: The error reveals a systematic misunderstanding (e.g., "always check procedure type first", "primary diagnosis should be most severe")
   - These rules help prevent similar mistakes across many cases
   - Example for CPT: "When multiple procedures are listed, prioritize the most invasive/complex procedure for anesthesia CPT coding"
   - Example for ICD: "When multiple diagnoses are present, the primary diagnosis should be the most severe or the one most directly related to the procedure"
   
2. HARDCODED RULE: A specific if-then rule for this exact scenario
   - Use when: This is a rare edge case or specific exception that needs explicit handling
   - These rules are very specific and may only apply to this exact situation
   - Example for CPT: "If procedure is 'knee arthroscopy' AND patient has ASA status 3, use anesthesia CPT code 01382"
   - Example for ICD: "If procedure is 'laparoscopic cholecystectomy' AND patient has history of cholelithiasis, use ICD code K80.20"
   - WARNING: Only use hardcoded rules when absolutely necessary - they can cause overfitting

ANALYSIS PROCESS:
1. Read the PDF carefully and identify WHY "{expected}" is correct and "{predicted}" is wrong
2. Determine if this is a pattern (general rule) or exception (hardcoded rule)
3. If pattern: Add a generalizable rule that will help with similar cases
4. If exception: Add a specific if-then rule ONLY if necessary
5. Build upon existing instructions - don't start from scratch
6. Keep instructions clear and actionable

OUTPUT FORMAT:
Respond with ONLY a JSON object in this exact format:
{{
  "improved_instructions": "Your improved instruction text here...",
  "reasoning": "Brief explanation of what changed and why (2-3 sentences)",
  "rule_type": "general" or "hardcoded",
  "rule_explanation": "Why you chose general vs hardcoded (1-2 sentences)"
}}

The improved_instructions should be the complete, refined instruction text that replaces the current instructions.
The reasoning should explain the key changes made.
The rule_type should be either "general" or "hardcoded" based on your analysis.
The rule_explanation should justify your choice.

Respond with ONLY the JSON object, nothing else."""

    # Build content with images
    if use_openrouter:
        # Build OpenRouter format content
        content = [{"type": "text", "text": prompt}]
        
        # Add images in OpenRouter format
        logger.info(f"Adding {len(pdf_images)} PDF page images for focused refinement")
        for img_data in pdf_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_data}"
                }
            })
        
        messages = [{"role": "user", "content": content}]
        
        # OpenRouter API call
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key_value}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Retry mechanism
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_text = response_data["choices"][0]["message"]["content"].strip()
                else:
                    return None, "OpenRouter did not return a valid response", None
                
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
                    rule_type = result.get("rule_type", "general")
                    rule_explanation = result.get("rule_explanation", "")
                    
                    if improved_instructions:
                        full_reasoning = f"{reasoning} [Rule Type: {rule_type}] {rule_explanation}"
                        return improved_instructions, full_reasoning, rule_type
                    else:
                        return None, "Model did not return improved_instructions", None
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    logger.warning(f"Response text: {response_text[:500]}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return None, f"Failed to parse JSON response: {str(e)}", None
            
            except Exception as e:
                logger.error(f"OpenRouter API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None, f"Max retries reached: {str(e)}", None
        
        return None, "Max retries reached", None
    else:
        # Build Gemini format content
        parts = [types.Part.from_text(text=prompt)]
        
        # Add ONLY this PDF's images
        logger.info(f"Adding {len(pdf_images)} PDF page images for focused refinement")
        for img_data in pdf_images:
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
                    rule_type = result.get("rule_type", "general")
                    rule_explanation = result.get("rule_explanation", "")
                    
                    if improved_instructions:
                        full_reasoning = f"{reasoning} [Rule Type: {rule_type}] {rule_explanation}"
                        return improved_instructions, full_reasoning, rule_type
                    else:
                        return None, "Gemini did not return improved_instructions", None
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    logger.warning(f"Response text: {response_text[:500]}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None, f"Failed to parse JSON response: {str(e)}", None
            
            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None, f"Max retries reached: {str(e)}", None
        
        return None, "Max retries reached", None


def refine_icd_instructions(
    current_instructions: str,
    error_cases: List[Dict[str, Any]],
    pdf_mapping: Dict[str, str],
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    user_guidance: Optional[str] = None,
    pdf_image_cache: Optional[Dict[str, List[str]]] = None,
    instruction_history: Optional[List[Dict[str, Any]]] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Use AI model to refine ICD instructions based on error cases.
    
    Args:
        current_instructions: Current instruction text
        error_cases: List of error dictionaries with account_id, pdf_path, predicted, expected
        pdf_mapping: Dictionary mapping account_id to PDF file path
        model: Model to use (Gemini or OpenRouter format like "deepseek/deepseek-v3.2")
        api_key: API key (optional, uses env var if not provided)
        user_guidance: Optional user-provided guidance/prompt for the refinement agent
        pdf_image_cache: Optional pre-loaded PDF image cache for faster access
        instruction_history: Optional list of previous instruction attempts with accuracies.
                           Each dict should have: {"instructions": str, "accuracy": float, "iteration": int}
    
    Returns:
        Tuple of (improved_instructions, reasoning) or (None, error_message)
    """
    # Check if using Gemini or OpenRouter model
    use_openrouter = not is_gemini_model(model)
    
    if use_openrouter:
        # Use OpenRouter for non-Gemini models
        if api_key:
            api_key_value = api_key
        else:
            api_key_value = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not api_key_value:
            return None, "No OpenRouter API key provided"
    else:
        # Use Gemini
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
    
    # Select up to 10 diverse error cases with PDF context
    selected_errors = error_cases[:10]
    
    # Build error examples text
    error_examples = []
    pdf_images = []
    
    for idx, error in enumerate(selected_errors):
        account_id = error.get('account_id', 'N/A')
        pdf_path = pdf_mapping.get(account_id, '')
        predicted_icd1 = error.get('predicted_icd1', error.get('predicted', 'N/A'))
        predicted_icd2 = error.get('predicted_icd2', '')
        predicted_icd3 = error.get('predicted_icd3', '')
        predicted_icd4 = error.get('predicted_icd4', '')
        expected_icd1 = error.get('expected_icd1', error.get('expected', 'N/A'))
        expected_icd2 = error.get('expected_icd2', '')
        expected_icd3 = error.get('expected_icd3', '')
        expected_icd4 = error.get('expected_icd4', '')
        
        # Build predicted ICD codes string
        predicted_icds = [predicted_icd1, predicted_icd2, predicted_icd3, predicted_icd4]
        predicted_icds = [icd for icd in predicted_icds if icd]  # Remove empty
        predicted_str = ', '.join(predicted_icds) if predicted_icds else '(none)'
        
        # Build expected ICD codes string
        expected_icds = [expected_icd1, expected_icd2, expected_icd3, expected_icd4]
        expected_icds = [icd for icd in expected_icds if icd]  # Remove empty
        expected_str = ', '.join(expected_icds) if expected_icds else '(none)'
        
        error_examples.append(f"""
Error Case {idx + 1}:
- Account ID: {account_id}
- Predicted ICD codes: {predicted_str}
  (ICD1: {predicted_icd1}, ICD2: {predicted_icd2 or '(empty)'}, ICD3: {predicted_icd3 or '(empty)'}, ICD4: {predicted_icd4 or '(empty)'})
- Expected ICD codes: {expected_str}
  (ICD1: {expected_icd1}, ICD2: {expected_icd2 or '(empty)'}, ICD3: {expected_icd3 or '(empty)'}, ICD4: {expected_icd4 or '(empty)'})
- PDF: {Path(pdf_path).name if pdf_path else 'N/A'}
""")
        
        # Load COMPLETE PDF images for this error case (all pages)
        if pdf_path and os.path.exists(pdf_path):
            images = load_pdf_as_images(pdf_path, max_pages=None)  # Load all pages
            pdf_images.extend(images)
            logger.info(f"Loaded {len(images)} pages from PDF: {Path(pdf_path).name}")
    
    error_examples_text = "\n".join(error_examples)
    
    # Build prompt
    # Use default guidance if user_guidance is not provided
    effective_guidance = user_guidance if user_guidance and user_guidance.strip() else DEFAULT_REFINEMENT_GUIDANCE
    
    user_guidance_section = f"""

USER GUIDANCE:
The user has provided the following guidance for this refinement process. Please incorporate these instructions and priorities into your refinement:
{effective_guidance}

"""
    
    # Build instruction history section if provided
    history_section = ""
    if instruction_history and len(instruction_history) > 0:
        history_lines = ["\nPREVIOUS REFINEMENT ATTEMPTS (Learn from what was tried before):"]
        for hist_item in instruction_history:
            iteration = hist_item.get('iteration', '?')
            accuracy = hist_item.get('accuracy', None)
            instructions = hist_item.get('instructions', '')
            
            if accuracy is not None:
                history_lines.append(f"\n--- Iteration {iteration} (Accuracy: {accuracy:.2%}) ---")
            else:
                history_lines.append(f"\n--- Iteration {iteration} (Original) ---")
            
            # Truncate very long instructions for readability
            if len(instructions) > 2000:
                instructions_preview = instructions[:2000] + "\n... [truncated for length]"
            else:
                instructions_preview = instructions
            
            history_lines.append(instructions_preview)
        
        history_lines.append("\n\nIMPORTANT: Analyze what worked and what didn't in previous attempts. Avoid repeating approaches that didn't improve accuracy significantly. Build on what worked, and try different strategies for what didn't.")
        history_section = "\n".join(history_lines) + "\n"
    
    prompt = f"""You are an expert medical coding instruction optimizer.

TASK:
Analyze the following errors and improve the ICD coding instructions to prevent these mistakes.

{history_section}CURRENT INSTRUCTIONS (to improve):
{current_instructions}

{user_guidance_section}ERROR ANALYSIS WITH PDF CONTEXT:
Below are the error cases. Each error is followed immediately by its COMPLETE PDF document (all pages) for easy mapping.

IMPORTANT: Each error includes the model's reasoning for why it chose each predicted ICD code. Use this reasoning to understand the model's thought process and identify where it went wrong. This will help you create better instructions that address the root cause of the mistake.

CRITICAL FOCUS - ICD1 IS PRIMARY:
- ICD1 (PRIMARY diagnosis) is THE MOST IMPORTANT code for medical billing - this is what determines payment
- Accuracy is ONLY tested on ICD1 - this is the ONLY code that matters for success metrics
- ICD2, ICD3, and ICD4 are provided ONLY for context - they are less important and NOT tested
- Your PRIMARY goal is to fix ICD1 prediction errors - focus ALL improvements on getting ICD1 correct
- The secondary codes (ICD2-4) may help you understand context, but DO NOT prioritize fixing them
- If ICD1 is wrong but ICD2-4 are correct, that's still a FAILURE - focus on fixing ICD1
- Pay special attention to the model's reasoning for ICD1 - understand WHY it chose the wrong primary diagnosis
"""

    # Build content with interleaved errors and PDFs for easy mapping
    # Load all PDF images first
    all_error_images = []
    for idx, error in enumerate(selected_errors):
        account_id = error.get('account_id', 'N/A')
        pdf_path = pdf_mapping.get(account_id, error.get('pdf_path', ''))
        
        # Try to get images from cache first
        images = None
        if pdf_image_cache and account_id in pdf_image_cache:
            images = pdf_image_cache[account_id]
            logger.debug(f"Using cached images for {Path(pdf_path).name} (error {idx + 1})")
        
        if not images and pdf_path and os.path.exists(pdf_path):
            images = load_pdf_as_images(pdf_path, max_pages=None, use_cache=True)  # Load all pages, cache for next time
            logger.info(f"Loaded {len(images)} pages from PDF: {Path(pdf_path).name} for error {idx + 1}")
            # Add to cache for next time
            if pdf_image_cache is not None:
                pdf_image_cache[account_id] = images
        
        all_error_images.append(images if images else [])
    
    # Build final instructions text
    final_instructions = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIREMENTS:
1. FOCUS ON ICD1: Analyze patterns in ICD1 errors specifically - what mistakes is the AI making with PRIMARY diagnosis?
2. Improve instructions to prevent ICD1 errors - this is your PRIMARY objective
3. Use ICD2-4 only as context clues - they may help understand why ICD1 was wrong, but don't optimize for them
4. Make instructions GENERALIZABLE - focus on pattern recognition for PRIMARY diagnosis, not hardcoded rules
5. Use if-else logic ONLY when absolutely necessary for specific edge cases
6. Build upon the existing instructions - don't start from scratch
7. Keep instructions clear, concise, and actionable
8. Prioritize rules that will generalize to similar cases and improve ICD1 accuracy
9. Remember: For anesthesia medical billing, ICD1 is CRITICAL - getting it wrong means NO PAYMENT

OUTPUT FORMAT:
Respond with ONLY a JSON object in this exact format:
{{
  "improved_instructions": "Your improved instruction text here...",
  "reasoning": "Brief explanation of what changed and why (2-3 sentences)"
}}

The improved_instructions should be the complete, refined instruction text that replaces the current instructions.
The reasoning should explain the key changes made.

Respond with ONLY the JSON object, nothing else."""
    
    if use_openrouter:
        # Build OpenRouter format content
        content = [{"type": "text", "text": prompt}]
        
        # Add each error with its PDF images
        for idx, error in enumerate(selected_errors):
            predicted_icd1 = error.get('predicted_icd1', error.get('predicted', 'N/A'))
            predicted_icd2 = error.get('predicted_icd2', '')
            predicted_icd3 = error.get('predicted_icd3', '')
            predicted_icd4 = error.get('predicted_icd4', '')
            expected_icd1 = error.get('expected_icd1', error.get('expected', 'N/A'))
            expected_icd2 = error.get('expected_icd2', '')
            expected_icd3 = error.get('expected_icd3', '')
            expected_icd4 = error.get('expected_icd4', '')
            
            # Build predicted ICD codes string
            predicted_icds = [predicted_icd1, predicted_icd2, predicted_icd3, predicted_icd4]
            predicted_icds = [icd for icd in predicted_icds if icd]  # Remove empty
            predicted_str = ', '.join(predicted_icds) if predicted_icds else '(none)'
            
            # Build expected ICD codes string
            expected_icds = [expected_icd1, expected_icd2, expected_icd3, expected_icd4]
            expected_icds = [icd for icd in expected_icds if icd]  # Remove empty
            expected_str = ', '.join(expected_icds) if expected_icds else '(none)'
            
            # Extract reasoning for each ICD code if available
            predicted_icd1_reasoning = error.get('predicted_icd1_reasoning', '')
            predicted_icd2_reasoning = error.get('predicted_icd2_reasoning', '')
            predicted_icd3_reasoning = error.get('predicted_icd3_reasoning', '')
            predicted_icd4_reasoning = error.get('predicted_icd4_reasoning', '')
            
            error_text = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nERROR {idx + 1}:\n- Predicted ICD codes: {predicted_str}\n  (ICD1: {predicted_icd1}, ICD2: {predicted_icd2 or '(empty)'}, ICD3: {predicted_icd3 or '(empty)'}, ICD4: {predicted_icd4 or '(empty)'})\n- Expected ICD codes: {expected_str}\n  (ICD1: {expected_icd1}, ICD2: {expected_icd2 or '(empty)'}, ICD3: {expected_icd3 or '(empty)'}, ICD4: {expected_icd4 or '(empty)'})"
            
            # Add reasoning if available
            if predicted_icd1_reasoning:
                error_text += f"\n- Model's Reasoning for ICD1 '{predicted_icd1}': {predicted_icd1_reasoning}"
            if predicted_icd2 and predicted_icd2_reasoning:
                error_text += f"\n- Model's Reasoning for ICD2 '{predicted_icd2}': {predicted_icd2_reasoning}"
            if predicted_icd3 and predicted_icd3_reasoning:
                error_text += f"\n- Model's Reasoning for ICD3 '{predicted_icd3}': {predicted_icd3_reasoning}"
            if predicted_icd4 and predicted_icd4_reasoning:
                error_text += f"\n- Model's Reasoning for ICD4 '{predicted_icd4}': {predicted_icd4_reasoning}"
            
            error_text += f"\n\nPDF DOCUMENT FOR ERROR {idx + 1}:\n"
            content.append({"type": "text", "text": error_text})
            
            # Add images in OpenRouter format
            images = all_error_images[idx]
            if images:
                for img_data in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_data}"
                        }
                    })
            else:
                content.append({"type": "text", "text": "(PDF not available for this error)\n"})
        
        content.append({"type": "text", "text": final_instructions})
        
        messages = [{"role": "user", "content": content}]
        
        # OpenRouter API call
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key_value}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Retry mechanism
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_text = response_data["choices"][0]["message"]["content"].strip()
                else:
                    return None, "OpenRouter did not return a valid response"
                
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
                        return None, "Model did not return improved_instructions"
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                    logger.warning(f"Response text: {response_text[:500]}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        # Try to extract instructions from plain text
                        if "improved_instructions" in response_text.lower():
                            return response_text, "Parsed from plain text response"
                        return None, f"Failed to parse JSON response: {str(e)}"
            
            except Exception as e:
                logger.error(f"OpenRouter API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None, f"Max retries reached: {str(e)}"
        
        return None, "Max retries reached"
    else:
        # Build Gemini format content
        parts = [types.Part.from_text(text=prompt)]
        
        # Add each error with its PDF images immediately after
        for idx, error in enumerate(selected_errors):
            predicted_icd1 = error.get('predicted_icd1', error.get('predicted', 'N/A'))
            predicted_icd2 = error.get('predicted_icd2', '')
            predicted_icd3 = error.get('predicted_icd3', '')
            predicted_icd4 = error.get('predicted_icd4', '')
            expected_icd1 = error.get('expected_icd1', error.get('expected', 'N/A'))
            expected_icd2 = error.get('expected_icd2', '')
            expected_icd3 = error.get('expected_icd3', '')
            expected_icd4 = error.get('expected_icd4', '')
            
            # Build predicted ICD codes string
            predicted_icds = [predicted_icd1, predicted_icd2, predicted_icd3, predicted_icd4]
            predicted_icds = [icd for icd in predicted_icds if icd]  # Remove empty
            predicted_str = ', '.join(predicted_icds) if predicted_icds else '(none)'
            
            # Build expected ICD codes string
            expected_icds = [expected_icd1, expected_icd2, expected_icd3, expected_icd4]
            expected_icds = [icd for icd in expected_icds if icd]  # Remove empty
            expected_str = ', '.join(expected_icds) if expected_icds else '(none)'
            
            # Add error text
            # Extract reasoning for each ICD code if available
            predicted_icd1_reasoning = error.get('predicted_icd1_reasoning', '')
            predicted_icd2_reasoning = error.get('predicted_icd2_reasoning', '')
            predicted_icd3_reasoning = error.get('predicted_icd3_reasoning', '')
            predicted_icd4_reasoning = error.get('predicted_icd4_reasoning', '')
            
            error_text = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nERROR {idx + 1}:\n- Predicted ICD codes: {predicted_str}\n  (ICD1: {predicted_icd1}, ICD2: {predicted_icd2 or '(empty)'}, ICD3: {predicted_icd3 or '(empty)'}, ICD4: {predicted_icd4 or '(empty)'})\n- Expected ICD codes: {expected_str}\n  (ICD1: {expected_icd1}, ICD2: {expected_icd2 or '(empty)'}, ICD3: {expected_icd3 or '(empty)'}, ICD4: {expected_icd4 or '(empty)'})"
            
            # Add reasoning if available
            if predicted_icd1_reasoning:
                error_text += f"\n- Model's Reasoning for ICD1 '{predicted_icd1}': {predicted_icd1_reasoning}"
            if predicted_icd2 and predicted_icd2_reasoning:
                error_text += f"\n- Model's Reasoning for ICD2 '{predicted_icd2}': {predicted_icd2_reasoning}"
            if predicted_icd3 and predicted_icd3_reasoning:
                error_text += f"\n- Model's Reasoning for ICD3 '{predicted_icd3}': {predicted_icd3_reasoning}"
            if predicted_icd4 and predicted_icd4_reasoning:
                error_text += f"\n- Model's Reasoning for ICD4 '{predicted_icd4}': {predicted_icd4_reasoning}"
            
            error_text += f"\n\nPDF DOCUMENT FOR ERROR {idx + 1}:\n"
            parts.append(types.Part.from_text(text=error_text))
            
            # Add images
            images = all_error_images[idx]
            if images:
                for img_data in images:
                    try:
                        img_bytes = base64.b64decode(img_data)
                        parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
                    except Exception as e:
                        logger.warning(f"Failed to add image: {e}")
            else:
                parts.append(types.Part.from_text(text="(PDF not available for this error)\n"))
        
        parts.append(types.Part.from_text(text=final_instructions))
        
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
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None, f"Max retries reached: {str(e)}"
        
        return None, "Max retries reached"

