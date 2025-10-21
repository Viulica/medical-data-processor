#!/usr/bin/env python3
"""
Test LLM on ASA Code Prediction Task
"""

import pandas as pd
import json
from openai import OpenAI
import time
from collections import Counter
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Initialize OpenAI client
client = OpenAI(api_key="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")

def load_cpt_codes():
    """Load CPT codes from cpt_codes.txt"""
    print("📋 Loading CPT codes...")
    with open('cpt_codes.txt', 'r') as f:
        cpt_codes_text = f.read()
    print(f"✅ Loaded CPT codes reference")
    return cpt_codes_text


def load_test_data():
    """Load test dataset from CSV"""
    print("📊 Loading test data...")
    df = pd.read_csv('data.csv')
    
    # Filter records with ASA codes
    df = df[df['ASA Code'].notna() & (df['ASA Code'] != '')]
    print(f"✅ Loaded {len(df)} records with ASA codes")
    
    # Clean ASA codes (remove .0 if present)
    df['ASA Code'] = df['ASA Code'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    return df

def predict_asa_code(procedure, preop_diagnosis, postop_diagnosis, cpt_codes_text, model="gpt-4o"):
    """
    Predict ASA code using OpenAI API with web search
    
    Args:
        procedure: Procedure description
        preop_diagnosis: Pre-operative diagnosis
        postop_diagnosis: Post-operative diagnosis
        cpt_codes_text: Reference text containing all valid CPT codes
        model: Model to use (gpt-5, gpt-4o, etc.)
    
    Returns:
        predicted_code: Predicted ASA code
        raw_response: Raw response from the API
    """
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
        # The new API uses response.output_text
        predicted_code = response.output_text.strip()
        
        # Remove leading zeros from the predicted code
        predicted_code = predicted_code.lstrip('0') or '0'  # Handle case where code is all zeros
        return predicted_code, response
        
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        return None, None

def process_single_test(row, cpt_codes_text, model, log_queue, test_index):
    """
    Process a single test case - designed for threading
    """
    try:
        procedure = str(row['Procedure']) if pd.notna(row['Procedure']) else ""
        preop = str(row['Pre-op diagnosis']) if pd.notna(row['Pre-op diagnosis']) else ""
        postop = str(row['Post-op diagnosis']) if pd.notna(row['Post-op diagnosis']) else ""
        true_code = str(row['ASA Code'])
        
        log_queue.put(f"\n[{test_index}] Testing record...")
        log_queue.put(f"  Procedure: {procedure[:100]}...")
        log_queue.put(f"  True ASA Code: {true_code}")
        
        # Get prediction
        predicted_code, response = predict_asa_code(procedure, preop, postop, cpt_codes_text, model)
        
        if predicted_code is None:
            log_queue.put(f"  ❌ Error getting prediction")
            return {
                'index': test_index,
                'procedure': procedure,
                'preop': preop,
                'postop': postop,
                'true_code': true_code,
                'predicted_code': None,
                'correct': False,
                'error': True,
                'tokens': 0,
                'cost': 0
            }
        
        log_queue.put(f"  Predicted: {predicted_code}")
        
        # VERBOSE LOGGING for analysis
        log_queue.put(f"\n  📝 VERBOSE LOG:")
        log_queue.put(f"    INPUT PROCEDURE: '{procedure}'")
        log_queue.put(f"    INPUT PRE-OP: '{preop}'")
        log_queue.put(f"    INPUT POST-OP: '{postop}'")
        log_queue.put(f"    EXPECTED CODE: '{true_code}'")
        log_queue.put(f"    PREDICTED CODE: '{predicted_code}'")
        log_queue.put(f"    CORRECT: {predicted_code == true_code}")
        
        # Handle usage and cost calculation
        tokens = 0
        cost = 0
        if response and hasattr(response, 'usage'):
            usage = response.usage
            total_tokens = getattr(usage, 'total_tokens', 0)
            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            completion_tokens = getattr(usage, 'completion_tokens', 0)
            tokens = total_tokens
            
            log_queue.put(f"    TOKENS USED: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
            
            # Cost estimation
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
        
        # Handle response format
        if response:
            raw_response = response.output_text.strip()
        else:
            raw_response = 'N/A'
        log_queue.put(f"    RAW RESPONSE: '{raw_response}'")
        
        # Check if correct
        is_correct = predicted_code == true_code
        if is_correct:
            log_queue.put(f"  ✅ Correct!")
        else:
            log_queue.put(f"  ❌ Incorrect (Expected: {true_code})")
        
        return {
            'index': test_index,
            'procedure': procedure,
            'preop': preop,
            'postop': postop,
            'true_code': true_code,
            'predicted_code': predicted_code,
            'correct': is_correct,
            'error': False,
            'tokens': tokens,
            'cost': cost,
            'confusion_pair': (true_code, predicted_code) if not is_correct else None
        }
        
    except Exception as e:
        log_queue.put(f"  ❌ Exception in test {test_index}: {e}")
        return {
            'index': test_index,
            'procedure': str(row.get('Procedure', '')),
            'preop': str(row.get('Pre-op diagnosis', '')),
            'postop': str(row.get('Post-op diagnosis', '')),
            'true_code': str(row.get('ASA Code', '')),
            'predicted_code': None,
            'correct': False,
            'error': True,
            'tokens': 0,
            'cost': 0
        }

def test_llm_accuracy(df, cpt_codes_text, model="gpt-5-nano", max_samples=None, start_idx=0, max_workers=14):
    """
    Test LLM accuracy on the dataset using multi-threading
    
    Args:
        df: DataFrame with test data
        cpt_codes_text: Reference text containing all valid CPT codes
        model: Model to use
        max_samples: Maximum number of samples to test (None for all)
        start_idx: Starting index for testing (useful for resuming)
        max_workers: Number of concurrent threads (default: 5)
    
    Returns:
        results: Dictionary with results
    """
    # Create log file for continuous logging
    log_file = f'llm_test_log_{model}_{start_idx}.txt'
    
    def log_to_file(message):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        print(message)  # Also print to console
    
    log_to_file(f"\n🧪 Testing LLM: {model} (Multi-threaded with {max_workers} workers)")
    log_to_file("=" * 60)
    
    # Limit samples if specified
    if max_samples:
        df_test = df.iloc[start_idx:start_idx+max_samples]
    else:
        df_test = df.iloc[start_idx:]
    
    log_to_file(f"📊 Testing on {len(df_test)} samples (starting from index {start_idx})")
    
    results = {
        'correct': 0,
        'incorrect': 0,
        'errors': 0,
        'predictions': [],
        'confusion': Counter(),
        'total_tokens': 0,
        'total_cost': 0.0
    }
    
    # Create a queue for thread-safe logging
    log_queue = queue.Queue()
    
    # Prepare test data for threading
    test_data = []
    for idx, row in df_test.iterrows():
        test_data.append((row, idx + 1))  # idx + 1 for 1-based indexing
    
    log_to_file(f"🚀 Starting {len(test_data)} tests with {max_workers} concurrent workers...")
    
    # Process tests using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_data = {
            executor.submit(process_single_test, row, cpt_codes_text, model, log_queue, test_index): (row, test_index)
            for row, test_index in test_data
        }
        
        # Process completed tasks and log messages
        completed = 0
        for future in as_completed(future_to_data):
            # Process log messages from queue
            while not log_queue.empty():
                try:
                    message = log_queue.get_nowait()
                    log_to_file(message)
                except queue.Empty:
                    break
            
            # Get result
            try:
                result = future.result()
                completed += 1
                
                # Update results
                if result['error']:
                    results['errors'] += 1
                elif result['correct']:
                    results['correct'] += 1
                else:
                    results['incorrect'] += 1
                    if result['confusion_pair']:
                        results['confusion'][result['confusion_pair']] += 1
                
                # Update totals
                results['total_tokens'] += result['tokens']
                results['total_cost'] += result['cost']
                
                # Store prediction
                results['predictions'].append({
                    'index': result['index'],
                    'procedure': result['procedure'],
                    'preop': result['preop'],
                    'postop': result['postop'],
                    'true_code': result['true_code'],
                    'predicted_code': result['predicted_code'],
                    'correct': result['correct']
                })
                
                # Calculate running accuracy (excluding errors)
                valid_predictions = results['correct'] + results['incorrect']
                if valid_predictions > 0:
                    running_accuracy = (results['correct'] / valid_predictions) * 100
                    log_to_file(f"📈 Progress: {completed}/{len(test_data)} completed | Accuracy: {running_accuracy:.1f}% ({results['correct']}/{valid_predictions}) | Errors: {results['errors']}")
                else:
                    log_to_file(f"📈 Progress: {completed}/{len(test_data)} completed | Errors: {results['errors']}")
                
            except Exception as e:
                log_to_file(f"❌ Error processing result: {e}")
                results['errors'] += 1
    
    # Process any remaining log messages
    while not log_queue.empty():
        try:
            message = log_queue.get_nowait()
            log_to_file(message)
        except queue.Empty:
            break
    
    log_to_file(f"\n🏁 Testing completed! Log saved to: {log_file}")
    return results

def print_results(results):
    """Print test results"""
    total = results['correct'] + results['incorrect'] + results['errors']
    valid_predictions = results['correct'] + results['incorrect']
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    if valid_predictions > 0:
        accuracy = results['correct'] / valid_predictions
        print(f"🎯 Accuracy: {accuracy:.2%} ({results['correct']}/{valid_predictions})")
    else:
        print(f"❌ No valid predictions")
    
    print(f"✅ Correct predictions: {results['correct']}")
    print(f"❌ Incorrect predictions: {results['incorrect']}")
    print(f"⚠️  Errors: {results['errors']}")
    print(f"📝 Total samples tested: {total}")
    
    print(f"\n💰 Cost Statistics:")
    print(f"  Total tokens used: {results['total_tokens']:,}")
    print(f"  Estimated total cost: ${results['total_cost']:.4f}")
    
    if results['confusion']:
        print(f"\n🔍 Top 10 Confusion Pairs (True -> Predicted):")
        for (true_code, pred_code), count in results['confusion'].most_common(10):
            print(f"  {true_code} -> {pred_code}: {count} times")
    
    # Show some incorrect predictions
    incorrect_preds = [p for p in results['predictions'] if not p['correct']]
    if incorrect_preds:
        print(f"\n❌ Sample Incorrect Predictions:")
        for i, pred in enumerate(incorrect_preds[:5]):
            print(f"\n  {i+1}. Procedure: {pred['procedure'][:80]}...")
            print(f"     True: {pred['true_code']}, Predicted: {pred['predicted_code']}")
    
    return results

def save_results(results, model_name, output_file='llm_test_results.json'):
    """Save results to JSON file"""
    results_to_save = {
        'model': model_name,
        'accuracy': results['correct'] / (results['correct'] + results['incorrect']) if (results['correct'] + results['incorrect']) > 0 else 0,
        'correct': results['correct'],
        'incorrect': results['incorrect'],
        'errors': results['errors'],
        'total_tokens': results['total_tokens'],
        'total_cost': results['total_cost'],
        'confusion': [(k, v) for k, v in results['confusion'].most_common()],
        'predictions': results['predictions']
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

def save_verbose_logs(results, model_name, log_file='llm_verbose_logs.txt'):
    """Save detailed verbose logs for prompt analysis"""
    with open(log_file, 'w') as f:
        f.write(f"LLM TEST VERBOSE LOGS - {model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, pred in enumerate(results['predictions']):
            f.write(f"RECORD {i+1}:\n")
            f.write(f"  INPUT PROCEDURE: '{pred['procedure']}'\n")
            f.write(f"  INPUT PRE-OP: '{pred['preop']}'\n")
            f.write(f"  INPUT POST-OP: '{pred['postop']}'\n")
            f.write(f"  EXPECTED CODE: '{pred['true_code']}'\n")
            f.write(f"  PREDICTED CODE: '{pred['predicted_code']}'\n")
            f.write(f"  CORRECT: {pred['correct']}\n")
            f.write(f"  STATUS: {'✅ CORRECT' if pred['correct'] else '❌ INCORRECT'}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"📝 Verbose logs saved to: {log_file}")

if __name__ == "__main__":
    # Load CPT codes reference
    cpt_codes_text = load_cpt_codes()
    
    # Load data
    df = load_test_data()
    
    # Configuration
    MODEL = "gpt-5"  # Change to "gpt-4o", "gpt-4-turbo", etc. for different models
    MAX_SAMPLES = None  # Set to None to test on all data, or a number for limited testing
    START_IDX = 0  # Starting index (useful for resuming)
    MAX_WORKERS = 5  # Number of concurrent threads (adjust based on API rate limits)
    
    print(f"\n⚙️  Configuration:")
    print(f"  Model: {MODEL}")
    print(f"  Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"  Starting index: {START_IDX}")
    print(f"  Max workers: {MAX_WORKERS}")
    
    # Test the LLM
    results = test_llm_accuracy(df, cpt_codes_text, model=MODEL, max_samples=MAX_SAMPLES, start_idx=START_IDX, max_workers=MAX_WORKERS)
    
    # Print and save results
    print_results(results)
    save_results(results, MODEL)
    save_verbose_logs(results, MODEL)
    
    print("\n✅ Testing completed!")

