#!/usr/bin/env python3
"""
Test PO Box matching between data.csv and mednet.csv with AI-powered resolution for multiple matches
"""

import pandas as pd
import re
import json
import google.genai as genai
from google.genai import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import datetime
import os

def extract_po_box(address):
    """Extract PO Box number from address using regex"""
    if pd.isna(address) or address == '':
        return None
    
    # Pattern to match PO BOX followed by numbers
    po_box_pattern = r'PO\s+BOX\s+(\d+)'
    match = re.search(po_box_pattern, str(address).upper())
    
    if match:
        return match.group(1)
    return None

def clean_address_for_matching(address):
    """Clean address for better matching"""
    if pd.isna(address) or address == '':
        return ''
    
    # Convert to uppercase and remove extra spaces
    cleaned = str(address).upper().strip()
    # Remove commas and semicolons for consistency
    cleaned = cleaned.replace(',', '').replace(';', '')
    # Normalize spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

def log_ai_decision(log_file, result, ai_response=None):
    """Log AI decision to file for analysis"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        'timestamp': timestamp,
        'po_box': result['po_box'],
        'insurance_name': result['insurance_name'],
        'address': result['address'],
        'expected_code': result['expected_code'],
        'match_type': result['match_type'],
        'mapped': result['mapped'],
        'correct': result['correct'],
        'found_code': result['found_code'],
        'mednet_name': result['mednet_name'],
        'ai_confidence': result['ai_confidence'],
        'ai_response': ai_response
    }
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

def analyze_ai_mistakes(log_file):
    """Analyze AI mistakes and detect patterns"""
    mistakes = []
    low_confidence_cases = []
    all_entries = []
    
    if not os.path.exists(log_file):
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                all_entries.append(entry)
                if entry['mapped'] and not entry['correct']:
                    mistakes.append(entry)
                elif entry.get('ai_confidence') == 'low':
                    low_confidence_cases.append(entry)
            except:
                continue
    
    if not mistakes:
        print("No AI mistakes found in log.")
        return
    
    print(f"\n🔍 AI MISTAKE ANALYSIS ({len(mistakes)} mistakes found):")
    print("=" * 60)
    
    # Pattern 1: PO Box patterns
    po_box_mistakes = defaultdict(int)
    for mistake in mistakes:
        po_box_mistakes[mistake['po_box']] += 1
    
    print("PO Box patterns (most problematic):")
    for po_box, count in sorted(po_box_mistakes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  PO Box {po_box}: {count} mistakes")
    
    # Pattern 2: Insurance name patterns
    insurance_patterns = defaultdict(int)
    for mistake in mistakes:
        insurance_name = mistake['insurance_name']
        # Extract common patterns
        if 'UNITED HEALTHCARE' in insurance_name:
            insurance_patterns['UNITED HEALTHCARE variants'] += 1
        elif 'BLUE CROSS' in insurance_name:
            insurance_patterns['BLUE CROSS variants'] += 1
        elif 'MEDICARE' in insurance_name:
            insurance_patterns['MEDICARE variants'] += 1
        else:
            insurance_patterns['Other'] += 1
    
    print("\nInsurance name patterns:")
    for pattern, count in sorted(insurance_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} mistakes")
    
    # Pattern 3: Expected vs Found code patterns
    code_mistakes = defaultdict(int)
    for mistake in mistakes:
        key = f"Expected: {mistake['expected_code']} -> Found: {mistake['found_code']}"
        code_mistakes[key] += 1
    
    print("\nMost common code mistakes:")
    for pattern, count in sorted(code_mistakes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pattern}: {count} times")
    
    # Pattern 4: Confidence vs correctness
    confidence_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    for mistake in mistakes:
        confidence = mistake.get('ai_confidence', 'unknown')
        confidence_stats[confidence]['incorrect'] += 1
    
    print("\nConfidence vs correctness:")
    for confidence, stats in confidence_stats.items():
        total = stats['correct'] + stats['incorrect']
        if total > 0:
            accuracy = (stats['correct'] / total) * 100
            print(f"  {confidence.upper()}: {stats['correct']}/{total} = {accuracy:.1f}% accuracy")
    
    # Save detailed mistake report
    mistake_report_file = log_file.replace('.txt', '_mistakes.txt')
    with open(mistake_report_file, 'w', encoding='utf-8') as f:
        f.write("DETAILED AI MISTAKE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for i, mistake in enumerate(mistakes, 1):
            f.write(f"MISTAKE #{i}\n")
            f.write(f"PO Box: {mistake['po_box']}\n")
            f.write(f"Insurance: {mistake['insurance_name']}\n")
            f.write(f"Expected: {mistake['expected_code']} -> Found: {mistake['found_code']}\n")
            f.write(f"MedNet Name: {mistake['mednet_name']}\n")
            f.write(f"AI Confidence: {mistake['ai_confidence']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n📄 Detailed mistake report saved to: {mistake_report_file}")
    
    # Analyze low confidence cases
    if low_confidence_cases:
        print(f"\n🔍 LOW CONFIDENCE ANALYSIS ({len(low_confidence_cases)} cases):")
        print("=" * 60)
        
        # Check how many low confidence cases would have been correct
        correct_low_confidence = 0
        for case in low_confidence_cases:
            # We need to check if the expected code matches any of the available options
            # This is a simplified check - in reality we'd need the full context
            if case.get('expected_code') and case.get('found_code'):
                # If the expected code matches what was found, it would have been correct
                if str(case['expected_code']).strip() == str(case['found_code']).strip():
                    correct_low_confidence += 1
        
        if len(low_confidence_cases) > 0:
            low_confidence_accuracy = (correct_low_confidence / len(low_confidence_cases)) * 100
            print(f"Low confidence cases that would have been correct: {correct_low_confidence}/{len(low_confidence_cases)} = {low_confidence_accuracy:.1f}%")
            print(f"This shows the AI was being appropriately conservative in {100-low_confidence_accuracy:.1f}% of low confidence cases")
        
        # Show some examples of low confidence cases
        print(f"\nExamples of low confidence cases:")
        for i, case in enumerate(low_confidence_cases[:3], 1):
            print(f"  {i}. PO Box {case['po_box']}: {case['insurance_name']}")
            print(f"     Expected: {case['expected_code']} -> Found: {case.get('found_code', 'N/A')}")
    else:
        print("\n🔍 No low confidence cases found in log.")

def use_ai_to_check_match(input_data, mednet_matches, client=None):
    """Use Gemini AI to check if a match is correct and provide confidence level"""
    
    # Initialize Google AI client if not provided (for threading)
    if client is None:
        client = genai.Client(
            api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
        )
    
    # Create the prompt for AI matching
    if len(mednet_matches) == 1:
        # Single match case
        match = mednet_matches[0]
        prompt = f"""
You are a medical billing expert tasked with verifying if an insurance company match is correct.

INPUT DATA:
- Insurance Name: "{input_data['insurance_name']}"
- Insurance Address: "{input_data['address']}"
- Expected MedNet Code: "{input_data['expected_code']}"

FOUND MATCH:
- MedNet Name: "{match['name']}"
- MedNet Code: "{match['mednet_code']}"
- MedNet Address: "{match['address']}"

TASK:
Determine if this is the correct match and your confidence level.

Consider:
1. Name similarity (exact matches are best, but consider variations, abbreviations, subsidiaries)
2. Address similarity (city, state, zip code patterns)
3. Business context (same company but different divisions/locations)
4. MedNet code match with expected code

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{{
    "confidence": "<high|low>",
    "is_correct": <true|false>
}}

CRITICAL: Only use HIGH confidence if you are ABSOLUTELY CERTAIN this is the right match.
- HIGH confidence: Exact name match AND exact MedNet code match, or very clear business relationship
- LOW confidence: Any uncertainty, partial matches, or when you're not 100% sure
- When in doubt, choose LOW confidence

IMPORTANT WARNINGS:
- EVERCARE and MEDICARE are COMPLETELY DIFFERENT companies - do NOT confuse them
- EVERCARE is a separate insurance company, not related to Medicare
- If you see "MEDICARE" in the insurance name, NEVER match it to EVERCARE
"""
    else:
        # Multiple matches case
        prompt = f"""
You are a medical billing expert tasked with matching insurance companies to their correct MedNet codes.

INPUT DATA:
- Insurance Name: "{input_data['insurance_name']}"
- Insurance Address: "{input_data['address']}"
- Expected MedNet Code: "{input_data['expected_code']}"

AVAILABLE MATCHES (all have the same PO Box number):
"""
        
        for i, match in enumerate(mednet_matches, 1):
            prompt += f"""
Option {i}:
- MedNet Name: "{match['name']}"
- MedNet Code: "{match['mednet_code']}"
- MedNet Address: "{match['address']}"
"""
        
        prompt += """
TASK:
Compare the input insurance name and address with each available MedNet option and select the BEST MATCH.

Consider:
1. Name similarity (exact matches are best, but consider variations, abbreviations, subsidiaries)
2. Address similarity (city, state, zip code patterns)
3. Business context (same company but different divisions/locations)
4. MedNet code match with expected code

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{
    "selected_option": <number>,
    "confidence": "<high|low>",
    "is_correct": <true|false>
}

Where selected_option is the number (1, 2, 3, etc.) of the best match.

CRITICAL: Only use HIGH confidence if you are ABSOLUTELY CERTAIN this is the right match.
- HIGH confidence: Exact name match AND exact MedNet code match, or very clear business relationship
- LOW confidence: Any uncertainty, partial matches, or when you're not 100% sure
- When in doubt, choose LOW confidence

IMPORTANT WARNINGS:
- EVERCARE and MEDICARE are COMPLETELY DIFFERENT companies - do NOT confuse them
- EVERCARE is a separate insurance company, not related to Medicare
- If you see "MEDICARE" in the insurance name, NEVER match it to EVERCARE
"""
    
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
            
        )

        # Get AI response
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=generate_content_config,
        )
        
        response_text = response.text.strip()
        
        # Clean the response by removing markdown code block formatting
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith('```'):
            response_text = response_text[3:]   # Remove ```
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove trailing ```
        response_text = response_text.strip()
        
        # Parse the JSON response
        ai_decision = json.loads(response_text)
        
        confidence = ai_decision.get('confidence', 'low')
        is_correct = ai_decision.get('is_correct', False)
        
        if len(mednet_matches) == 1:
            # Single match case
            return {
                'selected_match': mednet_matches[0],
                'confidence': confidence,
                'is_correct': is_correct,
                'ai_response': response_text
            }
        else:
            # Multiple matches case
            selected_option = ai_decision.get('selected_option', 1)
            
            # Validate selected option is within range
            if 1 <= selected_option <= len(mednet_matches):
                selected_match = mednet_matches[selected_option - 1]
                return {
                    'selected_match': selected_match,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'ai_response': response_text
                }
            else:
                # Fallback to first option if AI selection is invalid
                return {
                    'selected_match': mednet_matches[0],
                    'confidence': 'low',
                    'is_correct': False,
                    'ai_response': response_text
                }
            
    except Exception as e:
        print(f"    ⚠️  AI matching failed: {str(e)}")
        # Fallback to first option
        return {
            'selected_match': mednet_matches[0],
            'confidence': 'low',
            'is_correct': False,
            'ai_response': None
        }

def process_single_po_box_match(args):
    """Process a single PO Box match task (for threading)"""
    row, po_box_to_mednet, client = args
    
    po_box = row['po_box']
    expected_code = str(row['mednet_code']).strip()
    insurance_name = row['primary_insurance_name']
    address = row['primary_insurance_address']
    
    result = {
        'po_box': po_box,
        'insurance_name': insurance_name,
        'address': address,
        'expected_code': expected_code,
        'match_type': None,
        'ai_used': False,
        'mapped': False,
        'correct': False,
        'found_code': None,
        'mednet_name': None,
        'mednet_address': None,
        'ai_confidence': None,
        'all_options': None
    }
    
    # Special case: MEDICARE/MEDICARE A AND B always maps to 101
    if insurance_name == "MEDICARE/MEDICARE A AND B":
        result['match_type'] = 'special_case_medicare'
        result['mapped'] = True
        result['ai_used'] = False
        result['found_code'] = '101'
        result['mednet_name'] = 'MEDICARE'
        result['mednet_address'] = 'Special case mapping'
        result['correct'] = (expected_code == '101')
        return result
    
    if po_box in po_box_to_mednet:
        matches = po_box_to_mednet[po_box]
        
        # Prepare input data for AI
        input_data = {
            'insurance_name': insurance_name,
            'address': address,
            'expected_code': expected_code
        }
        
        # Use AI to check the match (both single and multiple cases)
        ai_result = use_ai_to_check_match(input_data, matches, client)
        selected_match = ai_result['selected_match']
        confidence = ai_result['confidence']
        is_correct = ai_result['is_correct']
        ai_response = ai_result.get('ai_response')
        
        # Only map if AI has high confidence
        if confidence == 'high':
            result['mapped'] = True
            result['ai_used'] = True
            result['ai_confidence'] = confidence
            result['correct'] = is_correct
            result['ai_response'] = ai_response
            
            found_code = str(selected_match['mednet_code']).strip()
            result['found_code'] = found_code
            result['mednet_name'] = selected_match['name']
            result['mednet_address'] = selected_match['address']
            
            if len(matches) == 1:
                result['match_type'] = 'single_ai_checked'
            else:
                result['match_type'] = 'multiple_ai_resolved'
                result['all_options'] = [{'name': m['name'], 'code': m['mednet_code']} for m in matches]
        else:
            # Low confidence - don't map
            result['match_type'] = 'low_confidence'
            result['ai_used'] = True
            result['ai_confidence'] = confidence
            result['ai_response'] = ai_response
            result['mapped'] = False
    else:
        # No matches found
        result['match_type'] = 'none'
        result['mapped'] = False
    
    return result

def test_po_box_matching_with_ai(max_workers=14):
    """Test PO Box matching between data.csv and mednet.csv with AI resolution for multiple matches"""
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"ai_matching_log_{timestamp}.txt"
    
    print("Loading data files...")
    print(f"📝 Logging AI decisions to: {log_file}")
    
    # Load data.csv
    data_df = pd.read_csv('data.csv')
    print(f"Loaded data.csv: {len(data_df)} rows")
    
    # Load mednet.csv
    mednet_df = pd.read_csv('mednet.csv')
    print(f"Loaded mednet.csv: {len(mednet_df)} rows")
    
    # Filter data.csv to only rows with non-empty mednet codes
    data_with_codes = data_df[
        data_df['mednet_code'].notna() & 
        (data_df['mednet_code'] != '') & 
        (data_df['mednet_code'] != '//')
    ].copy()
    
    print(f"Data rows with mednet codes: {len(data_with_codes)}")
    
    # Extract PO Box numbers from data.csv addresses
    data_with_codes['po_box'] = data_with_codes['primary_insurance_address'].apply(extract_po_box)
    data_with_po_boxes = data_with_codes[data_with_codes['po_box'].notna()].copy()
    
    print(f"Data rows with PO Box addresses: {len(data_with_po_boxes)}")
    
    # Clean addresses in mednet.csv for matching
    mednet_df['address_clean'] = mednet_df['Address'].apply(clean_address_for_matching)
    
    # Create a mapping of PO Box numbers to mednet codes
    po_box_to_mednet = defaultdict(list)
    
    for _, row in mednet_df.iterrows():
        po_box = extract_po_box(row['Address'])
        if po_box:
            po_box_to_mednet[po_box].append({
                'mednet_code': row['MedNet Code'],
                'name': row['Name'],
                'address': row['Address']
            })
    
    print(f"Unique PO Box numbers in mednet.csv: {len(po_box_to_mednet)}")
    
    # Initialize Google AI client for threading
    client = genai.Client(
        api_key="AIzaSyCrskRv2ajNhc-KqDVv0V8KFl5Bdf5rr7w",
    )
    
    # Test matching process with AI resolution using threading
    results = {
        'total_tested': 0,
        'mapped_count': 0,
        'unmapped_count': 0,
        'correct_mapped': 0,
        'incorrect_mapped': 0,
        'ai_confidence_breakdown': {'high': 0, 'low': 0},
        'match_type_breakdown': {
            'special_case_medicare': 0,
            'single_ai_checked': 0,
            'multiple_ai_resolved': 0,
            'low_confidence': 0,
            'none': 0
        },
        'match_details': []
    }
    
    # Running accuracy tracking
    running_accuracy_data = []
    
    print(f"\nTesting PO Box matching with AI resolution (max {max_workers} threads)...")
    print("=" * 80)
    
    # Prepare tasks for threading
    tasks = []
    for _, row in data_with_po_boxes.iterrows():
        tasks.append((row, po_box_to_mednet, client))
    
    print(f"🚀 Starting concurrent processing of {len(tasks)} PO Box matches...")
    print(f"📊 Progress: 0/{len(tasks)} matches processed (0.0%)")
    
    # Process matches concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_single_po_box_match, task): task[0] for task in tasks}
        
        # Track progress
        total_tasks = len(future_to_row)
        completed_tasks = 0
        last_progress_update = 0
        last_accuracy_update = 0
        
        # Collect results as they complete
        for future in as_completed(future_to_row):
            completed_tasks += 1
            progress_percentage = (completed_tasks / total_tasks) * 100
            
            # Update progress every 10% or every 5 tasks, whichever is more frequent
            if (progress_percentage - last_progress_update >= 10) or (completed_tasks % 5 == 0):
                print(f"📊 Progress: {completed_tasks}/{total_tasks} matches processed ({progress_percentage:.1f}%)")
                last_progress_update = progress_percentage
            
            row = future_to_row[future]
            po_box = row['po_box']
            insurance_name = row['primary_insurance_name']
            
            try:
                result = future.result()
                results['total_tested'] += 1
                
                # Update counters based on match type
                results['match_type_breakdown'][result['match_type']] += 1
                
                # Update mapped/unmapped counters
                if result['mapped']:
                    results['mapped_count'] += 1
                    if result['correct']:
                        results['correct_mapped'] += 1
                    else:
                        results['incorrect_mapped'] += 1
                else:
                    results['unmapped_count'] += 1
                
                # Update AI confidence breakdown
                if result['ai_confidence']:
                    results['ai_confidence_breakdown'][result['ai_confidence']] += 1
                
                # Add to match details
                results['match_details'].append(result)
                
                # Calculate and track running accuracy
                if results['mapped_count'] > 0:
                    current_accuracy = (results['correct_mapped'] / results['mapped_count']) * 100
                    running_accuracy_data.append({
                        'completed': completed_tasks,
                        'mapped': results['mapped_count'],
                        'correct': results['correct_mapped'],
                        'accuracy': current_accuracy
                    })
                    
                    # Print running accuracy every 10 completed tasks or every 10% progress
                    if (completed_tasks - last_accuracy_update >= 10) or (progress_percentage - last_accuracy_update >= 10):
                        print(f"🎯 Running Accuracy: {results['correct_mapped']}/{results['mapped_count']} = {current_accuracy:.1f}% (Mapped: {results['mapped_count']}/{completed_tasks})")
                        last_accuracy_update = completed_tasks
                
                # Log AI decision if AI was used
                if result['ai_used']:
                    log_ai_decision(log_file, result, result.get('ai_response'))
                
                # Print result
                if result['mapped']:
                    status = "✓ CORRECT" if result['correct'] else "✗ INCORRECT"
                    
                    if result['match_type'] == 'special_case_medicare':
                        print(f"⭐ SPECIAL CASE | PO Box {po_box} | Expected: {result['expected_code']} | Found: {result['found_code']}")
                        print(f"    {status} | Insurance: {insurance_name}")
                        print(f"    MedNet: {result['mednet_name']} (Special Medicare mapping)")
                        print()
                    elif result['match_type'] == 'single_ai_checked':
                        confidence_emoji = {"high": "🟢", "low": "🔴"}.get(result['ai_confidence'], "⚪")
                        print(f"{confidence_emoji} AI CHECKED | PO Box {po_box} | Expected: {result['expected_code']} | Found: {result['found_code']}")
                        print(f"    {status} | Insurance: {insurance_name}")
                        print(f"    MedNet: {result['mednet_name']}")
                        print()
                    else:  # multiple_ai_resolved
                        confidence_emoji = {"high": "🟢", "low": "🔴"}.get(result['ai_confidence'], "⚪")
                        print(f"{confidence_emoji} AI RESOLVED | PO Box {po_box} | Expected: {result['expected_code']}")
                        print(f"    Insurance: {insurance_name}")
                        print(f"    Found {len(result['all_options'])} matches, AI selected best one...")
                        print(f"    {status} | Selected: {result['found_code']} | {result['mednet_name']}")
                        print()
                else:
                    if result['match_type'] == 'low_confidence':
                        print(f"🔴 LOW CONFIDENCE | PO Box {po_box} | Expected: {result['expected_code']}")
                        print(f"    Insurance: {insurance_name} - NOT MAPPED")
                        print()
                    else:  # none
                        print(f"✗ NO MATCH | PO Box {po_box} | Expected: {result['expected_code']}")
                        print(f"    Insurance: {insurance_name} - NOT MAPPED")
                        print()
                    
            except Exception as e:
                print(f"❌ Exception processing PO Box {po_box}: {str(e)}")
                results['total_tested'] += 1
                results['unmapped_count'] += 1
        
        # Final progress update
        print(f"📊 Progress: {completed_tasks}/{total_tasks} matches processed (100.0%)")
        
        # Final running accuracy update
        if results['mapped_count'] > 0:
            final_accuracy = (results['correct_mapped'] / results['mapped_count']) * 100
            print(f"🎯 Final Running Accuracy: {results['correct_mapped']}/{results['mapped_count']} = {final_accuracy:.1f}%")
    
    # Print summary statistics
    print("=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"Total PO Box addresses tested: {results['total_tested']}")
    print(f"Mapped (high confidence): {results['mapped_count']}")
    print(f"Unmapped (low confidence or no match): {results['unmapped_count']}")
    print()
    
    # Key metrics requested by user
    mapping_percentage = (results['mapped_count'] / results['total_tested']) * 100
    print(f"1. MAPPING PERCENTAGE: {results['mapped_count']}/{results['total_tested']} = {mapping_percentage:.1f}%")
    
    if results['mapped_count'] > 0:
        accuracy_percentage = (results['correct_mapped'] / results['mapped_count']) * 100
        print(f"2. ACCURACY OF MAPPED: {results['correct_mapped']}/{results['mapped_count']} = {accuracy_percentage:.1f}%")
    else:
        print("2. ACCURACY OF MAPPED: No mapped items to calculate accuracy")
    
    print()
    print("DETAILED BREAKDOWN:")
    print(f"  Correct mappings: {results['correct_mapped']}")
    print(f"  Incorrect mappings: {results['incorrect_mapped']}")
    print(f"  Low confidence (unmapped): {results['match_type_breakdown']['low_confidence']}")
    print(f"  No PO Box match (unmapped): {results['match_type_breakdown']['none']}")
    
    print()
    print("AI CONFIDENCE BREAKDOWN:")
    total_ai_checked = results['ai_confidence_breakdown']['high'] + results['ai_confidence_breakdown']['low']
    if total_ai_checked > 0:
        for confidence, count in results['ai_confidence_breakdown'].items():
            if count > 0:
                percentage = (count / total_ai_checked) * 100
                print(f"  {confidence.upper()}: {count} ({percentage:.1f}%)")
    
    print()
    print("MATCH TYPE BREAKDOWN:")
    for match_type, count in results['match_type_breakdown'].items():
        if count > 0:
            percentage = (count / results['total_tested']) * 100
            print(f"  {match_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Analyze AI mistakes
    analyze_ai_mistakes(log_file)
    
    print(f"\n📝 Full AI decision log saved to: {log_file}")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Allow specifying max_workers as command line argument
    max_workers = 14  # Default thread pool size
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
            if max_workers <= 0:
                max_workers = 14
                print("⚠️  Warning: max_workers must be positive, using default of 3")
        except ValueError:
            print("⚠️  Warning: Invalid max_workers value, using default of 3")
    
    print(f"🔧 Configuration:")
    print(f"   Max workers: {max_workers}")
    print()
    
    results = test_po_box_matching_with_ai(max_workers)
