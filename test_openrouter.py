#!/usr/bin/env python3
"""
Test script to debug OpenRouter API issues with DeepSeek models
"""
import os
import requests
import json

def test_openrouter_models():
    """Test various DeepSeek model names on OpenRouter"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set in environment")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Test different model names
    test_models = [
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
        "deepseek/deepseek-chat-v3",
        "deepseek/deepseek-v3",
    ]
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/medical-data-processor",
        "X-Title": "Medical Data Processor"
    }
    
    # Simple test message
    test_message = {
        "role": "user",
        "content": "Hello, respond with 'OK' if you can read this."
    }
    
    print("\n" + "="*60)
    print("Testing OpenRouter API with different DeepSeek models")
    print("="*60 + "\n")
    
    for model in test_models:
        print(f"Testing model: {model}")
        payload = {
            "model": model,
            "messages": [test_message]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                    print(f"  ✅ SUCCESS - Response: {content[:50]}...")
                else:
                    print(f"  ⚠️  Unexpected response format: {json.dumps(data, indent=2)[:200]}")
            elif response.status_code == 404:
                error_data = response.json() if response.text else {}
                print(f"  ❌ 404 ERROR - Model not found")
                print(f"     Response: {response.text[:200]}")
            else:
                error_data = response.json() if response.text else {}
                print(f"  ❌ ERROR {response.status_code}")
                print(f"     Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
        
        print()
    
    # Also test listing available models
    print("="*60)
    print("Testing OpenRouter models endpoint")
    print("="*60 + "\n")
    
    try:
        models_url = "https://openrouter.ai/api/v1/models"
        models_response = requests.get(models_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        if models_response.status_code == 200:
            models_data = models_response.json()
            deepseek_models = [m for m in models_data.get("data", []) if "deepseek" in m.get("id", "").lower()]
            print(f"Found {len(deepseek_models)} DeepSeek models:")
            for model in deepseek_models[:10]:  # Show first 10
                print(f"  - {model.get('id')} ({model.get('name', 'N/A')})")
        else:
            print(f"Failed to fetch models: {models_response.status_code}")
            print(models_response.text[:200])
    except Exception as e:
        print(f"Exception fetching models: {str(e)}")

if __name__ == "__main__":
    test_openrouter_models()

