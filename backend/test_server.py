#!/usr/bin/env python3
"""
Simple test script to verify the backend server configuration
"""
import os
import sys
import requests
import time

def test_server():
    """Test if the server is running and responding"""
    port = int(os.environ.get('PORT', 8000))
    base_url = f"http://localhost:{port}"
    
    print(f"Testing server on {base_url}")
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root endpoint: {response.status_code} - {response.json()}")
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health endpoint: {response.status_code} - {response.json()}")
        
        print("✅ Server is running correctly!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_server() 