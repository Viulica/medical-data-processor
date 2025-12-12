#!/bin/bash
# Test script for OCR.space integration

echo "🧪 OCR.space API Test"
echo "===================="
echo ""

# Check if API key is set
if [ -z "$OCRSPACE_API_KEY" ]; then
    echo "❌ OCRSPACE_API_KEY environment variable not set!"
    echo ""
    echo "Please set it first:"
    echo "  export OCRSPACE_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

# Show masked API key
KEY_LENGTH=${#OCRSPACE_API_KEY}
KEY_START=${OCRSPACE_API_KEY:0:8}
KEY_END=${OCRSPACE_API_KEY: -4}
echo "✅ API key found: ${KEY_START}...${KEY_END} (length: $KEY_LENGTH)"
echo ""

# Run the test
cd "$(dirname "$0")"
python current/1-split_pdf_ocrspace.py DUN.pdf output "Patient Address"

