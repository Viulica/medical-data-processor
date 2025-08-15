#!/bin/bash

echo "🚀 Starting Medical Data Processor API..."
echo "📡 PORT: $PORT"
echo "🌍 Environment: $RAILWAY_ENVIRONMENT"
echo "🏗️ Project ID: $RAILWAY_PROJECT_ID"
echo "🚂 Service: $RAILWAY_SERVICE_NAME"

# Check if PORT is set
if [ -z "$PORT" ]; then
    echo "❌ PORT environment variable not set!"
    exit 1
fi

# Start the application
echo "✅ Starting uvicorn on port $PORT..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT 