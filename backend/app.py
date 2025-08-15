# This is an alternative entry point for Railway/Nixpacks
from main import app

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting app.py on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 