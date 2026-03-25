import sys
import os

# This tells Python to look inside your subfolder for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "python-ai-services"))

from ai_service import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
