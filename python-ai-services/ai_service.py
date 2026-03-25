import sys
import os

# Add the current folder to the Python search path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from fastapi import FastAPI, Body
from query_agent import ask_agent # This imports the logic we built earlier

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import Request
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()
# Add this right after app = FastAPI()
# THIS IS THE UNFREEZE BUTTON
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your HTML file to talk to the server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/ask")
async def ask(request: Request):
    # This reads the RAW text body from your HTML fetch
    body = await request.body()
    query_str = body.decode("utf-8")
    
    # Pass the string to your agent
    response = ask_agent(query_str)
    return response
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
