import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from query_agent import ask_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def handle_ask(request: Request):
    try:
        # Read the raw body string from the fetch call
        body = await request.body()
        user_query = body.decode("utf-8")
        
        # Pass both arguments to prevent 'Internal Server Error'
        answer = ask_agent(user_query, "") 
        return answer
    except Exception as e:
        print(f"🔥 Server Error: {e}")
        return f"Nexus Error: {str(e)}"

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)