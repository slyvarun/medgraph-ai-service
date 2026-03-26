"""
ai_service.py  —  MedGraph Nexus  |  FastAPI entry point
=========================================================
Starts the HTTP server, serves the frontend, and routes
POST /ask → query_agent.ask_agent().

Run locally:
    python ai_service.py

Or via uvicorn directly:
    uvicorn ai_service:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator

from query_agent import ask_agent, close_driver

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
)
log = logging.getLogger("ai_service")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedGraph Nexus",
    description="RAG clinical assistant — Neo4j graph + Gemini LLM",
    version="2.0.0",
    docs_url="/docs",        # Swagger UI at /docs (handy during dev)
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allows the HTML frontend (any origin) to call /ask.
# Lock down allow_origins to your domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────────────────────
# index.html lives next to this file; FastAPI serves it at the root URL.
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="index.html not found next to ai_service.py")
    return FileResponse(path)


# ── Models ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

    @validator("question")
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or fewer.")
        return v


class AskResponse(BaseModel):
    answer: str
    status: str = "ok"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """Liveness probe — used by Render / Railway / Docker health checks."""
    return {"status": "ok", "service": "MedGraph Nexus"}


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(body: AskRequest):
    """
    Main RAG endpoint.
    Accepts a natural-language question, queries Neo4j for relevant
    medicine records, and returns a Gemini-generated answer.
    """
    log.info(f"Question received: {body.question!r}")

    try:
        answer = ask_agent(body.question)
        return AskResponse(answer=answer)

    except ValueError as exc:
        log.warning(f"Validation error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))

    except ConnectionError as exc:
        log.error(f"DB connection error: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Database temporarily unavailable. Please try again shortly.",
        )

    except Exception as exc:
        log.exception(f"Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    log.info("✅ MedGraph Nexus API is ready.")


@app.on_event("shutdown")
async def shutdown():
    close_driver()
    log.info("Neo4j driver closed. Goodbye.")


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Use the PORT env var provided by the host, default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)