"""
ai_service.py  —  MedGraph Nexus  |  FastAPI Entrypoint & API Server
====================================================================
Starts HTTP API server, serves single-page frontend, and routes RAG requests.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

import threading
import time
from query_agent import ask_agent, get_graph_visualization, close_driver, keep_alive_ping

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
)
log = logging.getLogger("ai_service")

CACHE_JSON_PATH = os.path.join(os.path.dirname(__file__), "medgraph_cache.json")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedGraph Nexus",
    description="Knowledge Graph RAG Clinical Assistant — Neo4j + Gemini 2.0 Flash",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{full_path:path}", include_in_schema=False)
async def options_handler(full_path: str):
    """Explicit preflight handler to guarantee Access-Control headers on all cross-origin requests."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE, PATCH, HEAD",
            "Access-Control-Allow-Headers": "*",
        },
    )



FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.dirname(__file__)

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def root():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path)


@app.get("/logo.png", response_class=FileResponse, include_in_schema=False)
async def logo():
    path = os.path.join(FRONTEND_DIR, "logo.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="logo.png not found")
    return FileResponse(path)





# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    language: Optional[str] = "en"

    @field_validator("question")
    @classmethod
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or fewer.")
        return v


class AskResponse(BaseModel):
    answer: str
    graph: Optional[Dict[str, Any]] = None
    status: str = "ok"


# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "MedGraph Nexus Clinical Engine"}


@app.get("/graph-stats", tags=["Knowledge Graph"])
async def graph_stats():
    """Return summary metadata of Knowledge Graph entities."""
    if os.path.exists(CACHE_JSON_PATH):
        try:
            with open(CACHE_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "status": "ok",
                "metadata": data.get("metadata", {}),
                "total_records": len(data.get("records", []))
            }
        except Exception as exc:
            log.warning(f"Error loading stats: {exc}")

    return {
        "status": "ok",
        "metadata": {
            "total_medicines": 64,
            "unique_indications": 8,
            "unique_categories": 8,
            "unique_manufacturers": 12,
            "unique_dosage_forms": 8
        }
    }


@app.get("/graph-subgraph", tags=["Knowledge Graph"])
async def graph_subgraph(q: str = Query(..., min_length=1, description="Query term to visualize")):
    """Returns nodes and edges formatted for Vis.js UI rendering."""
    try:
        viz = get_graph_visualization(q)
        return {"status": "ok", "query": q, "graph": viz}
    except Exception as exc:
        log.error(f"Error generating graph visualization: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch Knowledge Graph visualization.")


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(body: AskRequest):
    """
    Main GraphRAG endpoint.
    Retrieves connected medicine and symptom subgraphs, generates clinical answer,
    and returns response with visualizer nodes & edges.
    """
    log.info(f"Clinical question received: {body.question!r} | language: {body.language!r}")
    try:
        answer = ask_agent(body.question, language=body.language or "en")
        graph_data = get_graph_visualization(body.question)
        return AskResponse(answer=answer, graph=graph_data)


    except ValueError as exc:
        log.warning(f"Validation error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        log.exception(f"Unexpected error in /ask endpoint: {exc}")
        raise HTTPException(status_code=500, detail="Internal clinical engine server error.")


@app.get("/ping-heartbeat", tags=["System"])
async def ping_heartbeat():
    """
    Triggers a write query to Neo4j to keep Cloud AuraDB active and prevent auto-pausing.
    Can be called by cron services (e.g. cron-job.org) every 24 hours.
    """
    success = keep_alive_ping()
    if success:
        return {"status": "ok", "message": "Neo4j keep-alive write query executed successfully."}
    return {"status": "warning", "message": "Neo4j write ping failed or offline."}


# ── Lifecycle & Keep-Alive Loop ───────────────────────────────────────────────
_stop_heartbeat = False

def _heartbeat_loop():
    """Background thread that pings Neo4j every 12 hours to prevent database auto-pausing."""
    log.info("⚡ Background Neo4j keep-alive loop started (Pings every 12 hours).")
    while not _stop_heartbeat:
        keep_alive_ping()
        # Sleep for 12 hours (43,200 seconds)
        for _ in range(43200):
            if _stop_heartbeat:
                break
            time.sleep(1)

@app.on_event("startup")
async def startup():
    log.info("✅ MedGraph Nexus Clinical API ready.")
    keep_alive_ping()
    # Start background ping thread
    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()


@app.on_event("shutdown")
async def shutdown():
    global _stop_heartbeat
    _stop_heartbeat = True
    close_driver()
    log.info("Neo4j driver closed. Goodbye.")



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
