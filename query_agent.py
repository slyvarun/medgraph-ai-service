"""
query_agent.py  —  MedGraph Nexus  |  RAG Brain
================================================
This module owns:
  1. The Neo4j driver (one shared instance)
  2. search_graph()  — fuzzy Cypher search over Medicine nodes
  3. ask_agent()     — full RAG pipeline (search → context → Gemini)
  4. _call_gemini()  — retry logic + model fallback on 429 errors
  5. close_driver()  — graceful teardown called by FastAPI shutdown hook

Neo4j property schema (must match production_ingest.py COLUMN_MAP):
  :Medicine {
      name, category, indication,
      strength, manufacturer, dosage_form, classification
  }
"""

import os
import time
import logging
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("query_agent")

# ── Environment ───────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_required = {"NEO4J_URI": NEO4J_URI, "NEO4J_PASSWORD": NEO4J_PASSWORD, "GEMINI_API_KEY": GEMINI_API_KEY}
_missing  = [k for k, v in _required.items() if not v]
if _missing:
    raise EnvironmentError(
        f"Missing environment variables: {', '.join(_missing)}\n"
        "Create a .env file — see .env.example for the template."
    )

# ── Gemini ────────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

PRIMARY_MODEL   = "gemini-1.5-flash"
FALLBACK_MODEL  = "gemini-1.5-flash-8b"   # lighter quota tier
MAX_RETRIES     = 3
RETRY_BASE_SEC  = 5   # wait = RETRY_BASE_SEC × attempt  (5 s, 10 s, 15 s)

# ── Neo4j driver (module-level singleton) ──────────────────────────────────────
log.info(f"Connecting to Neo4j at {NEO4J_URI} ...")
try:
    _driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=30,
    )
    _driver.verify_connectivity()
    log.info("Neo4j connection verified ✅")
except AuthError:
    raise EnvironmentError(
        "Neo4j authentication failed.\n"
        "Check NEO4J_USER and NEO4J_PASSWORD in your .env file."
    )
except ServiceUnavailable:
    raise ConnectionError(
        "Cannot reach Neo4j.\n"
        "Check NEO4J_URI and make sure the Aura instance is running."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH SEARCH
# ─────────────────────────────────────────────────────────────────────────────

# Cypher: case-insensitive fuzzy CONTAINS across all searchable properties.
# CALL { ... UNION ... } deduplicates results that match on multiple fields.
_SEARCH_CYPHER = """
CALL {
    MATCH (m:Medicine)
    WHERE toLower(m.name) CONTAINS toLower($q)
    RETURN m LIMIT 5

    UNION

    MATCH (m:Medicine)
    WHERE toLower(m.indication) CONTAINS toLower($q)
    RETURN m LIMIT 5

    UNION

    MATCH (m:Medicine)
    WHERE toLower(m.category) CONTAINS toLower($q)
    RETURN m LIMIT 5

    UNION

    MATCH (m:Medicine)
    WHERE toLower(m.manufacturer) CONTAINS toLower($q)
    RETURN m LIMIT 5

    UNION

    MATCH (m:Medicine)
    WHERE toLower(m.classification) CONTAINS toLower($q)
    RETURN m LIMIT 5

    UNION

    MATCH (m:Medicine)
    WHERE toLower(m.dosage_form) CONTAINS toLower($q)
    RETURN m LIMIT 5
}
RETURN DISTINCT m {
    .name,
    .category,
    .indication,
    .strength,
    .manufacturer,
    .dosage_form,
    .classification
} AS medicine
LIMIT 12
"""


def search_graph(query: str) -> list[dict]:
    """
    Run a fuzzy, case-insensitive search across all Medicine properties.

    Returns up to 12 matching records as plain dicts.
    Returns [] on any database error (never raises — the LLM handles the
    empty-context case gracefully).
    """
    try:
        with _driver.session() as session:
            result  = session.run(_SEARCH_CYPHER, q=query)
            records = [dict(r["medicine"]) for r in result]
            log.info(f"Graph search '{query}' → {len(records)} record(s)")
            return records
    except Exception as exc:
        log.error(f"Neo4j search failed: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_context(medicines: list[dict]) -> str:
    """Format matched medicine records into a readable block for the LLM prompt."""
    if not medicines:
        return (
            "No matching medicines were found in the database for this query.\n"
            "Inform the user and suggest they try broader search terms."
        )

    lines = []
    for i, m in enumerate(medicines, 1):
        lines.append(
            f"{i}. {m.get('name') or 'Unknown'}"
            f"  [{m.get('classification') or 'N/A'}]\n"
            f"   Category    : {m.get('category')     or 'N/A'}\n"
            f"   Indication  : {m.get('indication')   or 'N/A'}\n"
            f"   Dosage Form : {m.get('dosage_form')  or 'N/A'}\n"
            f"   Strength    : {m.get('strength')     or 'N/A'}\n"
            f"   Manufacturer: {m.get('manufacturer') or 'N/A'}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are MedGraph Nexus, a precise and professional clinical AI assistant.
Your answers must be grounded EXCLUSIVELY in the DATABASE RECORDS provided below.

STRICT RULES:
1. Use ONLY the information present in the DATABASE RECORDS section.
2. Never invent, hallucinate, or add information not present in the records.
3. If the data is insufficient, say so clearly and recommend the user consult
   a licensed healthcare professional or pharmacist.
4. Format answers with clean Markdown:
   - Use **bold** for medicine names.
   - Use bullet lists for multiple items.
   - Use a Markdown table when comparing 3 or more medicines.
5. Be concise and clinically accurate.
6. Always end with this exact disclaimer on its own line:
   > ⚕ This information is for reference only. Consult a qualified healthcare
   > professional before making any medical decisions.

DATABASE RECORDS:
{context}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GEMINI CALL  (retry + fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(system_prompt: str, question: str) -> str:
    """
    Call Gemini with retry + model fallback.

    Retry schedule per model:
        attempt 1  →  wait 5 s  →  attempt 2  →  wait 10 s  →  attempt 3
    Model cascade:
        gemini-1.5-flash  →  gemini-1.5-flash-8b  →  error message
    """
    for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log.info(f"Calling {model_name} (attempt {attempt}/{MAX_RETRIES})")
                model    = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt,
                )
                response = model.generate_content(question)
                log.info(f"Response received from {model_name} ✅")
                return response.text

            except ResourceExhausted as exc:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BASE_SEC * attempt
                    log.warning(f"[429] {model_name} attempt {attempt} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    log.warning(f"[429] {model_name} all retries exhausted — switching model")
                    break

            except Exception as exc:
                log.error(f"Gemini error on {model_name}: {exc}")
                return (
                    f"⚠️ An error occurred while generating the response: {exc}\n\n"
                    "Please try again in a moment."
                )

    log.error("All Gemini models rate-limited.")
    return (
        "⚠️ The AI service is temporarily rate-limited. "
        "Please wait 60 seconds and try again."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def ask_agent(question: str) -> str:
    """
    Full RAG pipeline:
      1. search Neo4j for matching medicine nodes
      2. build a grounded context string
      3. inject context into the system prompt
      4. call Gemini (with retry/fallback)
      5. return Markdown-formatted answer

    Raises:
        ValueError  — if question is blank
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    # Step 1 — Graph retrieval
    medicines = search_graph(question)

    # Step 2 — Build context
    context = _build_context(medicines)

    # Step 3 — Inject into system prompt
    system_prompt = _SYSTEM_PROMPT.format(context=context)

    # Step 4 — Generate answer
    return _call_gemini(system_prompt, question)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def close_driver():
    """Close the Neo4j driver. Called by the FastAPI shutdown hook."""
    try:
        _driver.close()
        log.info("Neo4j driver closed.")
    except Exception:
        pass