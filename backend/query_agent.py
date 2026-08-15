"""
query_agent.py  —  MedGraph Nexus  |  GraphRAG Engine
===================================================
This module powers the clinical Knowledge Graph RAG:
  1. Neo4j driver connection with fallback to local JSON Graph Cache.
  2. Multi-hop Cypher queries retrieving Medicine, Symptom/Indication, Category, Manufacturer, and Alternatives.
  3. Interactive Knowledge Graph subgraph exporter for Vis.js visualization.
  4. Clinical reasoning prompt for Gemini LLM describing:
     - What the medicine is
     - Why do we use it
     - When should we use it (Symptoms & Indications)
     - Alternative options for the same symptom
  5. Fallback pipeline (openFDA API + deterministic Markdown rendering).
"""

import os
import time
import logging
import re
import json
from dotenv import load_dotenv

import google.generativeai as genai
import httpx
from google.api_core.exceptions import ResourceExhausted
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("query_agent")

# ── Environment ───────────────────────────────────────────────────────────────
NEO4J_URI        = os.getenv("NEO4J_URI")
NEO4J_USER       = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
OPENFDA_API_KEY  = os.getenv("OPENFDA_API_KEY")
OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
CACHE_JSON_PATH  = os.path.join(os.path.dirname(__file__), "medgraph_cache.json")

# ── Gemini Setup ──────────────────────────────────────────────────────────────
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

PRIMARY_MODEL   = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.0-flash")
FALLBACK_MODEL  = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash-lite")
MAX_RETRIES     = 3
RETRY_BASE_SEC  = 5

_STATIC_MODEL_FALLBACKS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-flash",
]


def _resolve_available_models(candidates: list[str]) -> list[str]:
    """Return an ordered list of available Gemini models instantly without network blocking."""
    return list(dict.fromkeys(candidates + _STATIC_MODEL_FALLBACKS))


MODEL_CANDIDATES = _resolve_available_models([PRIMARY_MODEL, FALLBACK_MODEL])


# ── Neo4j Driver Singleton ─────────────────────────────────────────────────────
_driver = None
if NEO4J_URI and NEO4J_PASSWORD:
    try:
        log.info(f"Connecting to Neo4j at {NEO4J_URI}...")
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=5,
        )
        _driver.verify_connectivity()
        log.info("Neo4j connection verified ✅")
    except Exception as exc:
        log.warning(f"Neo4j connection failed: {exc}. Will use local Knowledge Graph cache.")
        _driver = None
else:
    log.warning("NEO4J_URI or NEO4J_PASSWORD not configured. Will use local Knowledge Graph cache.")


# ── Embedded Local Cache ──────────────────────────────────────────────────────
_LOCAL_CACHE = None

def _get_local_cache() -> dict:
    global _LOCAL_CACHE
    if _LOCAL_CACHE is None:
        if os.path.exists(CACHE_JSON_PATH):
            try:
                with open(CACHE_JSON_PATH, "r", encoding="utf-8") as f:
                    _LOCAL_CACHE = json.load(f)
                log.info(f"Loaded local Knowledge Graph cache ({len(_LOCAL_CACHE.get('records', []))} medicines)")
            except Exception as exc:
                log.error(f"Error reading local JSON cache: {exc}")
                _LOCAL_CACHE = {"records": []}
        else:
            log.warning("No local medgraph_cache.json found.")
            _LOCAL_CACHE = {"records": []}
    return _LOCAL_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRAPH SEARCH (Cypher & Local Embedded Graph)
# ─────────────────────────────────────────────────────────────────────────────

# Cypher for Neo4j: Finds target medicines + connected Indications (Symptoms), Categories, Manufacturers, and Dosage Forms
_GRAPH_CYPHER = """
MATCH (m:Medicine)
WHERE toLower(m.name) CONTAINS toLower($q)
   OR toLower(m.indication) CONTAINS toLower($q)
   OR toLower(m.category) CONTAINS toLower($q)
   OR toLower(m.manufacturer) CONTAINS toLower($q)
   OR toLower(m.classification) CONTAINS toLower($q)
OPTIONAL MATCH (m)-[:TREATS_INDICATION]->(i:Indication)
OPTIONAL MATCH (m)-[:BELONGS_TO_CATEGORY]->(c:Category)
OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
OPTIONAL MATCH (m)-[:AVAILABLE_AS]->(df:DosageForm)
OPTIONAL MATCH (alt:Medicine)-[:TREATS_INDICATION]->(i)
WHERE alt.name <> m.name
RETURN DISTINCT m {
    .name,
    .category,
    .indication,
    .strength,
    .manufacturer,
    .dosage_form,
    .classification
} AS medicine,
collect(DISTINCT alt.name)[..3] AS alternatives
LIMIT 12
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:mg|ml|mcg|g|iu)?")
_STOPWORDS = {
    "medicine", "medicines", "drug", "drugs", "strength", "show",
    "list", "find", "with", "for", "the", "and", "or", "of", "a", "an",
    "why", "do", "we", "use", "when", "should", "i", "take", "what", "are",
    "symptom", "symptoms", "used", "treat", "treatment"
}


def _build_search_variants(query: str) -> list[str]:
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return []
    variants = [normalized]
    for token in _TOKEN_RE.findall(normalized):
        if token not in _STOPWORDS and (len(token) >= 3 or any(ch.isdigit() for ch in token)):
            variants.append(token)
    return list(dict.fromkeys(variants))


def _search_local_cache(query: str) -> list[dict]:
    """Fallback search using embedded local Knowledge Graph cache."""
    cache = _get_local_cache()
    records = cache.get("records", [])
    if not records:
        return []

    variants = _build_search_variants(query)
    matched = []
    seen = set()

    for v in variants:
        for r in records:
            if r["name"] in seen:
                continue
            # Match fields
            name_match = v in r.get("name", "").lower()
            ind_match  = v in r.get("indication", "").lower()
            cat_match  = v in r.get("category", "").lower()
            mfg_match  = v in r.get("manufacturer", "").lower()

            if name_match or ind_match or cat_match or mfg_match:
                seen.add(r["name"])
                # Find alternatives sharing indication
                alts = [
                    o["name"] for o in records
                    if o["name"] != r["name"] and o.get("indication") and o.get("indication") == r.get("indication")
                ][:3]

                matched.append({
                    "medicine": r,
                    "alternatives": alts
                })
                if len(matched) >= 12:
                    break

    log.info(f"Local graph cache search '{query}' → {len(matched)} match(es)")
    return matched


def search_graph(query: str) -> list[dict]:
    """Search Knowledge Graph (Neo4j if connected, else Local Cache)."""
    if _driver:
        try:
            variants = _build_search_variants(query)
            if not variants:
                return []
            seen = set()
            records = []
            with _driver.session() as session:
                for variant in variants:
                    result = session.run(_GRAPH_CYPHER, q=variant)
                    for r in result:
                        med = dict(r["medicine"])
                        name = med.get("name")
                        if name in seen:
                            continue
                        seen.add(name)
                        records.append({
                            "medicine": med,
                            "alternatives": r.get("alternatives", [])
                        })
                        if len(records) >= 12:
                            break
                    if len(records) >= 12:
                        break
            if records:
                log.info(f"Neo4j graph search '{query}' → {len(records)} record(s)")
                return records
        except Exception as exc:
            log.error(f"Neo4j search failed: {exc}. Falling back to local cache.")

    return _search_local_cache(query)


# ─────────────────────────────────────────────────────────────────────────────
# 2. OPENFDA FALLBACK & SUBGRAPH VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

def _openfda_to_medicine(item: dict) -> dict:
    openfda = item.get("openfda") or {}
    return {
        "name": ", ".join(openfda.get("brand_name", [])[:2]) or ", ".join(openfda.get("generic_name", [])[:2]) or "Unknown",
        "category": ", ".join(openfda.get("product_type", [])[:2]) or "N/A",
        "indication": " ".join((item.get("indications_and_usage") or ["N/A"])[0:1])[:400] or "N/A",
        "strength": "N/A",
        "manufacturer": ", ".join(openfda.get("manufacturer_name", [])[:2]) or "N/A",
        "dosage_form": ", ".join(openfda.get("dosage_form", [])[:2]) or "N/A",
        "classification": ", ".join(openfda.get("pharm_class_epc", [])[:2]) or "N/A",
    }


def search_openfda(query: str, limit: int = 8) -> list[dict]:
    variants = _build_search_variants(query)
    if not variants:
        return []
    seen = set()
    records = []
    try:
        with httpx.Client(timeout=10.0) as client:
            for term in variants:
                params = {"search": f'openfda.brand_name:"{term}" OR openfda.generic_name:"{term}"', "limit": str(limit)}
                if OPENFDA_API_KEY:
                    params["api_key"] = OPENFDA_API_KEY
                resp = client.get(OPENFDA_BASE_URL, params=params)
                if resp.status_code == 200:
                    for item in resp.json().get("results", []):
                        med = _openfda_to_medicine(item)
                        if med["name"] not in seen:
                            seen.add(med["name"])
                            records.append({"medicine": med, "alternatives": []})
                            if len(records) >= limit:
                                return records
    except Exception as exc:
        log.warning(f"openFDA search error: {exc}")
    return records


def get_graph_visualization(query: str) -> dict:
    """
    Returns nodes and edges formatted for Vis.js UI visualization.
    """
    matched = search_graph(query)
    nodes = []
    edges = []
    node_ids = set()

    for item in matched:
        m = item["medicine"]
        m_name = m.get("name", "Unknown")
        m_id = f"med_{m_name}"

        if m_id not in node_ids:
            nodes.append({
                "id": m_id,
                "label": m_name,
                "group": "Medicine",
                "title": f"Medicine: {m_name}\nClassification: {m.get('classification', 'N/A')}\nStrength: {m.get('strength', 'N/A')}"
            })
            node_ids.add(m_id)

        # Indication / Symptom Node
        ind = m.get("indication")
        if ind:
            ind_id = f"ind_{ind}"
            if ind_id not in node_ids:
                nodes.append({
                    "id": ind_id,
                    "label": f"🌡️ {ind}",
                    "group": "Indication",
                    "title": f"Indication/Symptom: {ind}"
                })
                node_ids.add(ind_id)
            edges.append({"from": m_id, "to": ind_id, "label": "TREATS_INDICATION"})

        # Category Node
        cat = m.get("category")
        if cat:
            cat_id = f"cat_{cat}"
            if cat_id not in node_ids:
                nodes.append({
                    "id": cat_id,
                    "label": f"🏷️ {cat}",
                    "group": "Category",
                    "title": f"Category: {cat}"
                })
                node_ids.add(cat_id)
            edges.append({"from": m_id, "to": cat_id, "label": "BELONGS_TO"})

        # Manufacturer Node
        mfg = m.get("manufacturer")
        if mfg:
            mfg_id = f"mfg_{mfg}"
            if mfg_id not in node_ids:
                nodes.append({
                    "id": mfg_id,
                    "label": f"🏢 {mfg}",
                    "group": "Manufacturer",
                    "title": f"Manufacturer: {mfg}"
                })
                node_ids.add(mfg_id)
            edges.append({"from": m_id, "to": mfg_id, "label": "MANUFACTURED_BY"})

    return {"nodes": nodes, "edges": edges}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT BUILDER & CLINICAL SYSTEM INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_context(matched_records: list[dict]) -> str:
    if not matched_records:
        return "No matching medicines found in the Knowledge Graph for this query."

    lines = []
    for i, item in enumerate(matched_records, 1):
        m = item["medicine"]
        alts = item.get("alternatives", [])
        alt_str = ", ".join(alts) if alts else "None listed"

        lines.append(
            f"{i}. **{m.get('name', 'Unknown')}** [{m.get('classification', 'N/A')}]\n"
            f"   - Category (What it is)        : {m.get('category', 'N/A')}\n"
            f"   - Indication (Why/When to use) : {m.get('indication', 'N/A')}\n"
            f"   - Dosage Form & Strength       : {m.get('dosage_form', 'N/A')} ({m.get('strength', 'N/A')})\n"
            f"   - Manufacturer                 : {m.get('manufacturer', 'N/A')}\n"
            f"   - Graph Alternatives for Indication: {alt_str}"
        )
    return "\n\n".join(lines)


_SYSTEM_PROMPT = """\
You are MedGraph Nexus, an expert clinical Doctor AI assistant powered by a Knowledge Graph.
Your task is to answer the user's question clearly, thoroughly, and accurately based strictly on the KNOWLEDGE GRAPH RECORDS provided below.

Structure your answer using clean Markdown headers:
1. 📌 **Overview & Classification**: Explain what the medicine is (Category, Strength, Dosage Form, Classification).
2. ❓ **Why Do We Use It?**: Detail its therapeutic purpose and indication.
3. 🌡️ **When Should You Use It? (Symptoms & Conditions)**: List specific symptoms and clinical conditions for which this medicine is prescribed.
4. 🏢 **Manufacturer & Availability**: Note who manufactures it and its prescription/OTC status.
5. 🔄 **Related / Alternative Medicines**: List alternative medicines in the Knowledge Graph that treat the same symptom or indication.

Rules:
- Base answers ONLY on the KNOWLEDGE GRAPH RECORDS provided below.
- Do NOT hallucinate unverified medical claims.
- Always include the mandatory medical disclaimer at the very end.

KNOWLEDGE GRAPH RECORDS:
{context}
"""


def _render_fallback_answer(matched_records: list[dict], source: str, language: str = "en") -> str:
    """Deterministic Multilingual Markdown answer when LLM API is rate-limited or unauthorized."""
    lang = (language or "en").lower().strip()

    if not matched_records:
        if lang == "te":
            return (
                "నాలెడ్జ్ గ్రాఫ్‌లో సరిపోలే మందుల వివరాలు లభించలేదు.\n\n"
                "దయచేసి ఉదాహరణకు *Amoxicillin*, *ఫీవర్ (Fever)*, *నొప్పి (Pain)* వంటి సాధారణ పదాలతో ప్రయత్నించండి.\n\n"
                "> ⚕️ ఈ సమాచారం కేవలం సమాచారం కొరకు మాత్రమే. వైద్యపరమైన నిర్ణయాలు తీసుకునే ముందు దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి."
            )
        elif lang == "hi":
            return (
                "नॉलेज ग्राफ में कोई मिलान वाली दवा का रिकॉर्ड नहीं मिला।\n\n"
                "कृपया *Amoxicillin*, *बुखार (Fever)*, *दर्द (Pain)* जैसे शब्दों से पुनः प्रयास करें।\n\n"
                "> ⚕️ यह जानकारी केवल संदर्भ के लिए है। कोई भी चिकित्सा निर्णय लेने से पहले कृपया किसी योग्य चिकित्सक से परामर्श लें।"
            )
        else:
            return (
                "I could not find matching medicine records in the Knowledge Graph.\n\n"
                "Please try a broader medicine name or symptom (e.g. *Amoxicillin*, *Fever*, *Pain*, *Infection*).\n\n"
                "> ⚕ This information is for reference only. Consult a qualified healthcare professional before making any medical decisions."
            )

    if lang == "te":
        lines = [f"### 👨‍⚕️ డాక్టర్ AI క్లినికల్ అసెస్మెంట్ (మూలం: {source})\n"]
        for item in matched_records:
            m = item["medicine"]
            alts = item.get("alternatives", [])
            lines.append(
                f"#### **{m.get('name', 'అజ్ఞాత')}** ({m.get('classification', 'N/A')})\n"
                f"- 📌 **వర్గీకరణ (What it is)**: {m.get('category', 'N/A')}\n"
                f"- ❓ **మనం దీనిని ఎందుకు ఉపయోగిస్తాము (Why use)**: {m.get('indication', 'N/A')} చికిత్స కోసం సూచించబడింది\n"
                f"- 🌡️ **ఎప్పుడు ఉపయోగించాలి / లక్షణాలు (When to use)**: {m.get('indication', 'N/A')} లక్షణాలు అనుభవించినప్పుడు\n"
                f"- 💊 **రూపం మరియు మోతాదు (Dosage Form & Strength)**: {m.get('dosage_form', 'N/A')}, {m.get('strength', 'N/A')}\n"
                f"- 🏢 **తయారీదారు (Manufacturer)**: {m.get('manufacturer', 'N/A')}\n"
            )
            if alts:
                lines.append(f"- 🔄 **సరిపోలే ఇతర మందులు (Alternatives)**: {', '.join(alts)}\n")
        lines.append(
            "\n> ⚕️ ఈ సమాచారం కేవలం సమాచారం కొరకు మాత్రమే. వైద్యపరమైన నిర్ణయాలు తీసుకునే ముందు దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి."
        )

    elif lang == "hi":
        lines = [f"### 👨‍⚕️ डॉक्टर AI क्लिनिकल मूल्यांकन (स्रोत: {source})\n"]
        for item in matched_records:
            m = item["medicine"]
            alts = item.get("alternatives", [])
            lines.append(
                f"#### **{m.get('name', 'अज्ञात')}** ({m.get('classification', 'N/A')})\n"
                f"- 📌 **विवरण एवं श्रेणी (What it is)**: {m.get('category', 'N/A')}\n"
                f"- ❓ **हम इसका उपयोग क्यों करते हैं (Why use)**: {m.get('indication', 'N/A')} के इलाज के लिए निर्धारित\n"
                f"- 🌡️ **इसका उपयोग कब करना चाहिए / लक्षण (When to use)**: {m.get('indication', 'N/A')} के लक्षण होने पर\n"
                f"- 💊 **खुराक और रूप (Dosage Form & Strength)**: {m.get('dosage_form', 'N/A')}, {m.get('strength', 'N/A')}\n"
                f"- 🏢 **निर्माता (Manufacturer)**: {m.get('manufacturer', 'N/A')}\n"
            )
            if alts:
                lines.append(f"- 🔄 **अन्य वैकल्पिक दवाएं (Alternatives)**: {', '.join(alts)}\n")
        lines.append(
            "\n> ⚕️ यह जानकारी केवल संदर्भ के लिए है। कोई भी चिकित्सा निर्णय लेने से पहले कृपया किसी योग्य चिकित्सक से परामर्श लें।"
        )

    else:
        lines = [f"### 👨‍⚕️ Doctor AI Clinical Assessment (Source: {source})\n"]
        for item in matched_records:
            m = item["medicine"]
            alts = item.get("alternatives", [])
            lines.append(
                f"#### **{m.get('name', 'Unknown')}** ({m.get('classification', 'N/A')})\n"
                f"- 📌 **Overview & Classification**: {m.get('category', 'N/A')}\n"
                f"- ❓ **Why Do We Use It**: Prescribed for {m.get('indication', 'N/A')}\n"
                f"- 🌡️ **When Should You Use It (Symptoms)**: When experiencing {m.get('indication', 'N/A')} symptoms\n"
                f"- 💊 **Dosage Form & Strength**: {m.get('dosage_form', 'N/A')}, {m.get('strength', 'N/A')}\n"
                f"- 🏢 **Manufacturer**: {m.get('manufacturer', 'N/A')}\n"
            )
            if alts:
                lines.append(f"- 🔄 **Related Alternative Medicines**: {', '.join(alts)}\n")
        lines.append(
            "\n> ⚕ This information is for reference only. Consult a qualified healthcare professional before making any medical decisions."
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEMINI GENERATION ENGINE & PUBLIC RAG ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(system_prompt: str, question: str) -> str:
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY not configured."

    for model_name in MODEL_CANDIDATES:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log.info(f"Calling Gemini {model_name} (attempt {attempt})...")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt
                )
                res = model.generate_content(question)
                return res.text
            except ResourceExhausted:
                log.warning(f"Rate limited on {model_name}. Retrying...")
                time.sleep(RETRY_BASE_SEC * attempt)
            except Exception as exc:
                err_str = str(exc).lower()
                log.warning(f"Error on {model_name}: {exc}")
                if "api key not valid" in err_str or "api_key_invalid" in err_str:
                    return "Gemini API key is invalid or unauthorized."
                break

    return "Gemini rate-limited or unavailable."


def ask_agent(question: str, language: str = "en") -> str:
    """Full Multilingual GraphRAG entry point."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    lang_map = {
        "en": "English",
        "te": "Telugu (తెలుగు)",
        "hi": "Hindi (हिन्दी)"
    }
    target_lang_name = lang_map.get(language.lower(), "English")

    # 1. Retrieve Knowledge Graph Subgraph
    matched = search_graph(question)
    source = "Neo4j Knowledge Graph" if _driver else "Embedded Knowledge Graph Cache"

    # 2. Fallback to openFDA if empty
    if not matched:
        matched = search_openfda(question)
        if matched:
            source = "openFDA API"

    # 3. Context & Multilingual System Prompt Construction
    context = _build_context(matched)
    system_prompt = _SYSTEM_PROMPT.format(context=context) + f"\n\nIMPORTANT LANGUAGE INSTRUCTION:\nRespond strictly in {target_lang_name}. Translate clinical explanations and symptoms clearly into {target_lang_name}."

    # 4. Generate LLM Answer
    answer = _call_gemini(system_prompt, question)

    # 5. Deterministic fallback if LLM call failed or rate limited
    lowered = answer.lower()
    if any(err in lowered for err in ["rate-limited", "unavailable", "gemini_api_key not", "error on", "invalid or unauthorized", "unauthorized"]):
        log.info(f"Using deterministic Graph Markdown renderer ({language}).")
        return _render_fallback_answer(matched, source, language=language)

    return answer


def keep_alive_ping() -> bool:
    """Executes a lightweight write query to Neo4j to keep Cloud AuraDB active and prevent auto-pause."""
    global _driver
    if _driver:
        try:
            with _driver.session() as session:
                session.run("MERGE (h:SystemHeartbeat {id: 'neo4j_keepalive'}) SET h.last_active = datetime()")
            log.info("⚡ Neo4j Keep-Alive heartbeat write ping executed successfully.")
            return True
        except Exception as exc:
            log.warning(f"Neo4j keep-alive ping warning: {exc}")
            return False
    return False



def close_driver():
    global _driver
    if _driver:
        try:
            _driver.close()
            log.info("Neo4j driver closed.")
        except Exception:
            pass

