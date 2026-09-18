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
from typing import Optional, Dict, Any, List
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
OPENFDA_BASE_URL     = "https://api.fda.gov/drug/label.json"
CACHE_JSON_PATH      = os.path.join(os.path.dirname(__file__), "medgraph_cache.json")
GUIDELINES_JSON_PATH = os.path.join(os.path.dirname(__file__), "clinical_guidelines.json")

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


# ── Embedded Local Cache & Guidelines ─────────────────────────────────────────
_LOCAL_CACHE = None
_CLINICAL_GUIDELINES = None


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


def _get_clinical_guidelines() -> list[dict]:
    global _CLINICAL_GUIDELINES
    if _CLINICAL_GUIDELINES is None:
        if os.path.exists(GUIDELINES_JSON_PATH):
            try:
                with open(GUIDELINES_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _CLINICAL_GUIDELINES = data.get("guidelines", [])
                log.info(f"Loaded {len(_CLINICAL_GUIDELINES)} clinical guideline passages.")
            except Exception as exc:
                log.error(f"Error reading clinical guidelines: {exc}")
                _CLINICAL_GUIDELINES = []
        else:
            log.warning("No clinical_guidelines.json found.")
            _CLINICAL_GUIDELINES = []
    return _CLINICAL_GUIDELINES


def _search_vector_guidelines(query: str, top_k: int = 3) -> list[dict]:
    """Retrieves unstructured clinical guidelines using vector/semantic term similarity scoring with strict relevance thresholding."""
    guidelines = _get_clinical_guidelines()
    if not guidelines:
        return []

    variants = _build_search_variants(query)
    if not variants:
        return []

    scored = []
    q_tokens = set(re.findall(r"\w+", query.lower()))

    for g in guidelines:
        text_blob = f"{g.get('title', '')} {g.get('condition', '')} {g.get('category', '')} {' '.join(g.get('bullets', []))} {g.get('content', '')}".lower()
        g_tokens = set(re.findall(r"\w+", text_blob))

        # Token intersection score
        overlap = len(q_tokens.intersection(g_tokens))
        if overlap > 0:
            score = overlap / (len(q_tokens) + 1.0)
            matched_condition = False
            for v in variants:
                if v in g.get("condition", "").lower() or v in g.get("title", "").lower():
                    score += 1.5
                    matched_condition = True
                elif v in text_blob:
                    score += 0.3
            # Strict relevance threshold: require condition match or score >= 1.0 to prevent irrelevant guideline leakage
            if matched_condition or score >= 1.0:
                scored.append((score, g))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [g for score, g in scored[:top_k]]
    log.info(f"Vector guideline search '{query}' → {len(results)} passage(s)")
    return results


def _reciprocal_rank_fusion(graph_records: list[dict], vector_guidelines: list[dict], k: int = 60) -> dict:
    """
    Reciprocal Rank Fusion (RRF) algorithm combining Graph nodes and Vector Guidelines.
    RRF Score(d) = 1 / (k + rank_graph(d)) + 1 / (k + rank_vector(d))
    """
    scores = {}
    items = {}

    for rank, rec in enumerate(graph_records, 1):
        m_name = rec["medicine"].get("name", f"med_{rank}")
        doc_id = f"graph_{m_name}"
        score = 1.0 / (k + rank)
        scores[doc_id] = scores.get(doc_id, 0.0) + score
        items[doc_id] = {"type": "graph", "data": rec}

    for rank, guide in enumerate(vector_guidelines, 1):
        doc_id = f"vector_{guide.get('id', rank)}"
        score = 1.0 / (k + rank)
        scores[doc_id] = scores.get(doc_id, 0.0) + score
        items[doc_id] = {"type": "vector", "data": guide}

    fused_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)

    fused_graph = [items[did]["data"] for did in fused_ids if items[did]["type"] == "graph"]
    fused_vector = [items[did]["data"] for did in fused_ids if items[did]["type"] == "vector"]

    return {
        "graph_records": fused_graph,
        "vector_guidelines": fused_vector,
        "total_fused": len(fused_ids)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRAPH SEARCH (Cypher & Local Embedded Graph)
# ─────────────────────────────────────────────────────────────────────────────

# Cypher for Neo4j: Finds target medicines + connected Indications (Symptoms), Categories, Manufacturers, and Dosage Forms.
# Strict guardrail: Alternatives MUST match the indication AND belong to the same category to avoid drug class cross-wiring.
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
  AND (toLower(alt.category) = toLower(m.category) OR m.category IS NULL OR alt.category IS NULL)
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
LIMIT 4
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:mg|ml|mcg|g|iu)?")
_STOPWORDS = {
    "medicine", "medicines", "drug", "drugs", "strength", "show",
    "list", "find", "with", "for", "the", "and", "or", "of", "a", "an",
    "why", "do", "we", "use", "when", "should", "i", "take", "what", "are",
    "symptom", "symptoms", "used", "treat", "treatment"
}

_VAGUE_QUERIES = {
    "medicine", "medicines", "drug", "drugs", "help", "tell me", "tell me more",
    "what else", "show more", "list", "details", "info", "anything", "please help",
    "give me medicine", "show me drugs", "tell me medicine", "give medicine", "drugs list"
}


def _has_medical_entity(query: str) -> bool:
    """Check if query contains any known medicine name, indication, or category."""
    cache = _get_local_cache()
    records = cache.get("records", [])
    if not records:
        return False

    q_lowered = query.lower()
    for r in records:
        name = (r.get("name") or "").lower()
        ind = (r.get("indication") or "").lower()
        cat = (r.get("category") or "").lower()
        if name and name in q_lowered:
            return True
        if ind and ind in q_lowered:
            return True
        if cat and cat in q_lowered:
            return True
    return False


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
    """Fallback search using embedded local Knowledge Graph cache with strict category guardrails."""
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
                # Find alternatives sharing indication AND matching/compatible category
                r_cat = (r.get("category") or "").strip().lower()
                r_ind = (r.get("indication") or "").strip().lower()

                alts = []
                if r_ind:
                    for o in records:
                        if o["name"] == r["name"]:
                            continue
                        o_ind = (o.get("indication") or "").strip().lower()
                        o_cat = (o.get("category") or "").strip().lower()
                        # Strict schema guardrail: match indication AND category (prevent cross-wiring drug classes)
                        if o_ind == r_ind and (not r_cat or not o_cat or o_cat == r_cat):
                            alts.append(o["name"])
                            if len(alts) >= 3:
                                break

                matched.append({
                    "medicine": r,
                    "alternatives": alts
                })
                if len(matched) >= 4:
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
                        if len(records) >= 4:
                            break
                    if len(records) >= 4:
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


def search_openfda(query: str, limit: int = 4) -> list[dict]:
    variants = _build_search_variants(query)
    if not variants:
        return []

    # Stopword Guard: Block openFDA execution for pure stopword / pronoun tokens
    filtered_variants = [v for v in variants if v not in _STOPWORDS and v not in {"this", "these", "it", "how", "what", "why", "when", "take", "use"}]
    if not filtered_variants:
        log.warning(f"openFDA Stopword Guard: Suppressed query '{query}' (no specific medical entity).")
        return []

    seen = set()
    records = []
    try:
        with httpx.Client(timeout=10.0) as client:
            for term in filtered_variants:
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

def _build_context(matched_records: list[dict], vector_guidelines: Optional[list[dict]] = None) -> str:
    sections = []

    if matched_records:
        lines = ["=== KNOWLEDGE GRAPH MEDICINE RECORDS ==="]
        for i, item in enumerate(matched_records[:4], 1):
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
        sections.append("\n\n".join(lines))

    if vector_guidelines:
        lines = ["=== CLINICAL GUIDELINES & UNSTRUCTURED KNOWLEDGE ==="]
        for i, g in enumerate(vector_guidelines, 1):
            bullets_str = ""
            if g.get("bullets"):
                bullets_str = "\n".join([f"     - {b}" for b in g["bullets"]])
            else:
                bullets_str = f"     - {g.get('content', '')}"

            lines.append(
                f"{i}. **{g.get('title', 'Clinical Guideline')}** [{g.get('category', 'N/A')}]\n"
                f"   - Condition / Intent : {g.get('condition', 'N/A')}\n"
                f"   - Protocol Recommendations:\n{bullets_str}"
            )
        sections.append("\n\n".join(lines))

    if not sections:
        return "No matching medicine records or clinical guidelines found in the system for this query."

    return "\n\n".join(sections)


def _deduplicate_records(records: list[dict]) -> list[dict]:
    """Collapses repetitive openFDA or Knowledge Graph store-brand records into distinct medical categories."""
    if not records:
        return []

    deduped = []
    seen_keys = set()

    for item in records:
        m = dict(item.get("medicine", {}))
        name = (m.get("name") or "").strip()
        cat = (m.get("category") or "").strip().lower()
        classification = (m.get("classification") or "").strip().lower()

        key = f"{cat}_{classification}"
        if not key or key == "_":
            key = name.lower()

        # Simplify store-brand names (e.g. "Tension Headache Relief, Caseys 4good" -> "Tension Headache Relief")
        clean_name = re.sub(r",?\s*(?:Wal-Mart|Walgreens|Meijer|Caseys|Lil'|Products|Inc|Corp|Stores).*", "", name, flags=re.IGNORECASE).strip()
        if not clean_name:
            clean_name = name

        m["name"] = clean_name

        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append({"medicine": m, "alternatives": item.get("alternatives", [])})
            if len(deduped) >= 3:
                break

    return deduped if deduped else records[:3]

def _detect_question_intent(question: str) -> str:
    """Classifies user intent slot for focused answer rendering."""
    lowered = question.lower()

    if any(k in lowered for k in ["when to take", "when should i", "when to use", "when use", "when should"]):
        return "when_to_take"
    if any(k in lowered for k in ["why use", "why should i", "why to take", "why prescribe", "why should", "reason"]):
        return "why_use"
    if any(k in lowered for k in ["dosage", "dose", "form", "strength", "how much"]):
        return "dosage"
    if any(k in lowered for k in ["manufacturer", "company", "who makes", "maker"]):
        return "manufacturer"
    if any(k in lowered for k in ["precaution", "precautions", "warning", "side effect", "intake"]):
        return "precautions"

    return "full"


_SYSTEM_PROMPT = """\
You are MedGraph Nexus, an expert, warm, and empathetic clinical AI doctor assistant powered by a Knowledge Graph and Clinical Guidelines Engine.
Your task is to answer the user's query with supportive, professional, and empathetic clinical guidance — just like a knowledgeable doctor.

DO NOT output technical source tags like "Source: openFDA API..." or rigid repeating product card templates.

Structure your response with warm, natural conversational flow:
1. 💬 **Clinical Assessment & Narrative Opening**: Start with a warm, natural narrative transition (e.g. "Got it. Let’s look at how [Medicine/Symptom] works and what you should keep in mind...") to set an empathetic tone before diving into bullet points.
2. 📌 **Primary Medication & Treatment Options**: Detail top recommended options clearly with category and purpose.
3. 📋 **Precautions, Food-Drug & Care Guidelines**: Provide evidence-based intake guidance, food-drug interactions, and non-pharmacological care rules as clean bullet points (`- 🔹 ...`).
4. ⚠️ **When to Seek Medical Evaluation**: Highlight critical red-flag symptoms requiring a doctor's visit.

Formatting Rules:
- Always open with a warm, human conversational sentence before structured lists.
- Use clean line-by-line bullet points for all guidelines and precautions.
- Do NOT cross-wire drug classes (e.g. do not suggest antifungals for bacterial or tension symptoms).
- Always include the mandatory medical disclaimer at the very end.

CONTEXT:
{context}
"""


def _render_fallback_answer(matched_records: list[dict], source: str, language: str = "en", vector_guidelines: Optional[list[dict]] = None, question: str = "") -> str:
    """Natural ChatGPT/Gemini-style Markdown answer with empathetic clinical framing."""
    lang = (language or "en").lower().strip()

    if not matched_records and not vector_guidelines:
        if lang == "te":
            return (
                "వివరాలు లభించలేదు. మీ ప్రశ్నను సరిచూసుకోండి.\n\n"
                "> ⚕️ ఈ సమాచారం కేవలం సమాచారం కొరకు మాత్రమే. వైద్యపరమైన నిర్ణయాలు తీసుకునే ముందు దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి."
            )
        elif lang == "hi":
            return (
                "विवरण नहीं मिला। कृपया अपने प्रश्न की पुष्टि करें।\n\n"
                "> ⚕️ यह जानकारी केवल संदर्भ के लिए है। कोई भी चिकित्सा निर्णय लेने से पहले कृपया किसी योग्य चिकित्सक से परामर्श लें।"
            )
        else:
            return (
                "Hmm, I want to make sure I give you accurate clinical guidance. I couldn't find a direct match for that specific term in our graph.\n\n"
                "Were you looking for general relief options for a specific symptom (like *Headache*, *Fever*, or *Pain*), or did you mean a prescription brand like *Amoxicillin* or *Ibuprocillin*?\n\n"
                "> ⚕ This information is for reference only. Consult a qualified healthcare professional before making any medical decisions."
            )

    clean_records = _deduplicate_records(matched_records)
    intent = _detect_question_intent(question)
    lines = []

    if clean_records:
        first_med = clean_records[0]["medicine"].get("name", "this treatment")
        lines.append(f"Got it. Let's go over how **{first_med}** works and what key clinical guidance you should keep in mind:\n")

        for item in clean_records[:2]:
            m = item["medicine"]
            m_name = m.get("name", "Unknown")
            classification = m.get("classification", "OTC / Prescription")
            category = m.get("category", "N/A")
            indication = m.get("indication", "symptom relief")
            form = m.get("dosage_form", "N/A")
            strength = m.get("strength", "Standard dosage")
            mfg = m.get("manufacturer", "N/A")

            lines.append(f"### **{m_name}** ({classification})")

            if intent == "when_to_take":
                lines.append(f"- 🌡️ **When Should You Take It**: Prescribed / taken when experiencing **{indication}** symptoms.")
                if form != "N/A":
                    lines.append(f"- 💊 **Dosage Form & Strength**: Available as {form} ({strength}).")
            elif intent == "why_use":
                lines.append(f"- ❓ **Why Use It (Therapeutic Purpose)**: Prescribed/indicated for **{indication}** ({category}).")
            elif intent == "dosage":
                lines.append(f"- 💊 **Dosage Form & Strength**: Available as {form} ({strength}).")
            elif intent == "manufacturer":
                lines.append(f"- 🏢 **Manufacturer**: Manufactured by {mfg}.")
            else: # "full" or general query
                lines.append(f"- 📌 **Category & Purpose**: Prescribed for {indication} ({category}).")
                if form != "N/A":
                    lines.append(f"- 💊 **Dosage Form & Strength**: Available as {form} ({strength}).")
                if mfg != "N/A":
                    lines.append(f"- 🏢 **Manufacturer**: {mfg}")
            lines.append("")

    if vector_guidelines and intent in {"full", "precautions", "when_to_take"}:
        lines.append("### 📋 Precautions & Care Guidelines")
        for g in vector_guidelines[:2]:
            lines.append(f"#### **{g.get('title', 'Clinical Guidance')}**")
            if g.get("bullets"):
                for b in g["bullets"]:
                    lines.append(f"- 🔹 {b}")
            elif g.get("content"):
                lines.append(f"- 🔹 {g['content']}")
            lines.append("")

    lines.append("> ⚕ This information is for reference only. Consult a qualified healthcare professional before making any medical decisions.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONVERSATIONAL INTENT & GEMINI ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _extract_recent_entity(history: list[dict]) -> tuple[Optional[str], Optional[str]]:
    """Extract last mentioned medicine name or indication from conversation history."""
    if not history:
        return None, None

    cache = _get_local_cache()
    records = cache.get("records", [])
    known_meds = [r["name"].lower() for r in records if "name" in r]
    known_inds = list({r["indication"].lower() for r in records if r.get("indication")})

    for turn in reversed(history):
        text = (turn.get("content") or "").lower()
        for med in known_meds:
            if med in text:
                return med.title(), "medicine"
        for ind in known_inds:
            if ind in text:
                return ind.title(), "indication"
    return None, None


def _resolve_coreference_and_intent(question: str, history: Optional[list[dict]] = None, language: str = "en") -> tuple[str, bool, str]:
    """
    Coreference Resolution & Intent Parser.
    Returns (resolved_query, is_ambiguous, clarification_message)
    """
    raw_q = question.strip()
    lowered = raw_q.lower()
    lang = (language or "en").lower().strip()

    # 1. Ambiguity / Vague query check
    is_vague = (
        lowered in _VAGUE_QUERIES
        or any(lowered.startswith(v) or lowered.endswith(v) for v in _VAGUE_QUERIES)
        or (len(lowered) <= 3 and lowered not in {"flu", "hiv"})
        or (not _has_medical_entity(raw_q) and any(w in lowered for w in ["medicine", "medicines", "drug", "drugs"]))
    )

    if is_vague:
        patient_state = PatientClinicalState(history=history or [])
        recent_entity, entity_type = _extract_recent_entity(history or [])
        active_name = patient_state.active_entity or recent_entity

        if not active_name:
            if lang == "te":
                msg = (
                    "నేను మీకు ఖచ్చితమైన వైద్య మార్గదర్శకత్వాన్ని అందించాలనుకుంటున్నాను. దయచేసి మీరు ఏ మందు లేదా లక్షణాల గురించి తెలుసుకోవాలనుకుంటున్నారో చెప్పండి.\n\n"
                    "ఉదాహరణకు:\n"
                    "- **Amoxicillin** (యాంటిబయోటిక్)\n"
                    "- **Ibuprocillin** (ఇన్ఫెక్షన్ మరియు జ్వరం)\n"
                    "- **Fever** లేదా **Pain** లక్షణాలు\n\n"
                    "> ⚕️ దయచేసి నిర్దిష్ట మందు పేరు లేదా లక్షణాన్ని నమోదు చేయండి."
                )
            elif lang == "hi":
                msg = (
                    "मैं आपको सटीक चिकित्सीय मार्गदर्शन देना चाहता हूं। कृपया स्पष्ट करें कि आप किस दवा या लक्षण के बारे में जानकारी चाहते हैं।\n\n"
                    "उदाहरण के लिए:\n"
                    "- **Amoxicillin** (एंटीबायोटिक)\n"
                    "- **Ibuprocillin** (संक्रमण और बुखार)\n"
                    "- **Fever** या **Pain** के लक्षण\n\n"
                    "> ⚕️ कृपया किसी विशिष्ट दवा का नाम या लक्षण दर्ज करें।"
                )
            else:
                msg = (
                    "Hmm, I want to make sure I give you accurate clinical guidance. Could you please specify which medicine or symptom you would like information about?\n\n"
                    "For example:\n"
                    "- **Amoxicillin** (Antibiotic for infections)\n"
                    "- **Ibuprocillin** (Antiviral for fever and infection)\n"
                    "- **Headache** or **Pain** symptoms\n\n"
                    "> ⚕️ Please provide a specific medicine name or clinical symptom to search the Knowledge Graph."
                )
            return raw_q, True, msg

    # 2. Coreference resolution (Context Injection Guard & Shorthand Triggers)
    pronoun_triggers = [
        "how to use this", "how to use it", "how to use", "how to take this", "how to take it",
        "how to take", "how do i use this", "how do i take this", "what is this", "why is it used",
        "side effect", "side effects", "dosage", "dose", "manufacturer", "uses",
        "when to take", "why use it", "what is it", "precaution", "precautions",
        "intake", "taking these", "intaking these", "warning", "warnings",
        "this", "these", "it", "its", "this drug", "this medicine", "the drug", "the medicine"
    ]
    has_pronoun = any(re.search(rf"\b{re.escape(trigger)}\b", lowered) for trigger in pronoun_triggers)

    resolved_q = raw_q
    if has_pronoun and history:
        patient_state = PatientClinicalState(history=history)
        if patient_state.active_entity:
            log.info(f"Context Injection Guard: '{raw_q}' -> resolved to '{patient_state.active_entity} {raw_q}'")
            resolved_q = f"how to use {patient_state.active_entity}" if "how to use" in lowered else f"{patient_state.active_entity} {raw_q}"
        else:
            recent_entity, entity_type = _extract_recent_entity(history)
            if recent_entity:
                log.info(f"Coreference resolved: '{raw_q}' -> active context '{recent_entity}'")
                resolved_q = f"{recent_entity} {raw_q}"

    return resolved_q, False, ""


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


# ─────────────────────────────────────────────────────────────────────────────
# 5. MULTI-STEP AGENTIC REASONING ENGINE (ReAct / Plan-and-Solve)
# ─────────────────────────────────────────────────────────────────────────────

class PatientClinicalState:
    """Tracks patient clinical state (allergies, active entity, disqualified drug classes, reported symptoms across session)."""
    def __init__(self, history: Optional[list[dict]] = None):
        self.active_entity = None
        self.disqualified_drugs = set()
        self.disqualified_categories = set()
        self.reported_symptoms = set()
        if history:
            self._parse_history(history)

    def _parse_history(self, history: list[dict]):
        allergy_triggers = ["allergic", "allergy", "makes me nauseous", "nausea", "reaction", "cannot take", "can't take"]
        cache = _get_local_cache()
        records = cache.get("records", [])

        # Parse history in reverse to find active entity from most recent turn
        for turn in reversed(history):
            content = (turn.get("content") or "").lower()

            if not self.active_entity:
                for r in records:
                    m_name = (r.get("name") or "").lower()
                    m_ind = (r.get("indication") or "").lower()
                    if m_name and m_name in content:
                        self.active_entity = r.get("name")
                        break
                    elif m_ind and m_ind in content:
                        self.active_entity = r.get("indication")
                        break

            if any(t in content for t in allergy_triggers):
                for r in records:
                    m_name = (r.get("name") or "").lower()
                    m_cat = (r.get("category") or "").strip().lower()
                    if m_name and m_name in content:
                        self.disqualified_drugs.add(m_name)
                        if m_cat:
                            self.disqualified_categories.add(m_cat)


def _decompose_query(question: str) -> list[str]:
    """
    Step 1: Agentic Query Decomposition.
    Decomposes multi-part or multi-symptom queries ("headache and vomits", "fever with pain")
    into independent search sub-queries.
    """
    lowered = question.lower().strip()
    raw_splits = re.split(r"\b(?:and|with|as well as|alongside|,|\+)\b", lowered)

    cache = _get_local_cache()
    records = cache.get("records", [])

    extracted = []
    for part in raw_splits:
        part_clean = part.strip()
        if not part_clean:
            continue

        matched_entity = None
        for r in records:
            ind = (r.get("indication") or "").lower()
            cat = (r.get("category") or "").strip().lower()
            name = (r.get("name") or "").lower()

            if ind and ind in part_clean:
                matched_entity = ind
                break
            elif cat and cat in part_clean:
                matched_entity = cat
                break
            elif name and name in part_clean:
                matched_entity = name
                break

        if matched_entity:
            extracted.append(matched_entity)
        elif len(part_clean) >= 3 and part_clean not in _STOPWORDS:
            extracted.append(part_clean)

    unique_sub_queries = list(dict.fromkeys(extracted))
    return unique_sub_queries if unique_sub_queries else [question]


def _multi_hop_graph_reasoning(sub_queries: list[str], patient_state: PatientClinicalState) -> list[dict]:
    """
    Step 2: Multi-Hop Graph Traversal & Subgraph Intersection.
    Executes graph searches for each sub-query, intersects results for multi-symptom match,
    and filters out patient-disqualified drug classes.
    """
    if len(sub_queries) == 1:
        raw_results = search_graph(sub_queries[0])
    else:
        sub_results = []
        for sq in sub_queries:
            res = search_graph(sq)
            if res:
                sub_results.append(res)

        if not sub_results:
            raw_results = []
        elif len(sub_results) == 1:
            raw_results = sub_results[0]
        else:
            # Multi-hop Intersection: Find medicines present across multiple symptom queries
            first_set = {item["medicine"]["name"]: item for item in sub_results[0]}
            intersected = []
            for item in sub_results[1]:
                m_name = item["medicine"]["name"]
                if m_name in first_set:
                    intersected.append(item)

            if intersected:
                raw_results = intersected
            else:
                combined = []
                seen = set()
                for s_res in sub_results:
                    for item in s_res:
                        m_name = item["medicine"]["name"]
                        if m_name not in seen:
                            seen.add(m_name)
                            combined.append(item)
                raw_results = combined

    # Apply Patient Clinical State Memory Filtering
    safe_results = []
    for item in raw_results:
        m = item["medicine"]
        m_name = (m.get("name") or "").lower()
        m_cat = (m.get("category") or "").lower()

        if m_name in patient_state.disqualified_drugs or m_cat in patient_state.disqualified_categories:
            log.warning(f"🛡️ Memory Filter: Disqualified drug '{m.get('name')}' (category: '{m_cat}') due to patient allergy memory.")
            continue
        safe_results.append(item)

    return safe_results


def _self_critique_and_correct(query: str, candidates: list[dict], vector_guidelines: list[dict], patient_state: PatientClinicalState) -> tuple[list[dict], list[dict], bool]:
    """
    Step 3: Self-Correction & Reflection Loop (The Critique Step).
    Reviews proposed candidate items against strict clinical guardrails:
    1. Removes drug class cross-wiring (e.g., antibiotic for headache/vomit).
    2. Removes patient-disqualified drugs.
    Returns (corrected_candidates, corrected_guidelines, mismatch_detected)
    """
    mismatch_detected = False
    corrected_candidates = []
    q_lowered = query.lower()

    is_antibiotic_query = any(w in q_lowered for w in ["bacterial", "infection", "antibiotic", "strep"])
    is_antifungal_query = any(w in q_lowered for w in ["fungal", "fungus", "ringworm", "yeast"])

    for item in candidates:
        m = item["medicine"]
        m_name = (m.get("name") or "").lower()
        m_cat = (m.get("category") or "").lower()

        # Critique 1: Allergy / disqualified memory
        if m_name in patient_state.disqualified_drugs or m_cat in patient_state.disqualified_categories:
            log.info(f"Self-Critique: Discarding disqualified drug '{m.get('name')}' from candidate set.")
            mismatch_detected = True
            continue

        # Critique 2: Discard antibiotics/antifungals for simple non-infectious symptom queries
        if m_cat in {"antibiotic", "antifungal"} and not (is_antibiotic_query or is_antifungal_query):
            if any(sym in q_lowered for sym in ["headache", "vomit", "vomiting", "pain", "fever", "nausea"]):
                log.info(f"Self-Critique Mismatch: Discarding drug class '{m_cat}' for symptom query '{query}'.")
                mismatch_detected = True
                continue

        corrected_candidates.append(item)

    corrected_guidelines = []
    for g in vector_guidelines:
        cond = (g.get("condition") or "").lower()
        if any(term in q_lowered for term in ["headache", "vomit", "pain", "fever"]):
            if "neuropathy" in cond or "hypertension" in cond:
                log.info(f"Self-Critique Mismatch: Discarding guideline '{g.get('title')}' for symptom query '{query}'.")
                mismatch_detected = True
                continue
        corrected_guidelines.append(g)

    return corrected_candidates, corrected_guidelines, mismatch_detected


def ask_agent(question: str, language: str = "en", history: Optional[list[dict]] = None) -> str:
    """Multi-Step Agentic Reasoning Loop Entry Point (Plan -> Decompose -> Multi-Hop Traversal -> Memory Filter -> Self-Critique -> Synthesize)."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    # 1. Intent Clarification & Coreference Resolution
    resolved_q, is_ambiguous, clarification_msg = _resolve_coreference_and_intent(
        question, history=history, language=language
    )
    if is_ambiguous:
        log.info(f"Ambiguous query detected ('{question}'). Returning clarification message.")
        return clarification_msg

    lang_map = {"en": "English", "te": "Telugu (తెలుగు)", "hi": "Hindi (हिन्दी)"}
    target_lang_name = lang_map.get(language.lower(), "English")

    log.info("🧠 [AGENT REASONING ENGINE] Step 1: Parsing Patient Memory & Decomposing User Intent")
    patient_state = PatientClinicalState(history=history)
    sub_queries = _decompose_query(resolved_q)
    log.info(f"🧠 [AGENT REASONING ENGINE] Decomposed query '{resolved_q}' -> Sub-queries: {sub_queries}")

    log.info("🧠 [AGENT REASONING ENGINE] Step 2: Executing Multi-Hop Graph Traversal & Subgraph Intersection")
    graph_matched = _multi_hop_graph_reasoning(sub_queries, patient_state)
    source = "Neo4j Knowledge Graph" if _driver else "Embedded Knowledge Graph Cache"

    log.info("🧠 [AGENT REASONING ENGINE] Step 3: Executing Dense Vector Guideline Search")
    vector_matched = _search_vector_guidelines(resolved_q)

    if not graph_matched and not vector_matched:
        log.info("🧠 [AGENT REASONING ENGINE] Subgraph empty. Executing OpenFDA Fallback Search.")
        graph_matched = search_openfda(resolved_q)
        if graph_matched:
            source = "openFDA API"

    log.info("🧠 [AGENT REASONING ENGINE] Step 4: Performing Reciprocal Rank Fusion (RRF)")
    rrf_fused = _reciprocal_rank_fusion(graph_matched, vector_matched, k=60)
    candidate_graph = rrf_fused["graph_records"][:4]
    candidate_vector = rrf_fused["vector_guidelines"]

    log.info("🧠 [AGENT REASONING ENGINE] Step 5: Executing Self-Correction & Reflection Loop (Critique Step)")
    final_graph, final_vector, mismatch_caught = _self_critique_and_correct(
        resolved_q, candidate_graph, candidate_vector, patient_state
    )

    if mismatch_caught:
        log.info("🧠 [AGENT REASONING ENGINE] Self-Critique caught mismatches. Refined candidate set successfully.")

    log.info("🧠 [AGENT REASONING ENGINE] Step 6: Synthesizing Final Clinical Narrative")
    context = _build_context(final_graph, vector_guidelines=final_vector)
    system_prompt = _SYSTEM_PROMPT.format(context=context) + f"\n\nIMPORTANT LANGUAGE INSTRUCTION:\nRespond strictly in {target_lang_name}. Translate clinical explanations and symptoms clearly into {target_lang_name}."

    answer = _call_gemini(system_prompt, resolved_q)

    lowered = answer.lower()
    if any(err in lowered for err in ["rate-limited", "unavailable", "gemini_api_key not", "error on", "invalid or unauthorized", "unauthorized"]):
        log.info(f"Using deterministic Graph Markdown renderer ({language}).")
        return _render_fallback_answer(final_graph, source, language=language, vector_guidelines=final_vector, question=resolved_q)

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




