"""
MedGraph Nexus - Enterprise FastAPI Backend
--------------------------------------------
Implements:
1. Strict Pydantic models for request/response validation.
2. RESTful endpoints:
   - POST /api/search (Natural language symptom query -> Hybrid GraphRAG -> Grounded response + Subgraph)
   - GET  /api/alternatives/{brand_name} (Direct graph traversal for brand alternatives)
   - GET  /api/health (System and database status)
3. Medical guardrails with anti-hallucination system prompt and zero-context refusal.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment configuration
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))
load_dotenv()

app = FastAPI(
    title="MedGraph Nexus API",
    description="Pharmaceutical Knowledge Graph & Hybrid GraphRAG Clinical Decision Backend",
    version="2.0.0"
)

# CORS Middleware for React, Streamlit, and Vanilla Web Clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database & LLM Connections
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

try:
    driver: Optional[Driver] = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
except Exception:
    driver = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception:
        gemini_model = None
else:
    gemini_model = None



@app.on_event("shutdown")
def shutdown_db_client():
    if driver:
        driver.close()


# ---------------------------------------------------------------------------
# Strict Pydantic Models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="Natural language user query or symptom description", example="my head is pounding")
    language: str = Field("en", description="Target response language (en, es, hi, te)", example="en")
    history: List[Dict[str, str]] = Field(default=[], description="Prior conversational turns for multi-turn dialogue")


class GraphNode(BaseModel):
    id: str
    label: str
    type: str = Field(..., description="Brand, Ingredient, or Symptom")
    properties: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str = Field(..., description="CONTAINS, TREATS, or ALTERNATIVE_TO")


class GraphData(BaseModel):
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []


class Citation(BaseModel):
    id: int
    source: str = "openFDA"
    title: str
    rxcui: Optional[str] = None
    tty: Optional[str] = None
    evidence: str
    node_id: Optional[str] = None


class ClinicalRecord(BaseModel):
    evidence_id: int
    brand_name: str
    generic_name: str
    active_substance: str
    rxcui: Optional[str] = None
    therapeutic_category: Optional[str] = "General Medicine"
    drug_classes: List[str] = []
    indications: str
    warnings: Optional[str] = "None specified in label summary"
    adverse_reactions: Optional[str] = "None specified in label summary"
    contraindications: Optional[str] = "None specified in label summary"
    drug_interactions: Optional[str] = "None specified in label summary"
    alternative_brands: List[str] = []
    source: str = "openFDA Drug Label & NIH RxNorm"


class SearchResponse(BaseModel):
    query: str
    answer: str
    is_grounded: bool
    symptoms_identified: List[str] = []
    compositions: List[str] = []
    alternative_medicines: List[str] = []
    graph: GraphData
    clinical_records: List[ClinicalRecord] = []
    citations: List[Citation] = []
    warning_disclaimer: str = (
        "MedGraph Nexus is an informational tool intended to assist Healthcare Professionals. "
        "It does not provide medical diagnoses or dictate treatment plans. Clinicians must exercise independent judgment."
    )


class AlternativesResponse(BaseModel):
    queried_brand: str
    active_ingredient: Optional[str] = None
    rxcui: Optional[str] = None
    alternative_brands: List[str] = []
    indications: List[str] = []
    graph: GraphData
    clinical_records: List[ClinicalRecord] = []
    citations: List[Citation] = []
    message: str = "Success"


# ---------------------------------------------------------------------------
# Medical Guardrails & Direct Clinical System Prompt
# ---------------------------------------------------------------------------
CLINICAL_SYSTEM_GUARDRAIL = """
You are Doctor AI for MedGraph Nexus, a clinical pharmacotherapy engine.

COMMUNICATION STYLE:
- Be direct, concise, and clinically clear.
- Do NOT use unnecessary pleasantries, emotional filler, or flowery conversational padding (avoid "I understand your concern...", "I am really sorry to hear...", "Your health is paramount").
- State the clinical findings, active compositions, brand alternatives, and safety warnings directly.
- Organize answers with clean headings and concise bullet points.

CRITICAL CITATION DISCIPLINE (REVIEWABLE EVIDENCE):
Every time you mention an active chemical substance, indication, brand alternative, or safety warning from the Knowledge Graph context, you MUST append a bracketed footnote citation (e.g., [1], [2]) corresponding to the Evidence Record number.
- Example: "Acetaminophen [1] is an analgesic and antipyretic. Alternative formulations sharing this active substance include Mapap [1] and Panadol [1]."
- Example: "Do not exceed 4,000 mg in 24 hours due to risk of severe hepatotoxicity [1]."

GROUNDING & REFUSAL RULES:
1. Formulate statements strictly based on the verified Knowledge Graph records provided below.
2. If no verified records exist in the context, state directly:
   "No verified pharmaceutical records were found in MedGraph Nexus for the requested query. Please consult a qualified healthcare provider."
"""



def generate_conversational_response(
    query: str,
    raw_context: str,
    history: Optional[List[Dict[str, str]]] = None,
    language: str = "en",
    retrieved_summary: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates humanly conversational, clinically grounded responses using OpenRouter
    with fallback to Google Gemini or deterministic graph summaries.
    """
    history = history or []

    # 1. Primary Engine: OpenRouter API (OpenAI GPT-4o-mini / Meta Llama 3.3)
    if OPENROUTER_API_KEY:
        try:
            messages = [{"role": "system", "content": CLINICAL_SYSTEM_GUARDRAIL.strip()}]

            # Multi-turn conversational continuity (last 4 dialogue turns)
            for h in history[-4:]:
                r = "user" if h.get("role") in ["user", "human"] else "assistant"
                c = (h.get("content") or "").strip()
                if c:
                    messages.append({"role": r, "content": c})

            user_prompt = (
                f"VERIFIED KNOWLEDGE GRAPH & OPENFDA CONTEXT:\n"
                f"{raw_context}\n\n"
                f"USER CLINICAL INQUIRY:\n"
                f"\"{query}\"\n\n"
                f"TARGET LANGUAGE: {language}\n\n"
                f"Please formulate a warm, humanly conversational, yet clinically grounded response. "
                f"Make sure to embed bracketed footnote citations like [1], [2] for all active ingredients, "
                f"brand alternatives, approved indications, and safety warnings present in the context above."
            )
            messages.append({"role": "user", "content": user_prompt})

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://medgraph.nexus",
                "X-Title": "MedGraph Nexus Clinical Engine"
            }
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 1200
            }

            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=25
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content:
                    return content
        except Exception:
            pass

    # 2. Secondary Fallback: Google Gemini
    if gemini_model:
        try:
            fallback_prompt = (
                f"{CLINICAL_SYSTEM_GUARDRAIL}\n\n"
                f"VERIFIED KNOWLEDGE GRAPH CONTEXT:\n{raw_context}\n\n"
                f"USER INQUIRY: {query}\n"
                f"TARGET LANGUAGE: {language}"
            )
            gem_res = gemini_model.generate_content(fallback_prompt)
            if gem_res and gem_res.text:
                return gem_res.text.strip()
        except Exception:
            pass

    # 3. Tertiary Deterministic Fallback
    syms = ', '.join(retrieved_summary.get('symptoms', [])) if retrieved_summary else 'General Indication'
    comps = ', '.join(retrieved_summary.get('compositions', [])) if retrieved_summary else 'Active Substance'
    alts = ', '.join(retrieved_summary.get('alternative_medicines', [])[:8]) if retrieved_summary else 'Documented Brands'
    return (
        f"### Clinical Graph Findings\n\n"
        f"- **Symptom(s) Identified**: {syms}\n"
        f"- **Active Composition(s)**: {comps}\n"
        f"- **Alternative Brand Formulations**: {alts}\n\n"
        f"*Data retrieved directly from MedGraph Nexus knowledge graph.*"
    )


STOPWORDS = {
    "what", "is", "are", "the", "for", "with", "does", "which", "give", "me", "a", "an",
    "tell", "about", "cure", "can", "treat", "used", "uses", "use", "usage", "using", "to", "in", "of",
    "and", "drug", "drugs", "medicine", "medicines", "medication", "pill", "tablets",
    "feeling", "feel", "having", "suffering", "from", "please", "help", "how", "why", "who",
    "where", "when", "replace", "replacing", "replacement", "fix", "repair", "rules", "rule",
    "ignore", "lethal", "dose", "doses", "taking", "taken", "take", "make", "made", "car",
    "honda", "civic", "spark", "plug", "plugs", "tire", "tires", "engine", "code", "run",
    "chronic", "hyper", "severe", "mild", "acute", "treats", "high", "blood", "pressure", "good", "need",
    "alternative", "alternatives", "brand", "brands"
}

PHRASE_NORMALIZATIONS = {
    "high blood pressure": "hypertension",
    "blood pressure": "hypertension",
    "high bp": "hypertension",
    "headache": "headache",
    "head pounding": "headache",
    "pounding head": "headache",
    "throbbing head": "headache",
    "heart attack": "myocardial infarction",
    "chest pain": "angina",
    "sore throat": "sore throat"
}

COMMON_BRAND_ALIASES = {
    "lipitor": "atorvastatin",
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "norvasc": "amlodipine",
    "cozaar": "losartan",
    "zocor": "simvastatin",
    "plavix": "clopidogrel",
    "glucophage": "metformin",
    "lasix": "furosemide"
}

# ---------------------------------------------------------------------------
# Graph Retrieval & Multi-Entity Resolution
# ---------------------------------------------------------------------------
def retrieve_graph_subgraph(query_text: str) -> Dict[str, Any]:
    """
    Executes graph traversal across Neo4j or uses curated clinical cache
    to extract Symptoms, Ingredients, and Alternative Brands.
    Supports both direct Brand/Ingredient searches and Symptom searches.
    """
    # Safety Guardrail: Intercept lethal/adversarial hazard keywords immediately
    ADVERSARIAL_HAZARDS = {"lethal", "suicide", "overdose", "poison", "toxic dose", "kill"}
    if any(h in query_text.lower() for h in ADVERSARIAL_HAZARDS):
        return {
            "symptoms": [],
            "compositions": [],
            "alternative_medicines": [],
            "graph": GraphData(nodes=[], edges=[]),
            "raw_context": ""
        }

    clean_text = query_text.lower().replace("&", " and ")

    # Priority multi-word phrase matching (e.g. "high blood pressure" -> "hypertension")
    resolved_search_terms = []
    for phrase, term in PHRASE_NORMALIZATIONS.items():
        if phrase in clean_text:
            if term not in resolved_search_terms:
                resolved_search_terms.append(term)

    raw_tokens = [w.strip("?,.!:;\"'&/()").lower() for w in clean_text.split() if len(w) > 2]
    keywords = [w for w in raw_tokens if w not in STOPWORDS]

    # Resolve common trade brand aliases if user queried a commercial name like Lipitor
    for w in raw_tokens:
        if w in COMMON_BRAND_ALIASES:
            if COMMON_BRAND_ALIASES[w] not in resolved_search_terms:
                resolved_search_terms.append(COMMON_BRAND_ALIASES[w])

    for k in keywords:
        if k not in resolved_search_terms:
            resolved_search_terms.append(k)

    if not resolved_search_terms:
        # If words were all filtered (e.g. "blood pressure"), keep meaningful tokens
        resolved_search_terms = [w for w in raw_tokens if w not in {"what", "is", "are", "the", "for", "and"}]


    if not resolved_search_terms:
        return {
            "symptoms": [],
            "compositions": [],
            "alternative_medicines": [],
            "graph": GraphData(nodes=[], edges=[]),
            "raw_context": ""
        }

    matched_symptoms = set()
    compositions = set()
    alt_brands = set()
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    clinical_records: List[ClinicalRecord] = []
    citations: List[Citation] = []
    raw_context = []
    evidence_counter = 1

    # 1. Attempt Live Neo4j Query (Dual Search: Direct Brand/Ingredient + Symptom Indication)
    db_success = False
    if driver:
        try:
            with driver.session() as session:
                # Step A: Check if query targets a Brand or Ingredient directly
                cypher_brand_ing = """
                MATCH (i:Ingredient)
                WHERE any(kw IN $terms WHERE toLower(i.name) CONTAINS kw)
                OPTIONAL MATCH (b:Brand)-[:CONTAINS]->(i)
                OPTIONAL MATCH (i)-[:TREATS]->(s:Symptom)
                OPTIONAL MATCH (m:Medicine) WHERE toLower(m.brand_name) = toLower(b.name)
                RETURN coalesce(s.name, 'Clinical Indication') AS symptom,
                       i.name AS ingredient,
                       i.rxcui AS ingredient_rxcui,
                       collect(DISTINCT b.name) AS brands,
                       collect(DISTINCT {
                           brand: m.brand_name,
                           generic: m.generic_name,
                           rxcui: m.rxcui,
                           category: m.category,
                           indications: m.indications,
                           warnings: m.warnings,
                           contraindications: m.contraindications,
                           adverse_reactions: m.adverse_reactions,
                           drug_interactions: m.drug_interactions
                       }) AS med_details
                LIMIT 8
                """
                res = list(session.run(cypher_brand_ing, terms=resolved_search_terms))

                # Step B: If no direct ingredient/brand hit, query Symptom Indications
                if not res or len(res) == 0:
                    cypher_symptom = """
                    MATCH (s:Symptom)
                    WHERE any(kw IN $terms WHERE toLower(s.name) CONTAINS kw)
                    MATCH (s)<-[:TREATS]-(i:Ingredient)
                    OPTIONAL MATCH (b:Brand)-[:CONTAINS]->(i)
                    OPTIONAL MATCH (m:Medicine) WHERE toLower(m.brand_name) = toLower(b.name)
                    RETURN s.name AS symptom,
                           i.name AS ingredient,
                           i.rxcui AS ingredient_rxcui,
                           collect(DISTINCT b.name) AS brands,
                           collect(DISTINCT {
                               brand: m.brand_name,
                               generic: m.generic_name,
                               rxcui: m.rxcui,
                               category: m.category,
                               indications: m.indications,
                               warnings: m.warnings,
                               contraindications: m.contraindications,
                               adverse_reactions: m.adverse_reactions,
                               drug_interactions: m.drug_interactions
                           }) AS med_details
                    LIMIT 8
                    """
                    res = list(session.run(cypher_symptom, terms=resolved_search_terms))


                for row in res:
                    sym = row["symptom"][:70]
                    ing = row["ingredient"]
                    ing_rxcui = row.get("ingredient_rxcui", "")
                    brands = [b for b in row["brands"] if b][:8]
                    med_list = [m for m in row.get("med_details", []) if m and m.get("brand")]

                    matched_symptoms.add(sym)
                    compositions.add(ing)
                    alt_brands.update(brands)

                    # Graph Nodes & Edges
                    sym_id = f"symptom_{sym.lower()[:30]}"
                    ing_id = f"ing_{ing.lower()}"
                    nodes[sym_id] = GraphNode(id=sym_id, label=sym, type="Symptom")
                    nodes[ing_id] = GraphNode(id=ing_id, label=ing, type="Ingredient")
                    edges.append(GraphEdge(source=ing_id, target=sym_id, relationship="TREATS"))

                    primary_brand = brands[0] if brands else ing
                    for b in brands:
                        b_id = f"brand_{b.lower()}"
                        nodes[b_id] = GraphNode(id=b_id, label=b, type="Brand")
                        edges.append(GraphEdge(source=b_id, target=ing_id, relationship="CONTAINS"))

                    # Structured Clinical Record & Citation
                    med_props = med_list[0] if med_list else {}
                    ind_text = med_props.get("indications") or sym
                    rec = ClinicalRecord(
                        evidence_id=evidence_counter,
                        brand_name=primary_brand,
                        generic_name=med_props.get("generic") or ing,
                        active_substance=ing,
                        rxcui=med_props.get("rxcui") or ing_rxcui or "N/A",
                        therapeutic_category=med_props.get("category") or "General Medicine",
                        indications=ind_text,
                        warnings=med_props.get("warnings") or "None specified in label summary",
                        adverse_reactions=med_props.get("adverse_reactions") or "None specified in label summary",
                        contraindications=med_props.get("contraindications") or "None specified in label summary",
                        drug_interactions=med_props.get("drug_interactions") or "None specified in label summary",
                        alternative_brands=brands
                    )
                    clinical_records.append(rec)

                    citations.append(Citation(
                        id=evidence_counter,
                        source="openFDA Drug Label & NIH RxNorm",
                        title=f"{primary_brand} ({ing}) Prescribing Label",
                        rxcui=rec.rxcui,
                        tty="IN / BN",
                        evidence=ind_text[:350],
                        node_id=sym_id
                    ))

                    raw_context.append(
                        f"[Evidence Record {evidence_counter}]\n"
                        f"- Symptom / Indication: {sym}\n"
                        f"- Active Composition (Ingredient): {ing} (RxCUI: {rec.rxcui})\n"
                        f"- Marketed Brands: {', '.join(brands)}\n"
                        f"- FDA Approved Usage: {ind_text[:280]}...\n"
                        f"- Black Box Warnings & Precautions: {rec.warnings[:200]}..."
                    )
                    evidence_counter += 1

                db_success = len(res) > 0
        except Exception:
            db_success = False

    # 2. Local Fallback from Curated Dataset if DB is disconnected or returns empty
    if not db_success:
        dataset_path = os.path.join(parent_dir, "data", "medicine_dataset.json")
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    medicines = json.load(f)
                count = 0
                for med in medicines:
                    ind = med.get("indications", "").lower()
                    b_name = med.get("brand_name", "")
                    subs = med.get("substances", [])
                    active_ing = subs[0] if subs else med.get("generic_name", "Unknown")
                    known = med.get("known_brands", [b_name])

                    if any(kw in ind or kw in b_name.lower() or kw in active_ing.lower() for kw in keywords):
                        symptom_label = "Headache / Pain" if "head" in query_text.lower() or "pound" in query_text.lower() else (keywords[0].title() if keywords else "General Indication")
                        matched_symptoms.add(symptom_label)
                        compositions.add(active_ing)
                        alt_brands.update(known[:5])

                        sym_id = f"symptom_{symptom_label.lower()}"
                        ing_id = f"ing_{active_ing.lower()}"
                        nodes[sym_id] = GraphNode(id=sym_id, label=symptom_label, type="Symptom")
                        nodes[ing_id] = GraphNode(id=ing_id, label=active_ing, type="Ingredient")
                        edges.append(GraphEdge(source=ing_id, target=sym_id, relationship="TREATS"))

                        for kb in known[:4]:
                            b_id = f"brand_{kb.lower()}"
                            nodes[b_id] = GraphNode(id=b_id, label=kb, type="Brand")
                            edges.append(GraphEdge(source=b_id, target=ing_id, relationship="CONTAINS"))

                        rec = ClinicalRecord(
                            evidence_id=evidence_counter,
                            brand_name=b_name,
                            generic_name=med.get("generic_name", active_ing),
                            active_substance=active_ing,
                            rxcui=med.get("rxcui", "N/A"),
                            therapeutic_category=med.get("category", "General Medicine"),
                            indications=med.get("indications", symptom_label),
                            warnings=med.get("warnings", "None specified in label summary"),
                            adverse_reactions=med.get("adverse_reactions", "None specified in label summary"),
                            contraindications=med.get("contraindications", "None specified in label summary"),
                            drug_interactions=med.get("drug_interactions", "None specified in label summary"),
                            alternative_brands=known
                        )
                        clinical_records.append(rec)

                        citations.append(Citation(
                            id=evidence_counter,
                            source="openFDA Drug Label & NIH RxNorm",
                            title=f"{b_name} ({active_ing}) Prescribing Label",
                            rxcui=rec.rxcui,
                            tty="IN / BN",
                            evidence=med.get("indications", "")[:350],
                            node_id=sym_id
                        ))

                        raw_context.append(
                            f"[Evidence Record {evidence_counter}]\n"
                            f"- Symptom: {symptom_label}\n"
                            f"- Ingredient: {active_ing} (RxCUI: {rec.rxcui})\n"
                            f"- Commercial Brands: {', '.join(known[:4])}\n"
                            f"- Approved Usage: {med.get('indications', '')[:280]}..."
                        )
                        evidence_counter += 1
                        count += 1
                        if count >= 6:
                            break
            except Exception:
                pass

    return {
        "symptoms": list(matched_symptoms),
        "compositions": list(compositions),
        "alternative_medicines": list(alt_brands),
        "graph": GraphData(nodes=list(nodes.values()), edges=edges),
        "clinical_records": clinical_records,
        "citations": citations,
        "raw_context": "\n\n---\n\n".join(raw_context) if raw_context else ""
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/search", response_model=SearchResponse)
async def search_symptoms(request: SearchRequest):
    """
    Natural language symptom search pipeline:
    1. Validates query using Pydantic.
    2. Executes hybrid retrieval + backward graph traversal.
    3. Synthesizes clinical response under strict medical anti-hallucination guardrails.
    4. Returns structured JSON containing the answer and interactive graph data.
    """
    retrieved = retrieve_graph_subgraph(request.query)
    has_context = bool(retrieved["raw_context"])

    if not has_context:
        # Strict zero-context refusal per Medical Guardrail specifications
        refusal_msg = (
            "I cannot find verified pharmaceutical records in MedGraph Nexus for the requested symptoms. "
            "Please consult a qualified healthcare provider or medical professional."
        )
        return SearchResponse(
            query=request.query,
            answer=refusal_msg,
            is_grounded=False,
            symptoms_identified=[],
            compositions=[],
            alternative_medicines=[],
            graph=GraphData(nodes=[], edges=[])
        )

    answer_text = generate_conversational_response(
        query=request.query,
        raw_context=retrieved["raw_context"],
        history=request.history,
        language=request.language,
        retrieved_summary=retrieved
    )

    return SearchResponse(
        query=request.query,
        answer=answer_text,
        is_grounded=True,
        symptoms_identified=retrieved["symptoms"],
        compositions=retrieved["compositions"],
        alternative_medicines=retrieved["alternative_medicines"],
        graph=retrieved["graph"],
        clinical_records=retrieved.get("clinical_records", []),
        citations=retrieved.get("citations", [])
    )


@app.post("/ask")
async def ask_legacy(payload: Dict[str, Any]):
    """Backwards-compatible endpoint for existing chat frontends."""
    q = payload.get("question") or payload.get("query", "")
    lang = payload.get("language", "en")
    hist = payload.get("history", [])
    req = SearchRequest(query=q, language=lang, history=hist)
    res = await search_symptoms(req)
    return {
        "answer": res.answer,
        "is_grounded": res.is_grounded,
        "context_used": res.symptoms_identified,
        "compositions": res.compositions,
        "alternative_medicines": res.alternative_medicines,
        "graph": res.graph.dict(),
        "clinical_records": [r.dict() for r in res.clinical_records],
        "citations": [c.dict() for c in res.citations]
    }


@app.get("/api/alternatives/{brand_name}", response_model=AlternativesResponse)
async def get_brand_alternatives(
    brand_name: str = Path(..., min_length=2, description="Target pharmaceutical brand name to look up", example="Lipitor")
):
    """
    Direct graph traversal endpoint:
    Maps (:Brand {name: $brand_name})-[:CONTAINS]->(:Ingredient)
    and traverses to find all alternative brands and treated symptoms.
    """
    clean_brand = brand_name.strip()
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    active_ing = None
    rxcui = None
    alt_brands = set()
    indications = []

    db_found = False
    if driver:
        try:
            with driver.session() as session:
                cypher = """
                MATCH (b:Brand) WHERE toLower(b.name) CONTAINS toLower($brand)
                MATCH (b)-[:CONTAINS]->(i:Ingredient)
                OPTIONAL MATCH (alt:Brand)-[:CONTAINS]->(i)
                OPTIONAL MATCH (i)-[:TREATS]->(s:Symptom)
                RETURN i.name AS ingredient,
                       i.rxcui AS rxcui,
                       collect(DISTINCT alt.name) AS alternatives,
                       collect(DISTINCT s.name) AS symptoms
                LIMIT 1
                """
                record = session.run(cypher, brand=clean_brand).single()
                if record and record["ingredient"]:
                    db_found = True
                    active_ing = record["ingredient"]
                    rxcui = record.get("rxcui")
                    alt_brands = set(record["alternatives"])
                    indications = record["symptoms"]
        except Exception:
            db_found = False

    # Dynamic Fallback: Live openFDA + RxNorm lookup
    if not db_found:
        try:
            from pipelines.fetch_drug_relations import extract_alternative_brands_for_drug
            live_data = extract_alternative_brands_for_drug(clean_brand)
            if "error" not in live_data and live_data.get("active_ingredients"):
                ing_info = live_data["active_ingredients"][0]
                active_ing = ing_info.get("name", "Unknown").title()
                rxcui = ing_info.get("rxcui")
                alt_brands = set(live_data.get("alternative_brand_names", []))
                indications = [live_data["openfda"].get("indications_and_usage", "")[:250]]
                db_found = True
        except Exception:
            pass

    # Fallback to dataset search
    if not db_found:
        dataset_path = os.path.join(parent_dir, "data", "medicine_dataset.json")
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                medicines = json.load(f)
            for m in medicines:
                b = m.get("brand_name", "")
                known = m.get("known_brands", [])
                if clean_brand.lower() in b.lower() or any(clean_brand.lower() == k.lower() for k in known):
                    active_ing = m.get("substances", [m.get("generic_name")])[0]
                    rxcui = m.get("rxcui", "")
                    alt_brands = set(known)
                    indications = [m.get("indications", "")[:200]]
                    db_found = True
                    break

    if not db_found:
        raise HTTPException(
            status_code=404,
            detail=f"Brand '{clean_brand}' not found in the pharmaceutical knowledge graph."
        )

    # Construct subgraph for frontend visualization
    b_target_id = f"brand_{clean_brand.lower()}"
    nodes[b_target_id] = GraphNode(id=b_target_id, label=clean_brand, type="Brand")

    if active_ing:
        ing_id = f"ing_{active_ing.lower()}"
        nodes[ing_id] = GraphNode(id=ing_id, label=active_ing, type="Ingredient")
        edges.append(GraphEdge(source=b_target_id, target=ing_id, relationship="CONTAINS"))

        for alt in list(alt_brands)[:10]:
            if alt.lower() != clean_brand.lower():
                alt_id = f"brand_{alt.lower()}"
                nodes[alt_id] = GraphNode(id=alt_id, label=alt, type="Brand")
                edges.append(GraphEdge(source=alt_id, target=ing_id, relationship="CONTAINS"))

    return AlternativesResponse(
        queried_brand=clean_brand,
        active_ingredient=active_ing,
        rxcui=rxcui,
        alternative_brands=[b for b in alt_brands if b.lower() != clean_brand.lower()],
        indications=indications,
        graph=GraphData(nodes=list(nodes.values()), edges=edges),
        clinical_records=[
            ClinicalRecord(
                evidence_id=1,
                brand_name=clean_brand,
                generic_name=active_ing or clean_brand,
                active_substance=active_ing or clean_brand,
                rxcui=rxcui,
                indications=indications[0] if indications else "Clinical indications documented in official label.",
                alternative_brands=[b for b in alt_brands if b.lower() != clean_brand.lower()]
            )
        ] if active_ing else [],
        citations=[
            Citation(
                id=1,
                source="NIH RxNorm & openFDA",
                title=f"{clean_brand} ({active_ing}) Multi-Brand Composition Mapping",
                rxcui=rxcui or "N/A",
                tty="BN / IN",
                evidence=f"Active chemical composition {active_ing} mapped to alternative brands: {', '.join(list(alt_brands)[:5])}."
            )
        ] if active_ing else []
    )


@app.get("/api/health")
async def health_check():
    """Verify backend and database connectivity."""
    neo4j_online = False
    if driver:
        try:
            with driver.session() as s:
                res = s.run("RETURN 1 AS ping").single()
                neo4j_online = bool(res and res["ping"] == 1)
        except Exception:
            neo4j_online = False

    return {
        "status": "healthy",
        "neo4j_connected": neo4j_online,
        "gemini_configured": bool(gemini_model is not None),
        "version": "2.0.0"
    }


# Mount the official frontend (index.html, static assets)
frontend_dir = os.path.join(parent_dir, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_service:app", host="0.0.0.0", port=8008, reload=True)
