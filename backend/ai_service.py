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
import re
import json
import requests
from typing import List, Dict, Any, Optional, Literal
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
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

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
    language: str = Field("en", description="Target response language (en, es, fr, de, hi, te)", example="en")
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


class QueryIntent(BaseModel):
    intent: Literal["SYMPTOM_TO_MEDICINE", "BRAND_TO_ALTERNATIVE", "AMBIGUOUS"]
    extracted_entities: List[str] = Field(default_factory=list, description="Extracted clinical symptoms or medication entities")
    confidence: float = Field(default=1.0, description="Confidence score from 0.0 to 1.0")
    clarification_question: Optional[str] = Field(default=None, description="Conversational question for ambiguous queries")


class SearchResponse(BaseModel):
    query: str
    answer: str
    is_grounded: bool
    intent: Optional[str] = "SYMPTOM_TO_MEDICINE"
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
# Two-Step Prompting Strategy: Intent Classifier & Intent-Tailored Generators
# ---------------------------------------------------------------------------
INTENT_ROUTER_SYSTEM_PROMPT = """You are the clinical triage router for MedGraph Nexus.
Analyze the user's clinical or pharmaceutical inquiry and classify their intent into one of three strict categories:

1. SYMPTOM_TO_MEDICINE: The user describes physical feelings, medical conditions, or symptoms (e.g., 'my head is pounding and throbbing', 'high blood pressure', 'fever and chills', 'chest pain', 'arthritic joint pain', or in any language like 'j\'ai mal à la tête', 'ich habe starke Kopfschmerzen', 'dolor de cabeza', 'सिरदर्द', 'calentura'). They want to discover treatments, mechanisms, or therapeutic medications.
2. BRAND_TO_ALTERNATIVE: The user queries a pharmaceutical brand, commercial trade name, or active chemical substance (e.g., 'What are alternatives to Lisinopril?', 'alternatives à Lisinopril', 'Alternativen zu Lisinopril', 'Lisinopril', 'Advil', 'Exforge', 'Lipitor', 'Amlodipine'). They want composition, bioequivalent alternatives, or what the drug treats.
3. AMBIGUOUS: The input is a short, isolated phrase or single word without enough context to determine whether they have symptoms or are asking about a specific drug (e.g., 'blood pressure', 'pressure', 'medicine', 'meds', 'pills', 'help', 'drugs', 'sick', 'presión', 'médicament', 'tension', 'Medikamente', 'Blutdruck').

CRITICAL MULTILINGUAL INSTRUCTION:
Because the FDA Knowledge Graph is indexed in standard English clinical terms, you MUST always translate and normalize the 'extracted_entities' into canonical English clinical terms (e.g., 'mal de tête' / 'Kopfschmerzen' / 'dolor de cabeza' / 'सिरदर्द' -> 'Headache', 'douleur' / 'Schmerz' / 'dolor' / 'दर्द' -> 'Pain', 'hypertension artérielle' / 'Bluthochdruck' / 'presión alta' / 'उच्च रक्तचाप' -> 'Hypertension', 'fièvre' / 'Fieber' / 'fiebre' / 'बुखार' -> 'Fever', 'toux' / 'Husten' / 'tos' / 'खांसी' -> 'Cough', 'douleur articulaire' / 'Gelenkschmerzen' -> 'Arthralgia').

Return ONLY a valid JSON object matching this schema:
{
  "intent": "SYMPTOM_TO_MEDICINE" | "BRAND_TO_ALTERNATIVE" | "AMBIGUOUS",
  "extracted_entities": ["canonical_english_clinical_term"],
  "confidence": 0.95,
  "clarification_question": "polite clarifying question if AMBIGUOUS in the user's inquiry language, else null"
}
"""

PROMPT_INTENT_A = """You are Doctor AI for MedGraph Nexus, an expert conversational clinical pharmacotherapy assistant.
The user described symptoms or physical ailments, and you have retrieved verified Knowledge Graph records showing active chemical substances and marketed prescription brand formulations that treat them.

<strict_guardrails>
1. INDEPENDENT CLINICAL REVIEWABILITY (FDA NON-DEVICE CDS): You are an informational Clinical Decision Support (CDS) tool, NOT a diagnostic or prescriptive medical device. All findings provide educational reference data to assist licensed healthcare providers and informed patients.
2. ZERO SPECULATION & GROUNDED TRUTH: Rely EXCLUSIVELY on the provided VERIFIED KNOWLEDGE GRAPH CONTEXT. If the context does not contain sufficient pharmaceutical evidence to address the query, state: "I cannot find verified pharmaceutical records in MedGraph Nexus for the requested query. Please consult a qualified healthcare provider or medical professional." Do NOT fabricate mechanisms, off-label indications, or brand names.
3. ENTITY PROVENANCE HOOKS: You MUST format every clinical entity (condition, symptom, active ingredient, brand name, and drug class) inside bold brackets: **[Entity Name]** (e.g., **[Headache]**, **[Hypertension]**, **[Sumatriptan]**, **[Amlodipine Besylate]**, **[Exforge]**, **[Beta Blocker]**). This enables the frontend Provenance Pane to hook into terms for interactive openFDA inspection.
4. CITATION DISCIPLINE: Every active substance, therapeutic indication, brand name, and safety warning MUST include bracketed footnote citations like [1], [2] corresponding directly to the Evidence Record ID numbers provided in the context.
5. CONVERSATIONAL PROSE & EXPLANATION: Write in warm, compassionate, and professional clinical prose in flowing paragraphs. Avoid rigid bullet dumps or raw JSON keys. Explain what each active ingredient does biochemically and group brand formulations logically.
6. CLOSING CHECK-IN: End with an engaging conversational check-in and an explicit reminder that patients must consult their physician or pharmacist before starting or modifying any treatment.
</strict_guardrails>"""

PROMPT_INTENT_B = """You are Doctor AI for MedGraph Nexus, an expert conversational clinical pharmacotherapy assistant.
The user asked about a specific medication brand or active ingredient, and you have retrieved verified Knowledge Graph records showing its chemical composition, bioequivalent alternatives, and FDA label indications.

<strict_guardrails>
1. INDEPENDENT CLINICAL REVIEWABILITY (FDA NON-DEVICE CDS): You are an informational Clinical Decision Support (CDS) tool, NOT a diagnostic or prescriptive medical device. All findings provide educational reference data to assist licensed healthcare providers and informed patients.
2. ZERO SPECULATION & GROUNDED TRUTH: Rely EXCLUSIVELY on the provided VERIFIED KNOWLEDGE GRAPH CONTEXT. If the context does not contain sufficient pharmaceutical evidence to address the query, state: "I cannot find verified pharmaceutical records in MedGraph Nexus for the requested query. Please consult a qualified healthcare provider or medical professional." Do NOT invent parallel brands or indications.
3. ENTITY PROVENANCE HOOKS: You MUST format every clinical entity (queried drug, active chemical ingredient, alternative brand formulation, treated condition, and drug class) inside bold brackets: **[Entity Name]** (e.g., **[Lisinopril]**, **[Prinivil]**, **[Zestril]**, **[Hypertension]**, **[ACE Inhibitor]**). This enables the frontend Provenance Pane to hook into terms for interactive openFDA inspection.
4. CITATION DISCIPLINE: Every active substance, alternative brand formulation, indication, and safety warning MUST include bracketed footnote citations like [1], [2] corresponding directly to the Evidence Record ID numbers in the context.
5. CONVERSATIONAL PROSE & EXPLANATION: Write in warm, articulate, and professional clinical prose in flowing paragraphs. Clearly identify the primary active ingredient, detail its bioequivalent brand alternatives, explain primary therapeutic uses, and summarize notable label warnings.
6. CLOSING CHECK-IN: Conclude with an engaging conversational check-in (e.g., asking if they are exploring alternatives due to side effects, cost, or supply shortages) and an explicit reminder to consult their prescribing clinician or pharmacist before switching medications.
</strict_guardrails>"""


def classify_user_intent(query: str, history: Optional[List[Dict[str, str]]] = None) -> QueryIntent:
    """
    Step 1: Triage router classifying intent into:
    - SYMPTOM_TO_MEDICINE: Symptom or physical condition query
    - BRAND_TO_ALTERNATIVE: Brand, generic drug, or alternative query
    - AMBIGUOUS: Vague, single-word, or ambiguous concept (zero DB trip)
    """
    clean_q = query.strip()
    lower_q = clean_q.lower()
    stripped_tokens = [w.strip("?,.!:;\"'&/()").lower() for w in clean_q.split() if w.strip("?,.!:;\"'&/()")]

    # 1. Fast heuristic for Ambiguous single/double-word inputs
    AMBIGUOUS_KEYWORDS = {
        "blood pressure", "pressure", "medicine", "meds", "pills", "drugs", "options",
        "sick", "help", "treatment", "cure",
        # French
        "tension", "medicament", "médicament", "médicaments", "pilule", "pilules", "douleur", "maladie",
        # German
        "blutdruck", "medikament", "medikamente", "pillen", "tabletten", "schmerz", "krankheit",
        # Spanish & Hindi
        "presion", "presión", "medicamento", "medicamentos", "pastillas", "dolor", "दवा", "दवाई", "बीमार", "दर्द"
    }
    normalized_phrase = " ".join(stripped_tokens)
    if normalized_phrase in AMBIGUOUS_KEYWORDS:
        topic = "blood pressure" if ("pressure" in normalized_phrase or "tension" in normalized_phrase or "blutdruck" in normalized_phrase or "presión" in normalized_phrase or "presion" in normalized_phrase) else clean_q
        return QueryIntent(
            intent="AMBIGUOUS",
            extracted_entities=[clean_q],
            confidence=1.0,
            clarification_question=(
                f"I see you're asking about {topic}. Are you experiencing elevated blood pressure and looking for "
                f"treatment options, or do you have a specific medication you'd like to check alternatives for?"
                if topic == "blood pressure" else
                f"I see you mentioned '{clean_q}'. Are you looking for medications to treat a specific symptom, "
                f"or do you have a particular medication you want to find alternatives for?"
            )
        )

    # 2. Fast heuristic for Alternative / Substitute Queries
    ALT_PHRASES = [
        "alternative", "alternatives", "substitute", "substitutes", "instead of", "switch from", "other brand", "other brands",
        # French
        "substitut", "substituts", "remplacer", "au lieu de", "autre marque", "autres marques",
        # German
        "alternativen", "ersatz", "statt", "anstelle von", "andere marke", "weitere marken",
        # Spanish
        "sustituto", "sustitutos", "en lugar de", "otra marca"
    ]
    if any(p in lower_q for p in ALT_PHRASES):
        # Extract potential brand name
        candidate = clean_q
        for p in ALT_PHRASES:
            candidate = re.sub(rf"\b{p}\b", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\b(what|are|to|for|is|can|i|take|drugs?|medicines?|brands?|pour|pourquoi|zu|für)\b", "", candidate, flags=re.IGNORECASE).strip(" ?,.!:;")
        extracted = [candidate.strip()] if candidate.strip() else [clean_q]
        return QueryIntent(
            intent="BRAND_TO_ALTERNATIVE",
            extracted_entities=extracted,
            confidence=0.98
        )

    # 3. Fast heuristic for known brand aliases
    for b in COMMON_BRAND_ALIASES:
        if b in lower_q:
            return QueryIntent(
                intent="BRAND_TO_ALTERNATIVE",
                extracted_entities=[b.title()],
                confidence=0.95
            )

    # 4. OpenRouter Intent Classifier
    if OPENROUTER_API_KEY:
        try:
            messages = [{"role": "system", "content": INTENT_ROUTER_SYSTEM_PROMPT.strip()}]
            if history:
                for h in history[-2:]:
                    r = "user" if h.get("role") in ["user", "human"] else "assistant"
                    c = (h.get("content") or "").strip()
                    if c:
                        messages.append({"role": r, "content": c[:140]})
            messages.append({"role": "user", "content": clean_q})

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://medgraph.nexus",
                "X-Title": "MedGraph Nexus Intent Router"
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                },
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                raw_txt = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                parsed = json.loads(raw_txt)
                raw_intent = parsed.get("intent", "").upper()
                
                # Normalize intent labels
                if raw_intent in ["A", "SYMPTOM_TO_MEDICINE"]:
                    norm_intent = "SYMPTOM_TO_MEDICINE"
                elif raw_intent in ["B", "BRAND_TO_ALTERNATIVE"]:
                    norm_intent = "BRAND_TO_ALTERNATIVE"
                else:
                    norm_intent = "AMBIGUOUS"

                return QueryIntent(
                    intent=norm_intent,
                    extracted_entities=parsed.get("extracted_entities", [clean_q]),
                    confidence=float(parsed.get("confidence", 0.95)),
                    clarification_question=parsed.get("clarification_question")
                )
        except Exception:
            pass

    # 5. Heuristic fallback based on symptom phrases
    for phrase in PHRASE_NORMALIZATIONS:
        if phrase in lower_q:
            return QueryIntent(
                intent="SYMPTOM_TO_MEDICINE",
                extracted_entities=[PHRASE_NORMALIZATIONS[phrase]],
                confidence=0.9
            )

    return QueryIntent(
        intent="SYMPTOM_TO_MEDICINE",
        extracted_entities=[clean_q],
        confidence=0.8
    )


LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)"
}

REFUSAL_MESSAGES = {
    "en": "I cannot find verified pharmaceutical records in MedGraph Nexus for the requested query. Please consult a qualified healthcare provider or medical professional.",
    "es": "No puedo encontrar registros farmacéuticos verificados en MedGraph Nexus para la consulta solicitada. Por favor, consulte con un profesional de la salud o médico calificado.",
    "fr": "Je ne trouve pas d'enregistrements pharmaceutiques vérifiés dans MedGraph Nexus pour la requête demandée. Veuillez consulter un professionnel de la santé qualifié.",
    "de": "Ich kann in MedGraph Nexus keine verifizierten pharmazeutischen Datensätze für die angeforderte Abfrage finden. Bitte wenden Sie sich an einen qualifizierten Arzt oder Apotheker.",
    "hi": "मुझे अनुरोधित प्रश्न के लिए MedGraph Nexus में सत्यापित फार्मास्युटिकल रिकॉर्ड नहीं मिले। कृपया किसी योग्य स्वास्थ्य सेवा प्रदाता या चिकित्सक से परामर्श लें।",
    "te": "అభ్యర్థించిన ప్రశ్న కోసం MedGraph Nexusలో ధృవీకరించబడిన ఫార్మాస్యూటికల్ రికార్డులు కనుగొనబడలేదు. దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి."
}

AMBIGUOUS_FALLBACKS = {
    "en": "I see you mentioned '{query}'. Are you looking for medications to treat a specific symptom, or do you have a particular medication you want to find alternatives for?",
    "es": "Veo que mencionó '{query}'. ¿Está buscando medicamentos para tratar un síntoma específico o desea buscar alternativas para un medicamento en particular?",
    "fr": "Je vois que vous avez mentionné '{query}'. Cherchez-vous des médicaments pour traiter un symptôme précis, ou souhaitez-vous trouver des alternatives pour un médicament particulier ?",
    "de": "Sie haben '{query}' erwähnt. Suchen Sie nach Medikamenten zur Behandlung eines bestimmten Symptoms oder möchten Sie Alternativen für ein bestimmtes Medikament finden?",
    "hi": "आपने '{query}' का उल्लेख किया है। क्या आप किसी विशिष्ट लक्षण के उपचार के लिए दवाएं खोज रहे हैं, या किसी विशेष दवा के विकल्प ढूंढना चाहते हैं?",
    "te": "మీరు '{query}' ను పేర్కొన్నారు. మీరు నిర్దిష్ట లక్షణానికి చికిత్స కోసం మందులను చూస్తున్నారా లేదా ఏదైనా నిర్దిష్ట మందు కోసం ప్రత్యామ్నాయాలను కనుగొనాలనుకుంటున్నారా?"
}


def generate_conversational_response(
    query: str,
    raw_context: str,
    history: Optional[List[Dict[str, str]]] = None,
    language: str = "en",
    retrieved_summary: Optional[Dict[str, Any]] = None,
    intent: str = "A"
) -> str:
    """
    Step 2: Generates intent-tailored, humanly conversational, clinically grounded responses
    using OpenRouter with fallback to Google Gemini or deterministic graph summaries.
    """
    history = history or []
    system_prompt = PROMPT_INTENT_B if intent in ["B", "BRAND_TO_ALTERNATIVE"] else PROMPT_INTENT_A
    target_lang_desc = LANGUAGE_NAMES.get(language.lower(), language)

    # 1. Primary Engine: OpenRouter API (OpenAI GPT-4o-mini / Meta Llama 3.3)
    if OPENROUTER_API_KEY:
        try:
            messages = [{"role": "system", "content": system_prompt.strip()}]

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
                f"TARGET RESPONSE LANGUAGE: {target_lang_desc}\n"
                f"(CRITICAL: Write your entire conversational response fluently in {target_lang_desc}, "
                f"while strictly maintaining bracketed entity tokens like **[Entity Name]** and footnote citations like [1], [2]).\n\n"
                f"Please formulate a warm, conversational, yet clinically grounded response with proper footnote citations [1], [2] linked to the records above."
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
                f"{system_prompt}\n\n"
                f"VERIFIED KNOWLEDGE GRAPH CONTEXT:\n{raw_context}\n\n"
                f"USER INQUIRY: {query}\n"
                f"TARGET LANGUAGE: {language}"
            )
            gem_res = gemini_model.generate_content(fallback_prompt)
            if gem_res and gem_res.text:
                return gem_res.text.strip()
        except Exception:
            pass

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
def retrieve_graph_subgraph(
    query_text: str,
    intent: str = "A",
    extracted_entities: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes graph traversal across Neo4j or uses curated clinical cache
    to extract Symptoms, Ingredients, and Alternative Brands.
    Dynamically routes traversal based on intent:
    - Intent A: Queries Symptom index first -> traverses to Ingredients -> Brands
    - Intent B: Queries Brand/Ingredient index first -> traverses to Alternatives
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

    # Build search terms using extracted entities, normalizations, and aliases
    resolved_search_terms = []
    if extracted_entities:
        for ent in extracted_entities:
            clean_ent = ent.lower().strip()
            if clean_ent in PHRASE_NORMALIZATIONS:
                resolved_search_terms.append(PHRASE_NORMALIZATIONS[clean_ent])
            elif clean_ent in COMMON_BRAND_ALIASES:
                resolved_search_terms.append(COMMON_BRAND_ALIASES[clean_ent])
            elif clean_ent not in STOPWORDS and len(clean_ent) > 2:
                resolved_search_terms.append(clean_ent)

    # Priority multi-word phrase matching (e.g. "high blood pressure" -> "hypertension")
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

    # 1. Attempt Live Neo4j Query (Intent-Routed Traversal)
    db_success = False
    if driver:
        try:
            with driver.session() as session:
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

                if intent == "B":
                    # Intent B: Search Brand/Ingredient index first
                    res = list(session.run(cypher_brand_ing, terms=resolved_search_terms))
                    if not res or len(res) == 0:
                        res = list(session.run(cypher_symptom, terms=resolved_search_terms))
                else:
                    # Intent A (or default): Search Symptom index first
                    res = list(session.run(cypher_symptom, terms=resolved_search_terms))
                    if not res or len(res) == 0:
                        res = list(session.run(cypher_brand_ing, terms=resolved_search_terms))



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
    Two-step intent-routed clinical search pipeline:
    1. Step 1 (Intent Classifier): Triages query into:
       - Intent A: Symptom-to-Medicine
       - Intent B: Medicine-to-Alternative
       - Intent C: Ambiguous (asks clarifying question directly)
    2. Step 2 (Dynamic Execution):
       - If Intent C: Skips graph query and returns clarification.
       - If Intent A/B: Executes targeted Neo4j traversal.
    3. Step 3 (Conversational Synthesis): Generates intent-tailored clinical answer with citations.
    """
    # Step 1: Intent Triage Router
    intent_data = classify_user_intent(request.query, request.history)
    if isinstance(intent_data, QueryIntent):
        intent = intent_data.intent
        extracted_entities = intent_data.extracted_entities
        clarification_question = intent_data.clarification_question
    else:
        intent = intent_data.get("intent", "SYMPTOM_TO_MEDICINE")
        extracted_entities = intent_data.get("extracted_entities", [])
        clarification_question = intent_data.get("clarification_question")

    # If Intent C / AMBIGUOUS: Skip database search and return clarification directly with zero DB queries!
    lang_key = (request.language or "en").lower()
    if intent in ["AMBIGUOUS", "C"]:
        fallback_tmpl = AMBIGUOUS_FALLBACKS.get(lang_key, AMBIGUOUS_FALLBACKS["en"])
        clarify_msg = clarification_question or fallback_tmpl.format(query=request.query)
        return SearchResponse(
            query=request.query,
            answer=clarify_msg,
            is_grounded=False,
            intent="AMBIGUOUS",
            symptoms_identified=[],
            compositions=[],
            alternative_medicines=[],
            graph=GraphData(nodes=[], edges=[]),
            clinical_records=[],
            citations=[]
        )

    # Step 2: Dynamic Execution based on Intent
    retrieved = retrieve_graph_subgraph(
        query_text=request.query,
        intent=intent,
        extracted_entities=extracted_entities
    )
    has_context = bool(retrieved["raw_context"])

    if not has_context:
        refusal_msg = REFUSAL_MESSAGES.get(lang_key, REFUSAL_MESSAGES["en"])
        return SearchResponse(
            query=request.query,
            answer=refusal_msg,
            is_grounded=False,
            intent=intent,
            symptoms_identified=[],
            compositions=[],
            alternative_medicines=[],
            graph=GraphData(nodes=[], edges=[]),
            clinical_records=[],
            citations=[]
        )

    # Step 3: Intent-Tailored Conversational Output Generator
    answer_text = generate_conversational_response(
        query=request.query,
        raw_context=retrieved["raw_context"],
        history=request.history,
        language=request.language,
        retrieved_summary=retrieved,
        intent=intent
    )

    return SearchResponse(
        query=request.query,
        answer=answer_text,
        is_grounded=True,
        intent=intent,
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
        "intent": res.intent,
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


class EntityEvidenceResponse(BaseModel):
    entity_name: str
    entity_type: str = Field(..., description="Brand, Ingredient, Symptom, or Medicine")
    rxcui: Optional[str] = None
    therapeutic_category: Optional[str] = "General Medicine"
    indications: str = "None specified in label summary"
    warnings: str = "None specified in label summary"
    contraindications: str = "None specified in label summary"
    adverse_reactions: str = "None specified in label summary"
    drug_interactions: str = "None specified in label summary"
    alternative_brands: List[str] = []
    connected_ingredients: List[str] = []
    treated_symptoms: List[str] = []
    graph: GraphData
    source: str = "openFDA Drug Labeling API & NIH RxNorm Knowledge Graph"


@app.get("/api/evidence/{entity_name}", response_model=EntityEvidenceResponse)
async def get_entity_evidence(entity_name: str = Path(..., min_length=1, description="Target entity to retrieve evidence for")):
    """
    Direct indexed evidence lookup for FDA non-device CDS independent reviewability.
    Pulls raw openFDA indications, warnings, RxCUI, and local subgraph for any
    clicked Brand, Ingredient, or Symptom.
    """
    clean_name = entity_name.strip()
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    
    ent_type = "Entity"
    rxcui = None
    category = "General Medicine"
    indications = ""
    warnings = ""
    contraindications = ""
    adverse_reactions = ""
    drug_interactions = ""
    alt_brands = []
    ingredients = []
    symptoms = []
    found = False

    if driver:
        try:
            with driver.session() as session:
                # 1. Match Brand or Medicine
                cypher_brand = """
                MATCH (b) WHERE (b:Brand OR b:Medicine) AND (toLower(b.name) = toLower($name) OR (b.brand_name IS NOT NULL AND toLower(b.brand_name) = toLower($name)))
                OPTIONAL MATCH (b)-[:CONTAINS]->(i:Ingredient)
                OPTIONAL MATCH (alt:Brand)-[:CONTAINS]->(i)
                OPTIONAL MATCH (i)-[:TREATS]->(s:Symptom)
                RETURN 'Brand' AS entity_type,
                       coalesce(b.name, b.brand_name) AS name,
                       b.rxcui AS rxcui,
                       b.category AS category,
                       b.indications AS indications,
                       b.warnings AS warnings,
                       b.contraindications AS contraindications,
                       b.adverse_reactions AS adverse_reactions,
                       b.drug_interactions AS drug_interactions,
                       collect(DISTINCT i.name) AS ingredients,
                       collect(DISTINCT alt.name) AS alternatives,
                       collect(DISTINCT s.name) AS symptoms
                LIMIT 1
                """
                row = session.run(cypher_brand, name=clean_name).single()
                if not row or not row["name"]:
                    cypher_brand_sub = """
                    MATCH (b) WHERE (b:Brand OR b:Medicine) AND toLower(b.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (b)-[:CONTAINS]->(i:Ingredient)
                    OPTIONAL MATCH (alt:Brand)-[:CONTAINS]->(i)
                    OPTIONAL MATCH (i)-[:TREATS]->(s:Symptom)
                    RETURN 'Brand' AS entity_type,
                           coalesce(b.name, b.brand_name) AS name,
                           b.rxcui AS rxcui,
                           b.category AS category,
                           b.indications AS indications,
                           b.warnings AS warnings,
                           b.contraindications AS contraindications,
                           b.adverse_reactions AS adverse_reactions,
                           b.drug_interactions AS drug_interactions,
                           collect(DISTINCT i.name) AS ingredients,
                           collect(DISTINCT alt.name) AS alternatives,
                           collect(DISTINCT s.name) AS symptoms
                    LIMIT 1
                    """
                    row = session.run(cypher_brand_sub, name=clean_name).single()

                # 2. If not Brand, match Ingredient
                if not row or not row["name"]:
                    cypher_ing = """
                    MATCH (i:Ingredient) WHERE toLower(i.name) = toLower($name) OR toLower(i.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (b:Brand)-[:CONTAINS]->(i)
                    OPTIONAL MATCH (i)-[:TREATS]->(s:Symptom)
                    OPTIONAL MATCH (m:Medicine) WHERE toLower(m.brand_name) = toLower(b.name) OR (m.rxcui IS NOT NULL AND m.rxcui = b.rxcui)
                    RETURN 'Ingredient' AS entity_type,
                           i.name AS name,
                           coalesce(i.rxcui, m.rxcui) AS rxcui,
                           m.category AS category,
                           coalesce(m.indications, b.indications) AS indications,
                           coalesce(m.warnings, b.warnings) AS warnings,
                           m.contraindications AS contraindications,
                           m.adverse_reactions AS adverse_reactions,
                           m.drug_interactions AS drug_interactions,
                           [i.name] AS ingredients,
                           collect(DISTINCT b.name) AS alternatives,
                           collect(DISTINCT s.name) AS symptoms
                    LIMIT 1
                    """
                    row = session.run(cypher_ing, name=clean_name).single()

                # 3. If not Ingredient, match Symptom
                if not row or not row["name"]:
                    cypher_sym = """
                    MATCH (s:Symptom) WHERE toLower(s.name) = toLower($name) OR toLower(s.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (x)-[:TREATS]->(s)
                    MATCH (i:Ingredient) WHERE (x = i) OR ((x:Brand OR x:Medicine) AND (x)-[:CONTAINS]->(i))
                    OPTIONAL MATCH (b:Brand)-[:CONTAINS]->(i)
                    OPTIONAL MATCH (m:Medicine) WHERE toLower(m.brand_name) = toLower(b.name)
                    RETURN 'Symptom' AS entity_type,
                           s.name AS name,
                           null AS rxcui,
                           coalesce(m.category, 'General Medicine') AS category,
                           coalesce(s.description, m.indications, 'Clinical indication / symptom documented in graph.') AS indications,
                           m.warnings AS warnings,
                           m.contraindications AS contraindications,
                           m.adverse_reactions AS adverse_reactions,
                           m.drug_interactions AS drug_interactions,
                           collect(DISTINCT i.name) AS ingredients,
                           collect(DISTINCT b.name) AS alternatives,
                           [s.name] AS symptoms
                    LIMIT 1
                    """
                    row = session.run(cypher_sym, name=clean_name).single()

                if row and row["name"]:
                    found = True
                    ent_type = row["entity_type"]
                    clean_name = row["name"]
                    rxcui = row.get("rxcui")
                    category = row.get("category") or "General Medicine"
                    indications = row.get("indications") or ""
                    warnings = row.get("warnings") or ""
                    contraindications = row.get("contraindications") or ""
                    adverse_reactions = row.get("adverse_reactions") or ""
                    drug_interactions = row.get("drug_interactions") or ""
                    ingredients = [ing for ing in row.get("ingredients", []) if ing]
                    alt_brands = [b for b in row.get("alternatives", []) if b and b.lower() != clean_name.lower()]
                    symptoms = [s for s in row.get("symptoms", []) if s]
        except Exception as e:
            print(f"[!] Evidence lookup error: {e}")

    # Fallback to local dataset if Neo4j is offline or entity not found
    if not found:
        dataset_path = os.path.join(parent_dir, "data", "medicine_dataset.json")
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    medicines = json.load(f)
                for m in medicines:
                    b = m.get("brand_name", "")
                    known = m.get("known_brands", [])
                    subs = m.get("substances", [m.get("generic_name", "")])
                    if clean_name.lower() in b.lower() or any(clean_name.lower() in k.lower() for k in known) or any(clean_name.lower() in s.lower() for s in subs):
                        found = True
                        ent_type = "Brand"
                        clean_name = b
                        rxcui = m.get("rxcui")
                        category = m.get("category", "General Medicine")
                        indications = m.get("indications", "")
                        warnings = m.get("warnings", "")
                        contraindications = m.get("contraindications", "")
                        adverse_reactions = m.get("adverse_reactions", "")
                        drug_interactions = m.get("drug_interactions", "")
                        ingredients = subs
                        alt_brands = [k for k in known if k.lower() != clean_name.lower()]
                        symptoms = [m.get("indications", "")[:100]]
                        break
            except Exception:
                pass

    if not found:
        indications = f"Clinical records and indications for {clean_name} mapped via openFDA / RxNorm knowledge graph."

    # Build local subgraph for Vis.js rendering
    target_id = f"target_{clean_name.lower()[:25]}"
    nodes[target_id] = GraphNode(id=target_id, label=clean_name, type=ent_type)

    for ing in ingredients[:5]:
        ing_id = f"ing_{ing.lower()[:25]}"
        nodes[ing_id] = GraphNode(id=ing_id, label=ing, type="Ingredient")
        if ent_type == "Brand":
            edges.append(GraphEdge(source=target_id, target=ing_id, relationship="CONTAINS"))
        elif ent_type == "Symptom":
            edges.append(GraphEdge(source=ing_id, target=target_id, relationship="TREATS"))

    for alt in alt_brands[:8]:
        alt_id = f"brand_{alt.lower()[:25]}"
        nodes[alt_id] = GraphNode(id=alt_id, label=alt, type="Brand")
        if ingredients:
            first_ing_id = f"ing_{ingredients[0].lower()[:25]}"
            edges.append(GraphEdge(source=alt_id, target=first_ing_id, relationship="CONTAINS"))

    for sym in symptoms[:6]:
        sym_clean = sym[:45]
        sym_id = f"sym_{sym_clean.lower()[:25]}"
        nodes[sym_id] = GraphNode(id=sym_id, label=sym_clean, type="Symptom")
        if ingredients:
            first_ing_id = f"ing_{ingredients[0].lower()[:25]}"
            edges.append(GraphEdge(source=first_ing_id, target=sym_id, relationship="TREATS"))
        elif ent_type == "Brand":
            edges.append(GraphEdge(source=target_id, target=sym_id, relationship="TREATS"))

    return EntityEvidenceResponse(
        entity_name=clean_name,
        entity_type=ent_type,
        rxcui=rxcui or "N/A",
        therapeutic_category=category or "General Medicine",
        indications=indications or "Documented in FDA approved label.",
        warnings=warnings or "None specified in label summary",
        contraindications=contraindications or "None specified in label summary",
        adverse_reactions=adverse_reactions or "None specified in label summary",
        drug_interactions=drug_interactions or "None specified in label summary",
        alternative_brands=alt_brands[:10],
        connected_ingredients=ingredients[:5],
        treated_symptoms=[s[:60] for s in symptoms[:8]],
        graph=GraphData(nodes=list(nodes.values()), edges=edges),
        source="openFDA Drug Labeling API & NIH RxNorm Knowledge Graph"
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
