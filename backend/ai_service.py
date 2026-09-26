import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables (assume .env is in parent directory)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

app = FastAPI(title="MedGraph Nexus API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-3.5-flash')
else:
    model = None

class QueryRequest(BaseModel):
    question: str
    language: str = "en"
    history: list = []

@app.on_event("shutdown")
def shutdown_db_client():
    driver.close()

STOPWORDS = {
    "what", "is", "are", "the", "for", "with", "does", "which", "give", "me", "a", "an",
    "tell", "about", "cure", "can", "treat", "used", "uses", "use", "usage", "to", "in", "of", "and", "drug",
    "drugs", "medicine", "medicines", "medication", "medications", "pill", "pills", "tablet",
    "tablets", "how", "class", "available", "there", "show", "list", "their", "any", "some",
    "and", "or", "associated", "treats", "side", "effect", "effects", "adverse", "reaction",
    "reactions", "warning", "warnings", "precaution", "precautions", "contraindication",
    "contraindications", "interaction", "interactions", "information", "info", "help",
    "details", "detail", "take", "taking", "taken", "prescribed", "dose", "dosage",
    "it", "its", "this", "that", "these", "them", "same", "also", "other", "another",
    "hello", "hi", "hey", "doctor", "dr", "please", "thanks", "thank"
}

PRONOUNS_OR_FOLLOWUPS = {
    "it", "its", "this", "that", "these", "them", "same", "also", 
    "other", "another", "alternative", "alternatives", "instead", "such", "similar"
}

def extract_clinical_keywords(user_query: str, history: list = None) -> list:
    """Extract clinical target entities from query, resolving pronouns and follow-ups from history."""
    tokens = [w.strip("?,.!:;\"'").lower() for w in user_query.split()]
    raw_keywords = [w for w in tokens if len(w) > 2 and w not in STOPWORDS]

    # Check if the query is an anaphoric follow-up (e.g. "what are its side effects?", "are there alternatives?")
    has_followup = any(t in PRONOUNS_OR_FOLLOWUPS for t in tokens) or len(raw_keywords) == 0

    if has_followup and history:
        # Search backwards through user messages first to isolate the clean subject medicine
        for turn in reversed(history):
            if turn.get("role") == "user":
                prev_tokens = [w.strip("?,.!:;\"'()[]").lower() for w in turn.get("content", "").split()]
                prev_candidates = [w for w in prev_tokens if len(w) > 2 and w not in STOPWORDS]
                if prev_candidates:
                    combined = prev_candidates[:2] + raw_keywords
                    return list(dict.fromkeys(combined))

    return raw_keywords if raw_keywords else [user_query.strip().lower()]

def query_neo4j_for_context(user_query: str, history: list = None) -> str:
    keywords = extract_clinical_keywords(user_query, history)

    cypher = """
    MATCH (m:Medicine)
    WHERE any(kw IN $keywords WHERE toLower(m.brand_name) CONTAINS kw)
       OR any(kw IN $keywords WHERE toLower(m.generic_name) CONTAINS kw)
       OR any(kw IN $keywords WHERE any(b IN coalesce(m.known_brands, []) WHERE toLower(b) CONTAINS kw))
       OR any(kw IN $keywords WHERE toLower(coalesce(m.category, '')) CONTAINS kw)
       OR EXISTS {
           MATCH (m)-[:CONTAINS_SUBSTANCE]->(sub:ActiveSubstance)
           WHERE any(kw IN $keywords WHERE toLower(sub.name) CONTAINS kw)
       }
       OR EXISTS {
           MATCH (m)-[:BELONGS_TO_CLASS]->(cls:DrugClass)
           WHERE any(kw IN $keywords WHERE toLower(cls.name) CONTAINS kw)
       }
       OR EXISTS {
           MATCH (m)-[:TREATS_INDICATION]->(i:Indication)
           WHERE any(kw IN $keywords WHERE toLower(i.description) CONTAINS kw)
       }
    OPTIONAL MATCH (m)-[:TREATS_INDICATION]->(i:Indication)
    OPTIONAL MATCH (m)-[:CONTAINS_SUBSTANCE]->(sub:ActiveSubstance)
    OPTIONAL MATCH (m)-[:BELONGS_TO_CLASS]->(cls:DrugClass)
    OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
    WITH m, i, man,
         collect(DISTINCT sub.name) AS substances,
         collect(DISTINCT cls.name) AS drug_classes
    WITH m, i, man, substances, drug_classes,
         reduce(score = 0, kw IN $keywords |
            score + 
            (CASE WHEN toLower(m.brand_name) CONTAINS kw THEN 15 ELSE 0 END) +
            (CASE WHEN toLower(m.generic_name) CONTAINS kw THEN 15 ELSE 0 END) +
            (CASE WHEN any(b IN coalesce(m.known_brands, []) WHERE toLower(b) CONTAINS kw) THEN 14 ELSE 0 END) +
            (CASE WHEN any(s IN substances WHERE toLower(s) CONTAINS kw) THEN 12 ELSE 0 END) +
            (CASE WHEN any(c IN drug_classes WHERE toLower(c) CONTAINS kw) THEN 8 ELSE 0 END) +
            (CASE WHEN toLower(coalesce(m.category, '')) CONTAINS kw THEN 5 ELSE 0 END) +
            (CASE WHEN toLower(coalesce(i.description, '')) CONTAINS kw THEN 2 ELSE 0 END)
         ) AS relevance
    ORDER BY relevance DESC
    RETURN 
        m.brand_name AS brand, 
        m.generic_name AS generic,
        m.known_brands AS known_brands,
        m.rxcui AS rxcui,
        m.category AS category,
        m.warnings AS warnings,
        m.adverse_reactions AS side_effects,
        m.contraindications AS contraindications,
        m.drug_interactions AS drug_interactions,
        i.description AS indication,
        substances,
        drug_classes,
        man.name AS manufacturer
    LIMIT 6
    """

    try:
        with driver.session() as session:
            result = session.run(cypher, keywords=keywords)
            records = list(result)
            if not records:
                return "No relevant clinical records found in the Knowledge Graph for this inquiry."

            context_blocks = []
            for r in records:
                substances_str = ", ".join(r["substances"]) if r["substances"] else r["generic"]
                classes_str = ", ".join(r["drug_classes"]) if r["drug_classes"] else "Not Classified"
                rxcui_str = f" (RxCUI: {r['rxcui']})" if r.get("rxcui") else ""
                brands = [b for b in r.get("known_brands", []) if b and b.lower() != r["brand"].lower()]
                brands_str = f"Also Marketed As: {', '.join(brands[:4])}" if brands else ""

                block = [
                    f"Clinical Medicine: {r['brand']} [Generic: {r['generic']}{rxcui_str}]",
                    f"Therapeutic Category: {r.get('category', 'General Medicine')}",
                    f"Pharmacological Class: {classes_str}",
                    f"Active Chemical Substance: {substances_str}",
                    f"Licensed Manufacturer: {r['manufacturer']}",
                    f"Approved Indication/Usage: {r['indication']}"
                ]
                if brands_str:
                    block.append(brands_str)
                if r.get("side_effects") and r["side_effects"] != "None specified in label summary":
                    block.append(f"Adverse Reactions & Side Effects: {r['side_effects']}")
                if r.get("warnings") and r["warnings"] != "None specified in label summary":
                    block.append(f"Black Box Warnings & Precautions: {r['warnings']}")
                if r.get("contraindications") and r["contraindications"] != "None specified in label summary":
                    block.append(f"Contraindications: {r['contraindications']}")
                if r.get("drug_interactions") and r["drug_interactions"] != "None specified in label summary":
                    block.append(f"Drug-Drug Interactions: {r['drug_interactions']}")

                context_blocks.append("\n".join(block))
            return "\n\n---\n\n".join(context_blocks)
    except Exception as e:
        return f"Error retrieving from Knowledge Graph: {str(e)}"

@app.post("/ask")
async def ask_medgraph(request: QueryRequest):
    if not model:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured in .env")

    # 1. Retrieve Context from Neo4j with Conversational Coreference Resolution
    graph_context = query_neo4j_for_context(request.question, request.history)

    # 2. Format Conversational History for Multi-Turn Continuity
    formatted_history = "No previous context (Initial inquiry)."
    if request.history:
        turns = []
        for msg in request.history[-6:]:
            role = "Patient" if msg.get("role") == "user" else "Doctor AI"
            content = msg.get("content", "").strip()
            if content:
                # Truncate large previous blocks for optimal prompt token efficiency
                snippet = content[:280] + "..." if len(content) > 280 else content
                turns.append(f"{role}: {snippet}")
        if turns:
            formatted_history = "\n".join(turns)

    # 3. Construct Doctor AI Persona Prompt with Fluid Medical Prose
    prompt = f"""
You are Dr. Nexus, a compassionate, articulate, senior clinical physician and pharmacology specialist.
Your role is to explain clinical findings to the patient in fluent, empathetic, and beautifully structured medical language.

PATIENT CONSULTATION HISTORY:
{formatted_history}

CURRENT PATIENT QUESTION:
"{request.question}"

TARGET RESPONSE LANGUAGE:
{request.language} (en = English, te = Telugu, hi = Hindi)

VERIFIED KNOWLEDGE GRAPH GROUND-TRUTH:
{graph_context}

CRITICAL INSTRUCTIONS FOR SENTENCE FORMATION & PRESENTATION:
1. CONVERSATIONAL CONTINUITY & CONTEXT:
   - If the patient is asking a follow-up question (e.g. asking about "its side effects", "can I take it with food", or "alternatives"), maintain natural continuity! Acknowledge the medicine previously discussed (e.g., "Continuing from our discussion on Amoxicillin...").
   - Never sound like an amnesiac or a disjointed database.

2. NEVER DUMP RAW DATABASE FIELDS:
   - DO NOT copy-paste raw database lines (such as "• Medicine: X | Category: Y | Manufacturer: Z"). That is unacceptable.
   - Synthesize the facts into cohesive, elegant clinical prose. Write as an expert doctor speaks to an attentive patient.

3. STRUCTURED & READABLE CLINICAL PRESENTATION:
   - Organize your response into clear, inviting sections:
     • **Direct Answer & Clinical Overview**: Answer the patient's immediate question directly in warm, complete sentences.
     • **Mechanism of Action & Pharmacology**: Seamlessly explain the active chemical substance and pharmacologic drug class in accessible terms.
     • **Clinical Indications or Safety Precautions**: Use readable bullet points ONLY when listing distinct indications, adverse reactions, or warning points.
     • **Practical Physician Advice**: Emphasize safe administration, consulting their prescribing physician, or monitoring for adverse symptoms.

4. ACCURACY & ZERO HALLUCINATION:
   - Base all clinical claims (active substances, drug classes, indications, contraindications) STRICTLY on the Knowledge Graph ground truth above.
   - If a specific inquiry (e.g., exact pediatric dose for a rare condition) is not present in the graph data, explain transparently and advise professional medical evaluation.

5. NATIVE MULTILINGUAL ELOQUENCE:
   - If Target Language is Telugu ('te'): Respond in fluent, respectful Telugu, keeping key pharmaceutical drug names in parentheses for clarity.
   - If Target Language is Hindi ('hi'): Respond in fluent, professional Hindi, keeping key pharmaceutical drug names in parentheses.
   - If Target Language is English ('en'): Respond in polished, reassuring clinical English.
"""

    # 4. Generate Response with Graceful Clinical Fallback
    try:
        response = model.generate_content(prompt)
        return {"answer": response.text, "context_used": graph_context}
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg or "quota" in error_msg.lower():
            # Graceful structured physician fallback
            fallback_answer = (
                f"### ⚕️ Clinical Consultation Summary\n\n"
                f"**Clinical Note**: Here are the verified pharmacological facts retrieved directly from MedGraph Nexus for your inquiry:\n\n"
                f"{graph_context}\n\n"
                f"---\n*Note: High traffic reached temporary Gemini free-tier rate limits. Ground-truth clinical facts served directly.*"
            )
            return {"answer": fallback_answer, "context_used": graph_context}
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {error_msg}")

@app.get("/api/keepalive")
async def keepalive():
    """Endpoint to keep Neo4j Aura alive"""
    try:
        with driver.session() as session:
            session.run("MERGE (h:SystemHeartbeat {id: 'neo4j_keepalive'}) SET h.last_active = datetime()")
        return {"status": "success", "message": "Heartbeat sent to Neo4j"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_service:app", host="0.0.0.0", port=8000, reload=True)
