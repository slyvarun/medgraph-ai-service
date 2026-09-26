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
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    model = None

class QueryRequest(BaseModel):
    question: str
    language: str = "en"
    history: list = []

@app.on_event("shutdown")
def shutdown_db_client():
    driver.close()

def query_neo4j_for_context(user_query: str) -> str:
    # Filter common conversational words to extract clinical search tokens
    stopwords = {
        "what", "is", "are", "the", "for", "with", "does", "which", "give", "me", "a", "an",
        "tell", "about", "cure", "can", "treat", "used", "to", "in", "of", "and", "drug",
        "drugs", "medicine", "medicines", "medication", "medications", "pill", "pills", "tablet",
        "tablets", "how", "class", "available", "there", "show", "list", "their", "any", "some",
        "and", "or", "associated", "treats"
    }
    tokens = [w.strip("?,.!:;\"'").lower() for w in user_query.split()]
    keywords = [w for w in tokens if len(w) > 2 and w not in stopwords]
    
    if not keywords:
        keywords = [user_query.strip().lower()]

    cypher = """
    MATCH (m:Medicine)
    WHERE any(kw IN $keywords WHERE toLower(m.brand_name) CONTAINS kw)
       OR any(kw IN $keywords WHERE toLower(m.generic_name) CONTAINS kw)
       OR any(kw IN $keywords WHERE toLower(coalesce(m.category, '')) CONTAINS kw)
       OR EXISTS {
           MATCH (m)-[:TREATS_INDICATION]->(i:Indication)
           WHERE any(kw IN $keywords WHERE toLower(i.description) CONTAINS kw)
       }
       OR EXISTS {
           MATCH (m)-[:CONTAINS_SUBSTANCE]->(sub:ActiveSubstance)
           WHERE any(kw IN $keywords WHERE toLower(sub.name) CONTAINS kw)
       }
       OR EXISTS {
           MATCH (m)-[:BELONGS_TO_CLASS]->(cls:DrugClass)
           WHERE any(kw IN $keywords WHERE toLower(cls.name) CONTAINS kw)
       }
    OPTIONAL MATCH (m)-[:TREATS_INDICATION]->(i:Indication)
    OPTIONAL MATCH (m)-[:CONTAINS_SUBSTANCE]->(sub:ActiveSubstance)
    OPTIONAL MATCH (m)-[:BELONGS_TO_CLASS]->(cls:DrugClass)
    OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
    WITH m, i, man,
         collect(DISTINCT sub.name) AS substances,
         collect(DISTINCT cls.name) AS drug_classes
    RETURN 
        m.brand_name AS brand, 
        m.generic_name AS generic,
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
            context = []
            for record in result:
                substances_str = ", ".join(record["substances"]) if record["substances"] else record["generic"]
                classes_str = ", ".join(record["drug_classes"]) if record["drug_classes"] else "Not Classified"
                med_info = [
                    f"• Medicine: {record['brand']} (Generic: {record['generic']} | Category: {record.get('category', 'General')})",
                    f"  Pharmacologic Class: {classes_str}",
                    f"  Active Substance(s): {substances_str}",
                    f"  Manufacturer: {record['manufacturer']}",
                    f"  Indication/Usage: {record['indication']}"
                ]
                if record.get("side_effects") and record["side_effects"] != "None specified in label summary":
                    med_info.append(f"  Side Effects / Adverse Reactions: {record['side_effects']}")
                if record.get("warnings") and record["warnings"] != "None specified in label summary":
                    med_info.append(f"  Warnings & Precautions: {record['warnings']}")
                if record.get("contraindications") and record["contraindications"] != "None specified in label summary":
                    med_info.append(f"  Contraindications: {record['contraindications']}")
                if record.get("drug_interactions") and record["drug_interactions"] != "None specified in label summary":
                    med_info.append(f"  Drug-Drug Interactions: {record['drug_interactions']}")
                context.append("\n".join(med_info))
            return "\n\n".join(context) if context else "No relevant medical context found in the graph."
    except Exception as e:
        return f"Error retrieving from Knowledge Graph: {str(e)}"

@app.post("/ask")
async def ask_medgraph(request: QueryRequest):
    if not model:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured in .env")

    # 1. Retrieve Context from Neo4j
    graph_context = query_neo4j_for_context(request.question)

    # 2. Build the Prompt for GraphRAG
    prompt = f"""
    You are MedGraph Nexus, a highly knowledgeable and friendly clinical AI assistant.
    You answer medical queries based ONLY on the provided Knowledge Graph context.
    
    User Question: {request.question}
    Target Language Code: {request.language}
    
    Knowledge Graph Context:
    {graph_context}
    
    Instructions:
    - Answer the question accurately using only the provided context.
    - If the context doesn't contain the answer, politely state that you do not have that information in the database.
    - Respond strictly in the language specified by the Target Language Code (e.g. 'en' for English, 'hi' for Hindi, 'te' for Telugu).
    - Format your response nicely with markdown or bullet points if appropriate.
    """

    # 3. Call the LLM
    try:
        response = model.generate_content(prompt)
        return {"answer": response.text, "context_used": graph_context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")

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
