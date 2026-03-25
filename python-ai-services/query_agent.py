import os
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()

# 2026 Production Config
# Using the stable alias or the specific preview ID
MODEL_ID = "gemini-3-flash-preview" 

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def search_graph(entity_name):
    """Deep search across 50k nodes with case-insensitivity"""
    with driver.session() as session:
        # We search across Manufacturer, Category, AND Name
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.manufacturer) CONTAINS toLower($name)
        OR toLower(d.category) CONTAINS toLower($name)
        OR toLower(d.name) CONTAINS toLower($name)
        RETURN d.name as name, d.manufacturer as maker, d.indication as treats, d.category as cat
        LIMIT 10
        """
        results = session.run(query, name=entity_name)
        
        # Format for Gemini to understand clearly
        data = [f"Medicine: {r['name']} | Maker: {r['maker']} | Category: {r['cat']} | Used for: {r['treats']}" for r in results]
        
        if not data:
            print(f"⚠️ No graph data found for: {entity_name}")
        return data

def ask_agent(question, long_doc=""):
    try:
        # Phase 1: Entity Extraction
        intent_res = client.models.generate_content(
            model=MODEL_ID,
            contents=f"Extract the medicine name from: '{question}'. Reply ONLY with the name."
        )
        target = intent_res.text.strip()
        
        # Phase 2: Graph Retrieval
        facts = search_graph(target)
        
        # Phase 3: Synthesis
        final_prompt = f"""
        You are MedGraph Nexus (Gemini 3). Use these clinical facts to answer.
        GRAPH_DATA: {facts}
        DOC_DATA: {long_doc}
        USER_QUERY: {question}
        """
        
        response = client.models.generate_content(model=MODEL_ID, contents=final_prompt)
        return response.text
    except Exception as e:
        # Fallback to a guaranteed stable model if Gemini 3 is in maintenance
        print(f"Handled Redirect: {e}")
        stable_res = client.models.generate_content(model="gemini-2.5-flash", contents=question)
        return f"[Fallback Mode] {stable_res.text}"
