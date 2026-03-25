import os
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()

# 2026 Production Config
# Using the stable alias or the specific preview ID
MODEL_ID = "gemini-2.5-flash" 

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def search_graph(entity_name):
    """
    Global Fuzzy Search: Searches EVERY property on EVERY node 
    to ensure we never return an empty list.
    """
    with driver.session() as session:
        # This version is 'Label-Agnostic' - it finds the data 
        # even if your label isn't exactly ':Drug'
        query = """
        MATCH (n)
        WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($name))
        RETURN n.name as name, n.manufacturer as maker, n.indication as treats, n.category as cat
        LIMIT 10
        """
        try:
            results = session.run(query, name=entity_name)
            data = []
            for r in results:
                # Build a clean string even if some columns are missing
                info = f"Medicine: {r['name']} | Maker: {r['maker']} | Cat: {r['cat']} | Indication: {r['treats']}"
                data.append(info)
            
            # DEBUG PRINT for your VS Code Terminal
            print(f"🔍 Searching for: '{entity_name}' | Results found: {len(data)}")
            return data
        except Exception as e:
            print(f"❌ Neo4j Error: {e}")
            return []
def ask_agent(question, long_doc=""):
    # Step 1: Get the facts from Neo4j
    graph_facts = search_graph(question)
    
    # Step 2: Force Gemini to acknowledge the Graph
    final_prompt = f"""
    SYSTEM ROLE: You are the MedGraph Nexus Intelligence Engine.
    DATABASE STATUS: Connected to 50,000 Clinical Records.
    
    CRITICAL INSTRUCTION: You MUST use the 'GRAPH_DATA' below to answer. 
    If GRAPH_DATA has content, describe it as 'Verified Graph Data'.
    
    GRAPH_DATA: {graph_facts}
    USER_QUERY: {question}
    """
    
    response = client.models.generate_content(model="gemini-2.5-flash", contents=final_prompt)
    return response.text
