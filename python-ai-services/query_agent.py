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
    # If Gemini couldn't extract a name, we search for the whole query
    search_term = entity_name if entity_name else "Medicine"
    
    with driver.session() as session:
        # 2026 Optimized Regex Search
        query = """
        MATCH (d:Drug)
        WHERE d.name =~ ('(?i).*' + $name + '.*')
           OR d.manufacturer =~ ('(?i).*' + $name + '.*')
        RETURN d.name as n, d.manufacturer as m, d.indication as i, d.category as c
        LIMIT 10
        """
        results = session.run(query, name=search_term)
        data = [f"Found: {r['n']} | Maker: {r['m']} | Treats: {r['i']}" for r in results]
        
        # DEBUG: This will show in your VS Code terminal
        print(f"--- GRAPH SEARCH FOR '{search_term}' ---")
        print(f"Nodes found: {len(data)}")
        
        return data
def ask_agent(question, long_doc=""):
    try:
        # Phase 1: Direct Generation (Fast & Stable)
        # 2.5 Flash is optimized for high-volume RAG tasks
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                f"You are MedGraph Nexus. Use these clinical graph facts: {search_graph(question)}",
                f"User Question: {question}"
            ]
        )
        return response.text
    except Exception as e:
        print(f"Server Error: {e}")
        return "Nexus is currently recalibrating. Please try again in a moment."
