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
