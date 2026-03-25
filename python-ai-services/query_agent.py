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

import time

def ask_agent(question, long_doc=""):
    # List of models to try in order of preference
    models_to_try = ["gemini-3-flash-preview", "gemini-2.0-flash", "gemini-1.5-flash"]
    
    for model_id in models_to_try:
        try:
            # 1. Extraction Phase
            intent_res = client.models.generate_content(
                model=model_id,
                contents=f"Extract the medicine name from: '{question}'. Reply ONLY with the name."
            )
            target = intent_res.text.strip()
            
            # 2. Graph Retrieval
            facts = search_graph(target)
            
            # 3. Synthesis Phase
            final_prompt = f"""
            You are MedGraph Nexus. 
            CONTEXT: {facts if facts else "No specific graph data found."}
            QUERY: {question}
            """
            
            response = client.models.generate_content(model=model_id, contents=final_prompt)
            return response.text
            
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"⚠️ {model_id} busy, failing over to next model...")
                continue # Try the next model in the list
            else:
                return f"Nexus Engine Error: {str(e)}"
    
    return "All Nexus engines are currently at capacity. Please standby for 30 seconds."
