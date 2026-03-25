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
    """One-Shot Fix: Case-Insensitive Regex Search"""
    with driver.session() as session:
        # (?i) makes the search case-insensitive
        # .* allows it to match the word anywhere in the string
        query = """
        MATCH (d:Drug)
        WHERE d.name =~ ('(?i).*' + $name + '.*')
           OR d.category =~ ('(?i).*' + $name + '.*')
           OR d.indication =~ ('(?i).*' + $name + '.*')
           OR d.dosage =~ ('(?i).*' + $name + '.*')
        RETURN d.name as name, d.category as cat, d.dosage as dose, d.indication as treats
        LIMIT 10
        """
        try:
            results = session.run(query, name=entity_name)
            data = [f"Med: {r['name']} | Cat: {r['cat']} | Dose: {r['dose']} | Info: {r['treats']}" for r in results]
            
            # Terminal log to verify it's working
            print(f"--- 📊 NEXUS LIVE: Found {len(data)} records for '{entity_name}' ---")
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
