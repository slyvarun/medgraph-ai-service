import os
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()

# 2026 Production Config
# Using the stable alias or the specific preview ID
MODEL_ID = "gemini-1.5-flash" 

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def search_graph(entity_name):
    """
    Matched to your 4 existing keys: 
    name, category, strength, indication
    """
    with driver.session() as session:
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.name) CONTAINS toLower($name)
           OR toLower(d.category) CONTAINS toLower($name)
           OR toLower(d.indication) CONTAINS toLower($name)
        RETURN d.name as name, 
               d.category as cat, 
               d.indication as treats, 
               d.strength as str
        LIMIT 10
        """
        try:
            results = session.run(query, name=entity_name)
            data = [f"Med: {r['name']} | Cat: {r['cat']} | Info: {r['treats']} | Str: {r['str']}" for r in results]
            print(f"--- 📊 NEXUS LIVE: Found {len(data)} records for '{entity_name}' ---")
            return data
        except Exception as e:
            print(f"❌ Neo4j Search Error: {e}")
            return []
def ask_agent(question, long_doc=""):
    # RE-INITIALIZE variables inside the function every time it's called
    current_query = str(question).strip() 
    
    # Force a fresh graph search
    facts = search_graph(current_query)
    
    # Create a fresh prompt
    prompt = f"""
    CONTEXT: {facts if facts else "No data found for " + current_query}
    USER QUERY: {current_query}
    """
    
    # Use the model to generate a fresh answer
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text
