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
    Advanced Fuzzy Search: Searches Manufacturer, Category, and Name 
    across 50,000 nodes using Case-Insensitive Regex.
    """
    with driver.session() as session:
        # We use (?i) for Case-Insensitive Regex in Neo4j
        # This will match 'Roche', 'ROCHE', or 'roche' anywhere in the text
        query = """
        MATCH (d:Drug)
        WHERE d.manufacturer =~ ('(?i).*' + $name + '.*')
           OR d.category =~ ('(?i).*' + $name + '.*')
           OR d.name =~ ('(?i).*' + $name + '.*')
           OR d.indication =~ ('(?i).*' + $name + '.*')
        RETURN d.name as n, d.manufacturer as m, d.indication as i, d.category as c
        LIMIT 15
        """
        try:
            results = session.run(query, name=entity_name)
            data = [f"Medicine: {r['n']} | Maker: {r['m']} | Cat: {r['c']} | Treats: {r['i']}" for r in results]
            
            if not data:
                print(f"🔍 No graph match found for '{entity_name}'. Attempting broader search...")
            
            return data
        except Exception as e:
            print(f"Neo4j Query Error: {e}")
            return []
import time

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
