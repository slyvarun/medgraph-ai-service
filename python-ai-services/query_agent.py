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
