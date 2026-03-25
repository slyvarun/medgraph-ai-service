import os
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()

# Global Singleton for Driver (Speed Optimization)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def search_graph(entity_name):
    """Fuzzy search across 50,000 nodes."""
    with driver.session() as session:
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.name) CONTAINS toLower($name)
        OR toLower(d.indication) CONTAINS toLower($name)
        RETURN d.name as name, d.indication as treats, d.side_effects as effects, d.manufacturer as maker
        LIMIT 5
        """
        results = session.run(query, name=entity_name)
        return [f"Fact: {r['name']} by {r['maker']} treats {r['treats']}." for r in results]

def ask_agent(question, long_doc=""):
    # 1. Extract Medical Intent
    intent_prompt = f"Extract the medicine name from: '{question}'. Reply ONLY with the name."
    intent_res = client.models.generate_content(model="gemini-3-flash", contents=intent_prompt)
    target = intent_res.text.strip()
    
    # 2. Query Neo4j
    facts = search_graph(target)
    
    # 3. Final Synthesis
    final_prompt = f"""
    You are MedGraph Nexus AI. Answer using the clinical facts provided. 
    If no facts are found, use your internal training but mention it's not in the graph.
    
    NEO4J_FACTS: {facts}
    RESEARCH_DOC: {long_doc}
    QUESTION: {question}
    """
    
    response = client.models.generate_content(model="gemini-2.0-flash", contents=final_prompt)
    return response.text
