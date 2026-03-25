import os
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()
# Using the 2026 Standard: Gemini 3 Flash
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def search_graph(entity_name):
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    with driver.session() as session:
        query = """
        MATCH (e1:MedicalEntity {name: $name})-[r]->(e2)
        RETURN e1.name as source, type(r) as relationship, e2.name as target
        """
        results = session.run(query, name=entity_name)
        data = [f"{record['source']} {record['relationship']} {record['target']}" for record in results]
    driver.close()
    return data

def ask_agent(question, long_document_text=""):
    # Phase 1: Identify Entity
    intent_prompt = f"Identify the primary medical entity in this question: '{question}'. Output ONLY the name."
    entity_res = client.models.generate_content(model="gemini-3-flash-preview", contents=intent_prompt)
    target_entity = entity_res.text.strip()
    
    print(f"🔍 Agent is searching Graph for: {target_entity}...")
    graph_facts = search_graph(target_entity)
    
    # Phase 2: Final Answer with Long-Context Fallback
    # We provide the graph facts AND the long document to Gemini
    context_data = f"GRAPH FACTS: {graph_facts}\n\nLONG_DOC_REFERENCE: {long_document_text}"
    
    final_prompt = f"""
    You are a Medical AI. Use the provided context to answer the question.
    Prioritize GRAPH FACTS. If they are missing, use the LONG_DOC_REFERENCE.
    
    CONTEXT:
    {context_data}
    
    QUESTION: {question}
    """
    
    # NEW VERSION (Returns the text)
    final_res = client.models.generate_content(model="gemini-3-flash-preview", contents=final_prompt)
    return final_res.text  # This sends the text back to the API
if __name__ == "__main__":
    # Example: A long research snippet that ISN'T in your graph yet
    extra_research = "Recent 2026 studies suggest Metformin also shows potential in longevity research by activating AMPK pathways."
    
    query = "What can you tell me about Metformin?"
    ask_agent(query, extra_research)