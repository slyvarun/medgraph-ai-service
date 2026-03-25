import os
from dotenv import load_dotenv
from google import genai  # Updated import
from neo4j import GraphDatabase

# 1. Load credentials
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")

# 2. Initialize the New GenAI Client
# Passing the api_key here fixes the 'DefaultCredentialsError'
# Updated Line 15
client = genai.Client(
    api_key=GEMINI_KEY,
    http_options={'api_version': 'v1'}
)

# 3. Neo4j Logic (remains the same)
def save_relationship(entity1, relation, entity2):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    with driver.session() as session:
        query = """
        MERGE (a:MedicalEntity {name: $e1})
        MERGE (b:MedicalEntity {name: $e2})
        MERGE (a)-[r:RELATION {type: $rel}]->(b)
        RETURN a, r, b
        """
        session.run(query, e1=entity1, rel=relation, e2=entity2)
    driver.close()

# 4. Updated Extraction Layer
def extract_and_store(text):
    prompt = f"""
    Act as a Medical Data Engineer. Extract key medical entities and their 
    relationships from the text below. 
    Format the output EXACTLY as: Entity1 | Relationship | Entity2
    Example: Insulin | REGULATES | Glucose
    
    Text: {text}
    """
    
    # New syntax for generating content in 2026
    # Using the 2.5 series (Standard for 2026)
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )
    
    lines = response.text.strip().split('\n')
    for line in lines:
        if "|" in line:
            try:
                parts = line.split("|")
                e1, rel, e2 = parts[0].strip(), parts[1].strip(), parts[2].strip()
                print(f"✅ Storing: {e1} --[{rel}]--> {e2}")
                save_relationship(e1, rel, e2)
            except Exception as e:
                print(f"❌ Error parsing line: {line} - {e}")

# 5. Run the Test
if __name__ == "__main__":
    sample_text = "Metformin is commonly used to treat Type 2 Diabetes, but it may cause Vitamin B12 deficiency."
    print("🚀 Starting Ingestion Engine...")
    extract_and_store(sample_text)
    print("✨ Process Complete. Check your Neo4j Browser!")