import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# 1. Database Connection Setup
# We load credentials from the environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class KnowledgeGraphIngestor:
    def __init__(self, uri, user, password):
        # Initialize the Neo4j Driver
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # 2. The Core Cypher Query
    # This query uses MERGE to ensure we don't create duplicate nodes.
    # It dynamically creates the nodes and the relationships between them.
    @staticmethod
    def _create_medicine_graph(tx, medicine_data):
        query = """
        // Create or Match the Medicine Node
        MERGE (med:Medicine {brand_name: $brand_name})
        SET med.generic_name = $generic_name
        
        // Create or Match the Manufacturer Node
        MERGE (man:Manufacturer {name: $manufacturer})
        
        // Create or Match the Route Node
        MERGE (r:Route {name: $route})
        
        // Create or Match the ProductType Node
        MERGE (pt:ProductType {name: $product_type})
        
        // Create or Match the Indication Node
        MERGE (ind:Indication {description: $indications})
        
        // Create the Relationships connecting everything together
        MERGE (med)-[:MANUFACTURED_BY]->(man)
        MERGE (med)-[:ADMINISTERED_VIA]->(r)
        MERGE (med)-[:IS_TYPE]->(pt)
        MERGE (med)-[:TREATS_INDICATION]->(ind)
        """
        
        # Execute the query, passing in the data dictionary as parameters
        tx.run(query, 
               brand_name=medicine_data.get("brand_name"),
               generic_name=medicine_data.get("generic_name"),
               manufacturer=medicine_data.get("manufacturer", "Unknown"),
               route=medicine_data.get("route", "Unknown"),
               product_type=medicine_data.get("product_type", "Unknown"),
               indications=medicine_data.get("indications", "Unknown"))

    # 3. The Execution Loop
    def ingest_data(self, json_filepath):
        # Load the JSON data we fetched earlier
        with open(json_filepath, 'r', encoding='utf-8') as file:
            medicines = json.load(file)
            
        print(f"Starting ingestion of {len(medicines)} medicines...")
        
        # Open a database session
        with self.driver.session() as session:
            for i, med in enumerate(medicines):
                # Run the transaction for each medicine
                session.execute_write(self._create_medicine_graph, med)
                if (i + 1) % 10 == 0:
                    print(f"Ingested {i + 1} records...")
                    
        print("Ingestion complete!")

# 4. Running the script
if __name__ == "__main__":
    # In a real scenario, make sure Neo4j is running and credentials are correct!
    ingestor = KnowledgeGraphIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        ingestor.ingest_data("medicine_dataset.json")
    finally:
        ingestor.close()
