import json
import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Database Connection Setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class KnowledgeGraphIngestor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints_and_indexes(self):
        """Create schema constraints and indexes for high performance querying."""
        print("Ensuring constraints and indexes exist in Neo4j...")
        constraints = [
            "CREATE CONSTRAINT medicine_brand_unique IF NOT EXISTS FOR (m:Medicine) REQUIRE m.brand_name IS UNIQUE",
            "CREATE CONSTRAINT manufacturer_name_unique IF NOT EXISTS FOR (m:Manufacturer) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT route_name_unique IF NOT EXISTS FOR (r:Route) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT product_type_unique IF NOT EXISTS FOR (pt:ProductType) REQUIRE pt.name IS UNIQUE",
            "CREATE CONSTRAINT substance_name_unique IF NOT EXISTS FOR (s:ActiveSubstance) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT drug_class_unique IF NOT EXISTS FOR (c:DrugClass) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (cat:TherapeuticCategory) REQUIRE cat.name IS UNIQUE",
            "CREATE INDEX medicine_generic_idx IF NOT EXISTS FOR (m:Medicine) ON (m.generic_name)"
        ]
        
        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    pass
        print("Schema constraints and indexes verified!")

    @staticmethod
    def _ingest_batch(tx, batch):
        """Batched Cypher query with UNWIND for multi-entity creation."""
        cypher = """
        UNWIND $batch AS med
        
        // 1. Medicine Node
        MERGE (m:Medicine {brand_name: med.brand_name})
        SET m.generic_name = med.generic_name,
            m.category = coalesce(med.category, 'General'),
            m.warnings = med.warnings,
            m.adverse_reactions = med.adverse_reactions,
            m.contraindications = med.contraindications,
            m.drug_interactions = med.drug_interactions
            
        // 2. Peripheral Nodes
        MERGE (man:Manufacturer {name: coalesce(med.manufacturer, 'Unknown')})
        MERGE (r:Route {name: coalesce(med.route, 'Oral')})
        MERGE (pt:ProductType {name: coalesce(med.product_type, 'Prescription')})
        MERGE (cat:TherapeuticCategory {name: coalesce(med.category, 'General')})
        MERGE (ind:Indication {description: coalesce(med.indications, 'Unspecified')})
        
        // 3. Relationships
        MERGE (m)-[:MANUFACTURED_BY]->(man)
        MERGE (m)-[:ADMINISTERED_VIA]->(r)
        MERGE (m)-[:IS_TYPE]->(pt)
        MERGE (m)-[:IN_CATEGORY]->(cat)
        MERGE (m)-[:TREATS_INDICATION]->(ind)
        
        // 4. Active Substances
        FOREACH (sub_name IN med.substances |
            MERGE (sub:ActiveSubstance {name: sub_name})
            MERGE (m)-[:CONTAINS_SUBSTANCE]->(sub)
        )

        // 5. Pharmacologic Classes
        FOREACH (cls_name IN med.pharm_classes |
            MERGE (cls:DrugClass {name: cls_name})
            MERGE (m)-[:BELONGS_TO_CLASS]->(cls)
        )
        """
        tx.run(cypher, batch=batch)

    def ingest_data(self, json_filepath, batch_size=200):
        """Read JSON dataset and ingest into Neo4j in high-speed batches."""
        if not os.path.exists(json_filepath):
            print(f"Error: File {json_filepath} not found.")
            return

        with open(json_filepath, "r", encoding="utf-8") as f:
            medicines = json.load(f)

        total = len(medicines)
        print(f"\n=== Starting High-Speed Batch Ingestion of {total} Medicines ===")
        start_time = time.time()
        
        # 1. Setup Constraints
        self.create_constraints_and_indexes()

        # 2. Ingest in Batches
        with self.driver.session() as session:
            for start_idx in range(0, total, batch_size):
                batch = medicines[start_idx : start_idx + batch_size]
                session.execute_write(self._ingest_batch, batch)
                print(f"Ingested records {min(start_idx + batch_size, total)}/{total}...")

        elapsed = round(time.time() - start_time, 2)
        print(f"Ingestion successfully completed in {elapsed} seconds!")

    def verify_graph_counts(self):
        """Print summary of nodes and relationships in the database."""
        with self.driver.session() as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            med_count = session.run("MATCH (m:Medicine) RETURN count(m) AS cnt").single()["cnt"]
            sub_count = session.run("MATCH (s:ActiveSubstance) RETURN count(s) AS cnt").single()["cnt"]
            cls_count = session.run("MATCH (c:DrugClass) RETURN count(c) AS cnt").single()["cnt"]
            cat_count = session.run("MATCH (cat:TherapeuticCategory) RETURN count(cat) AS cnt").single()["cnt"]
            ind_count = session.run("MATCH (i:Indication) RETURN count(i) AS cnt").single()["cnt"]
            
            print("\n--- Knowledge Graph Ultrascale Summary ---")
            print(f"Total Nodes: {total_nodes}")
            print(f"Total Relationships: {total_rels}")
            print(f"Medicines: {med_count}")
            print(f"Active Substances: {sub_count}")
            print(f"Pharmacologic Drug Classes: {cls_count}")
            print(f"Therapeutic Categories: {cat_count}")
            print(f"Indications: {ind_count}")
            print("-------------------------------------------\n")

if __name__ == "__main__":
    ingestor = KnowledgeGraphIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        ingestor.ingest_data("medicine_dataset.json", batch_size=200)
        ingestor.verify_graph_counts()
    finally:
        ingestor.close()
