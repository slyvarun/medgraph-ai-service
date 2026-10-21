import json
import os
import time
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Load environment variables
load_dotenv(os.path.join(parent_dir, ".env"))
load_dotenv(os.path.join(parent_dir, "backend", ".env"))
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class KnowledgeGraphIngestor:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def test_connection(self) -> bool:
        """Verify Neo4j connectivity."""
        try:
            with self.driver.session() as session:
                res = session.run("RETURN 1 AS ping").single()
                return bool(res and res["ping"] == 1)
        except Exception as e:
            print(f"Connection test failed for {self.uri}: {e}")
            return False

    def clear_database(self):
        """Clean all old nodes and relationships for a zero-duplicate re-seed."""
        print("Clearing database for fresh graph ingestion...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared successfully!")

    def create_constraints_and_indexes(self):
        """Create schema constraints and text indexes for maximum query performance."""
        print("Ensuring schema constraints and text indexes in Neo4j...")
        constraints = [
            "CREATE CONSTRAINT medicine_brand_unique IF NOT EXISTS FOR (m:Medicine) REQUIRE m.brand_name IS UNIQUE",
            "CREATE CONSTRAINT manufacturer_name_unique IF NOT EXISTS FOR (m:Manufacturer) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT route_name_unique IF NOT EXISTS FOR (r:Route) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT product_type_unique IF NOT EXISTS FOR (pt:ProductType) REQUIRE pt.name IS UNIQUE",
            "CREATE CONSTRAINT substance_name_unique IF NOT EXISTS FOR (s:ActiveSubstance) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT drug_class_unique IF NOT EXISTS FOR (c:DrugClass) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (cat:TherapeuticCategory) REQUIRE cat.name IS UNIQUE",
            "CREATE CONSTRAINT rxcui_unique IF NOT EXISTS FOR (rc:RxNormCUI) REQUIRE rc.rxcui IS UNIQUE",
            "CREATE CONSTRAINT active_ingredient_unique IF NOT EXISTS FOR (ai:ActiveIngredient) REQUIRE ai.rxcui IS UNIQUE",
            "CREATE CONSTRAINT alt_brand_unique IF NOT EXISTS FOR (ab:AlternativeBrand) REQUIRE ab.name IS UNIQUE",
            "CREATE INDEX medicine_generic_idx IF NOT EXISTS FOR (m:Medicine) ON (m.generic_name)"
        ]
        
        indexes = [
            "CREATE TEXT INDEX medicine_brand_text_idx IF NOT EXISTS FOR (m:Medicine) ON (m.brand_name)",
            "CREATE TEXT INDEX medicine_generic_text_idx IF NOT EXISTS FOR (m:Medicine) ON (m.generic_name)",
            "CREATE TEXT INDEX substance_name_text_idx IF NOT EXISTS FOR (s:ActiveSubstance) ON (s.name)",
            "CREATE TEXT INDEX drug_class_text_idx IF NOT EXISTS FOR (c:DrugClass) ON (c.name)",
            "CREATE TEXT INDEX indication_desc_text_idx IF NOT EXISTS FOR (i:Indication) ON (i.description)"
        ]
        
        with self.driver.session() as session:
            for c in constraints + indexes:
                try:
                    session.run(c)
                except Exception:
                    pass
        print("Schema constraints and indexes active!")

    @staticmethod
    def _ingest_batch(tx, batch: List[Dict[str, Any]]):
        """Batched Cypher query linking Medicines, RxNorm CUIs, Ingredients, and Alternative Brands."""
        cypher = """
        UNWIND $batch AS med
        
        // 1. Medicine Primary Node
        MERGE (m:Medicine {brand_name: med.brand_name})
        SET m.generic_name = coalesce(med.generic_name, med.brand_name),
            m.category = coalesce(med.category, 'General Medicine'),
            m.warnings = coalesce(med.warnings, 'None specified in label summary'),
            m.adverse_reactions = coalesce(med.adverse_reactions, 'None specified in label summary'),
            m.contraindications = coalesce(med.contraindications, 'None specified in label summary'),
            m.drug_interactions = coalesce(med.drug_interactions, 'None specified in label summary'),
            m.indications = coalesce(med.indications, ''),
            m.rxcui = coalesce(med.rxcui, '')
            
        // 2. Peripheral Taxonomy Nodes
        MERGE (man:Manufacturer {name: coalesce(med.manufacturer, 'Unknown')})
        MERGE (r:Route {name: coalesce(med.route, 'Oral')})
        MERGE (pt:ProductType {name: coalesce(med.product_type, 'HUMAN PRESCRIPTION DRUG')})
        MERGE (cat:TherapeuticCategory {name: coalesce(med.category, 'General Medicine')})
        
        // 3. Peripheral Relationships
        MERGE (m)-[:MANUFACTURED_BY]->(man)
        MERGE (m)-[:ADMINISTERED_VIA]->(r)
        MERGE (m)-[:IS_TYPE]->(pt)
        MERGE (m)-[:IN_CATEGORY]->(cat)
        
        // 4. Clinical Indication Node & Relationship
        FOREACH (ind_text IN CASE WHEN med.indications IS NOT NULL AND med.indications <> '' THEN [med.indications] ELSE [] END |
            MERGE (ind:Indication {description: ind_text})
            MERGE (m)-[:TREATS_INDICATION]->(ind)
        )
        
        // 5. Active Substances from openFDA
        FOREACH (sub_name IN coalesce(med.substances, []) |
            MERGE (sub:ActiveSubstance {name: sub_name})
            MERGE (m)-[:CONTAINS_SUBSTANCE]->(sub)
        )

        // 6. Pharmacologic Classes from openFDA EPC
        FOREACH (cls_name IN coalesce(med.pharm_classes, []) |
            MERGE (cls:DrugClass {name: cls_name})
            MERGE (m)-[:BELONGS_TO_CLASS]->(cls)
        )

        // 7. Product RXCUIs
        FOREACH (cui IN coalesce(med.product_rxcuis, CASE WHEN med.rxcui IS NOT NULL AND med.rxcui <> '' THEN [med.rxcui] ELSE [] END) |
            MERGE (rc:RxNormCUI {rxcui: cui})
            MERGE (m)-[:HAS_RXCUI]->(rc)
        )

        // 8. Active Ingredient Nodes (RxNorm IN/MIN) and Alternative Brands
        FOREACH (ing IN coalesce(med.active_ingredients, []) |
            MERGE (ai:ActiveIngredient {rxcui: ing.rxcui})
            SET ai.name = ing.name
            MERGE (m)-[:HAS_ACTIVE_INGREDIENT]->(ai)
        )

        // 9. Alternative Brand Formulations
        FOREACH (alt_brand IN coalesce(med.alternative_brand_names, coalesce(med.known_brands, [])) |
            MERGE (ab:AlternativeBrand {name: alt_brand})
            MERGE (m)-[:HAS_ALTERNATIVE_BRAND]->(ab)
        )

        // 10. Link Active Ingredients directly to Alternative Brands
        FOREACH (ing IN coalesce(med.active_ingredients, []) |
            FOREACH (alt_brand IN coalesce(med.alternative_brand_names, []) |
                MERGE (ai_node:ActiveIngredient {rxcui: ing.rxcui})
                MERGE (ab_node:AlternativeBrand {name: alt_brand})
                MERGE (ai_node)-[:HAS_ALTERNATIVE_BRAND]->(ab_node)
            )
        )
        """
        tx.run(cypher, batch=batch)

    def ingest_data(self, json_filepath: str, batch_size: int = 200, clean_first: bool = True):
        """Read JSON dataset and ingest into Neo4j in high-speed batches."""
        if not os.path.exists(json_filepath):
            print(f"Error: File {json_filepath} not found.")
            return

        with open(json_filepath, "r", encoding="utf-8") as f:
            medicines = json.load(f)

        total = len(medicines)
        print(f"\n=== Starting Batch Ingestion of {total} Medicines ===")
        start_time = time.time()
        
        if clean_first:
            self.clear_database()

        self.create_constraints_and_indexes()

        with self.driver.session() as session:
            for start_idx in range(0, total, batch_size):
                batch = medicines[start_idx : start_idx + batch_size]
                for attempt in range(4):
                    try:
                        session.execute_write(self._ingest_batch, batch)
                        break
                    except Exception as e:
                        if attempt == 3:
                            print(f"Warning on batch {start_idx}: {e}")
                        else:
                            time.sleep(2 * (attempt + 1))
                print(f"Ingested records {min(start_idx + batch_size, total)}/{total}...")

        elapsed = round(time.time() - start_time, 2)
        print(f"Ingestion successfully completed in {elapsed} seconds!")

    def ingest_live_drugs(self, brand_names: List[str]):
        """Fetch live records from openFDA & RxNorm and ingest into Neo4j graph."""
        try:
            from fetch_drug_relations import extract_alternative_brands_for_drug
        except ImportError:
            from pipelines.fetch_drug_relations import extract_alternative_brands_for_drug

        records = []
        for name in brand_names:
            print(f"Harvesting live openFDA + RxNorm data for: {name}...")
            data = extract_alternative_brands_for_drug(name)
            if "error" in data:
                print(f"  -> Skipping {name}: {data['error']}")
                continue

            fda = data["openfda"]
            rec = {
                "brand_name": fda["brand_name"][0] if fda["brand_name"] else name,
                "generic_name": fda["substance_name"][0] if fda["substance_name"] else name,
                "substances": fda["substance_name"],
                "indications": fda["indications_and_usage"][:1000],
                "product_rxcuis": fda["product_rxcuis"],
                "rxcui": fda["product_rxcuis"][0] if fda["product_rxcuis"] else "",
                "active_ingredients": data.get("active_ingredients", []),
                "alternative_brand_names": data.get("alternative_brand_names", [])
            }
            records.append(rec)

        if records:
            print(f"Ingesting {len(records)} live clinical entities into Neo4j...")
            self.create_constraints_and_indexes()
            with self.driver.session() as session:
                session.execute_write(self._ingest_batch, records)
            print("Live entity ingestion complete!")

    def verify_graph_counts(self):
        """Print summary of nodes and relationships in the database."""
        with self.driver.session() as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            med_count = session.run("MATCH (m:Medicine) RETURN count(m) AS cnt").single()["cnt"]
            sub_count = session.run("MATCH (s:ActiveSubstance) RETURN count(s) AS cnt").single()["cnt"]
            ai_count = session.run("MATCH (ai:ActiveIngredient) RETURN count(ai) AS cnt").single()["cnt"]
            ab_count = session.run("MATCH (ab:AlternativeBrand) RETURN count(ab) AS cnt").single()["cnt"]
            rxcui_count = session.run("MATCH (rc:RxNormCUI) RETURN count(rc) AS cnt").single()["cnt"]
            cls_count = session.run("MATCH (c:DrugClass) RETURN count(c) AS cnt").single()["cnt"]
            cat_count = session.run("MATCH (cat:TherapeuticCategory) RETURN count(cat) AS cnt").single()["cnt"]
            ind_count = session.run("MATCH (i:Indication) RETURN count(i) AS cnt").single()["cnt"]
            
            print("\n--- Knowledge Graph Summary ---")
            print(f"Total Nodes: {total_nodes}")
            print(f"Total Relationships: {total_rels}")
            print(f"Medicines: {med_count}")
            print(f"Active Substances: {sub_count}")
            print(f"Active Ingredients (RxNorm): {ai_count}")
            print(f"Alternative Brands: {ab_count}")
            print(f"RxNorm CUIs: {rxcui_count}")
            print(f"Pharmacologic Drug Classes: {cls_count}")
            print(f"Therapeutic Categories: {cat_count}")
            print(f"Indications: {ind_count}")
            print("--------------------------------\n")


if __name__ == "__main__":
    candidates = [
        os.path.join(parent_dir, "data", "medicine_dataset.json"),
        os.path.join(base_dir, "data", "medicine_dataset.json"),
        os.path.join(parent_dir, "medicine_dataset.json"),
        os.path.join(base_dir, "medicine_dataset.json"),
        "data/medicine_dataset.json",
        "medicine_dataset.json"
    ]
    target_path = next((p for p in candidates if os.path.exists(p)), None)
    
    ingestor = KnowledgeGraphIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        if not ingestor.test_connection():
            print(f"\n[!] Cannot connect to Neo4j instance at {NEO4J_URI}.")
            print("Please ensure Neo4j is running and accessible with your credentials.")
        else:
            if target_path:
                print(f"Loading verified clinical dataset from: {target_path}")
                ingestor.ingest_data(target_path, batch_size=200, clean_first=False)
            
            # Example live ingestion for top benchmark drugs
            print("\nIngesting live openFDA + RxNorm relationships for benchmark drugs...")
            ingestor.ingest_live_drugs(["Lipitor", "Tylenol", "Advil", "Metformin"])
            ingestor.verify_graph_counts()
    finally:
        ingestor.close()
