import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

print(f"Connecting to Neo4j at {NEO4J_URI} as {NEO4J_USER}...")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Total counts
        total_nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        print(f"\n--- TOTAL STATS ---")
        print(f"Total Nodes: {total_nodes}")
        print(f"Total Relationships: {total_rels}")

        # Node labels count
        print(f"\n--- NODE LABELS & COUNTS ---")
        labels_res = session.run("""
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(n) AS count
            ORDER BY count DESC
        """)
        labels = []
        for r in labels_res:
            labels.append(r["label"])
            print(f"- {r['label']}: {r['count']}")

        # Relationship types count
        print(f"\n--- RELATIONSHIP TYPES & COUNTS ---")
        rels_res = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """)
        for r in rels_res:
            print(f"- {r['rel_type']}: {r['count']}")

        # Graph Schema (Source label -> Relationship -> Target label)
        print(f"\n--- SCHEMA / PATH PATTERNS (Source -[REL]-> Target) ---")
        schema_res = session.run("""
            MATCH (a)-[r]->(b)
            RETURN head(labels(a)) AS src, type(r) AS rel, head(labels(b)) AS tgt, count(*) AS count
            ORDER BY count DESC
        """)
        for r in schema_res:
            print(f"- (:{r['src']})-[:{r['rel']}]->(:{r['tgt']}) : {r['count']}")

        # Sample properties for each label
        print(f"\n--- SAMPLE PROPERTIES BY LABEL ---")
        for lbl in labels:
            sample_res = session.run(f"MATCH (n:`{lbl}`) RETURN properties(n) AS props LIMIT 1")
            record = sample_res.single()
            if record:
                props = record["props"]
                # print keys and sample values (truncated)
                summary = {k: (str(v)[:60] + '...' if len(str(v)) > 60 else v) for k, v in props.items()}
                print(f"\nLabel: {lbl}")
                for k, v in summary.items():
                    print(f"   {k}: {repr(v)}")

    driver.close()
    print("\nSuccessfully queried Neo4j!")
except Exception as e:
    print(f"Error querying Neo4j: {e}")
