from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USER, PWD))
with driver.session() as s:
    print("--- LIVE AUDIT: MEDGRAPH NEXUS AURADB FREE TIER ---")
    nodes = s.run("MATCH (n) RETURN labels(n) as lbl, count(*) as count ORDER BY count DESC")
    print("\nNode Counts by Label:")
    for r in nodes:
        lbl = ":".join(r["lbl"])
        print(f"  (:{lbl}): {r['count']:,}")

    rels = s.run("MATCH ()-[r]->() RETURN type(r) as t, count(*) as count ORDER BY count DESC")
    print("\nRelationship Counts by Type:")
    for r in rels:
        print(f"  [:{r['t']}]: {r['count']:,}")

    tot_n = s.run("MATCH (n) RETURN count(n) as c").single()["c"]
    tot_r = s.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
    print(f"\n==========================================")
    print(f"TOTAL NODES:         {tot_n:,} / 200,000 ({(tot_n/200000)*100:.2f}%)")
    print(f"TOTAL RELATIONSHIPS: {tot_r:,} / 400,000 ({(tot_r/400000)*100:.2f}%)")
    print(f"AURA SAFE HEADROOM:  {180000 - tot_n:,} nodes remaining")
    print(f"==========================================")

driver.close()
