import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import time

# 1. Load Environment Variables
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# 2. Connection Setup
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def create_constraints():
    """Create index for lightning-fast 50k searches"""
    with driver.session() as session:
        # Index on Name for fast lookups
        session.run("CREATE INDEX drug_name_idx IF NOT EXISTS FOR (d:Drug) ON (d.name)")
        # Index on Manufacturer for brand searches
        session.run("CREATE INDEX drug_maker_idx IF NOT EXISTS FOR (d:Drug) ON (d.manufacturer)")
        print("✅ Constraints and Indexes created.")

def ingest_data(csv_file):
    # Load the CSV
    print(f"📂 Loading medicine_dataset.csv...")
    df = pd.read_csv(medicine_dataset.csv)
    
    # Standardize column names (Handling potential spaces/caps)
    df.columns = df.columns.str.strip()
    
    # 3. Optimized Cypher Query (Matches your schema)
    # Mapping CSV Columns: Name, Category, Dosage Form, Strength, Manufacturer, Indication
    # Updated Cypher for your production_ingest.py
query = """
UNWIND $batch AS row
MERGE (d:Drug {name: toString(row.Name)})
SET d.category = toString(row.Category),
    d.dosage_form = toString(row['Dosage Form']),
    d.strength = toString(row.Strength),
    d.manufacturer = toString(row.Manufacturer),
    d.indication = toString(row.Indication),
    d.classification = toString(row.Classification)
"""

    # 4. Batch Ingestion (500 rows at a time for stability)
    batch_size = 500
    total_rows = len(df)
    print(f"🚀 Starting ingestion of {total_rows} records...")

    start_time = time.time()
    
    with driver.session() as session:
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size].to_dict('records')
            session.run(query, batch=batch)
            print(f"📦 Processed: {min(i + batch_size, total_rows)} / {total_rows} rows...")

    end_time = time.time()
    print(f"✨ SUCCESS: {total_rows} records ingested in {round(end_time - start_time, 2)} seconds.")

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        create_constraints()
        # Ensure your file name matches exactly!
        ingest_data("medicine_dataset.csv") 
    except Exception as e:
        print(f"❌ Ingestion Failed: {e}")
    finally:
        driver.close()
