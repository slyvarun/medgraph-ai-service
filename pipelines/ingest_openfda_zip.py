"""
MedGraph Nexus - Selective openFDA Bulk Zip Ingestion Engine
-----------------------------------------------------------
Processes openFDA bulk partition zip files directly in RAM, applies aggressive
clinical pruning rules to stay within Neo4j Aura Free Tier limits (200,000 nodes),
and executes high-throughput Cypher UNWIND batches.
"""

import os
import sys
import json
import zipfile
import argparse
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment credentials
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
load_dotenv(os.path.join(parent_dir, ".env"))
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")

AURA_FREE_TIER_NODE_CEILING = 180000  # Safe ceiling below 200,000 free-tier cap


def process_openfda_zip(zip_file_path: str) -> pd.DataFrame:
    """
    Extracts and prunes openFDA JSON data directly from the downloaded zip file.
    Applies aggressive clinical pruning rules:
    - Rule 1: Must be a HUMAN PRESCRIPTION DRUG.
    - Rule 2: Must contain non-null rxcui, brand_name, and generic_name.
    - Rule 3: Deduplicates on rxcui to prevent redundant node creation.
    """
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"Specified zip file not found: {zip_file_path}")

    print(f"[*] Processing openFDA archive: {zip_file_path}...")
    records = []

    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # Each openFDA partition zip contains one main JSON file
        json_filenames = [n for n in z.namelist() if n.endswith('.json')]
        if not json_filenames:
            raise ValueError(f"No JSON file found inside archive: {zip_file_path}")

        filename = json_filenames[0]
        print(f"[*] Reading '{filename}' directly from stream...")
        with z.open(filename) as f:
            data = json.load(f)

            for item in data.get('results', []):
                openfda = item.get('openfda', {})

                # RULE 1: Filter strictly for HUMAN PRESCRIPTION DRUG
                # Eliminates thousands of hand sanitizers, sunscreens, and cosmetics
                product_type = openfda.get('product_type', [''])[0]
                if "HUMAN PRESCRIPTION DRUG" not in product_type.upper():
                    continue

                # RULE 2: Must have a valid RxCUI to bridge to our RxNorm graph
                rxcui_list = openfda.get('rxcui', [])
                rxcui = rxcui_list[0] if rxcui_list else None
                brand_list = openfda.get('brand_name', [])
                brand_name = brand_list[0] if brand_list else None
                generic_list = openfda.get('generic_name', [])
                generic_name = generic_list[0] if generic_list else None

                # Extract indications and warnings if present for downstream clinical grounding
                ind_list = item.get('indications_and_usage', [])
                indications_text = ind_list[0][:400] if ind_list else ""
                warn_list = item.get('warnings', [])
                warnings_text = warn_list[0][:400] if warn_list else ""

                if brand_name and generic_name and rxcui:
                    records.append({
                        "brand_name": brand_name.strip().title(),
                        "generic_name": generic_name.strip().title(),
                        "rxcui": str(rxcui).strip(),
                        "indications": indications_text,
                        "warnings": warnings_text
                    })

    if not records:
        print("[!] No prescription drug records met the pruning criteria.")
        return pd.DataFrame()

    # RULE 3: Deduplicate on rxcui to prevent redundant node creation in Aura
    df = pd.DataFrame(records)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['rxcui'])
    print(f"[+] Extracted {len(df):,} clean prescription records (pruned {initial_count - len(df):,} duplicates).")
    return df


def ingest_brands_batch(tx, batch, link_ingredients: bool = True):
    """
    Executes Cypher UNWIND to merge nodes efficiently.
    Tags nodes as both :Brand and :Medicine for seamless compatibility
    with MedGraph Nexus GraphRAG queries, and optionally links to :Ingredient.
    """
    if link_ingredients:
        query = """
        UNWIND $batch AS drug
        MERGE (b:Brand {rxcui: drug.rxcui})
        ON CREATE SET 
            b.name = drug.brand_name,
            b.brand_name = drug.brand_name,
            b.generic_name = drug.generic_name,
            b.indications = drug.indications,
            b.warnings = drug.warnings,
            b.type = 'Prescription'
        ON MATCH SET
            b.indications = coalesce(b.indications, drug.indications),
            b.warnings = coalesce(b.warnings, drug.warnings)
        SET b:Medicine
        WITH b, drug
        WHERE drug.generic_name IS NOT NULL AND drug.generic_name <> ''
        MERGE (i:Ingredient {name: drug.generic_name})
        MERGE (b)-[:CONTAINS]->(i)
        """
    else:
        query = """
        UNWIND $batch AS drug
        MERGE (b:Brand {rxcui: drug.rxcui})
        ON CREATE SET 
            b.name = drug.brand_name,
            b.brand_name = drug.brand_name,
            b.generic_name = drug.generic_name,
            b.indications = drug.indications,
            b.warnings = drug.warnings,
            b.type = 'Prescription'
        ON MATCH SET
            b.indications = coalesce(b.indications, drug.indications),
            b.warnings = coalesce(b.warnings, drug.warnings)
        SET b:Medicine
        """
    tx.run(query, batch=batch)


def run_ingestion(df: pd.DataFrame, batch_size: int = 2000, link_ingredients: bool = True, force: bool = False):
    """
    Chunks the dataframe and pushes to Neo4j Aura with free-tier headroom monitoring.
    """
    if df.empty:
        print("[!] Dataframe is empty. Aborting ingestion.")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    drug_list = df.to_dict('records')

    with driver.session() as session:
        # Safety Check: Verify current Aura node count against 180,000 threshold
        current_node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        print(f"[*] Current Neo4j Aura node count: {current_node_count:,} / 200,000 limit.")

        if current_node_count + len(drug_list) > AURA_FREE_TIER_NODE_CEILING:
            print(f"[WARNING] Ingesting {len(drug_list):,} records would bring total to "
                  f"{current_node_count + len(drug_list):,}, exceeding safe threshold of {AURA_FREE_TIER_NODE_CEILING:,}!")
            if not force:
                confirm = input("Do you wish to continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("[*] Ingestion aborted by user.")
                    driver.close()
                    return

        print(f"[*] Starting batch ingestion of {len(drug_list):,} records in chunks of {batch_size}...")
        for i in range(0, len(drug_list), batch_size):
            batch = drug_list[i:i + batch_size]
            session.execute_write(ingest_brands_batch, batch, link_ingredients=link_ingredients)
            print(f"  -> Ingested records {i + 1} through {min(i + batch_size, len(drug_list))}...")

        final_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        print(f"[SUCCESS] Ingestion batch complete. Current totals: {final_count:,} nodes, {rel_count:,} relationships.")

    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGraph Nexus - Selective openFDA Bulk Zip Ingestor")
    parser.add_argument("--zip-path", type=str, default="", help="Path to downloaded openFDA zip file (e.g. data/drug-label-0001-of-0014.json.zip)")
    parser.add_argument("--all", action="store_true", help="Process and ingest all openFDA partitions found in data/ directory")
    parser.add_argument("--batch-size", type=int, default=2000, help="Cypher UNWIND batch size (default: 2000)")
    parser.add_argument("--no-link", action="store_true", help="Do not create (:Brand)-[:CONTAINS]->(:Ingredient) relationships")
    parser.add_argument("--dry-run", action="store_true", help="Parse and prune data locally without writing to Neo4j")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm prompt if approaching quota")
    args = parser.parse_args()

    data_dir = os.path.join(parent_dir, "data")
    
    if args.all:
        zip_files = []
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.startswith("drug-label") and f.endswith(".zip"):
                    zip_files.append(os.path.join(data_dir, f))
        if not zip_files:
            print("[!] No openFDA zip files found in data/ directory.")
            sys.exit(1)
        print(f"[*] Found {len(zip_files)} openFDA archive partitions to process.")
        
        frames = []
        for zp in zip_files:
            try:
                sub_df = process_openfda_zip(zp)
                if not sub_df.empty:
                    frames.append(sub_df)
            except Exception as e:
                print(f"[!] Error processing {zp}: {e}")
        
        if not frames:
            print("[!] No records extracted across partitions.")
            sys.exit(0)
            
        combined_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['rxcui'])
        print(f"\n[SUMMARY] Total unique pruned prescription drugs across all partitions: {len(combined_df):,}")
        
        if args.dry_run:
            print("[*] Dry run complete. No records written to Neo4j.")
            sys.exit(0)
            
        run_ingestion(combined_df, batch_size=args.batch_size, link_ingredients=(not args.no_link), force=args.yes)
        sys.exit(0)

    target_zip = args.zip_path
    if not target_zip:
        # Search for any candidate openFDA zip file in ./data or ./
        search_dirs = [data_dir, parent_dir]
        for d in search_dirs:
            if os.path.exists(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith("drug-label") and f.endswith(".zip"):
                        target_zip = os.path.join(d, f)
                        break
            if target_zip:
                break

    if not target_zip:
        print("[!] No openFDA zip archive specified or detected.")
        print("Usage:")
        print("  python pipelines/ingest_openfda_zip.py --zip-path data/drug-label-0001-of-0014.json.zip")
        print("  python pipelines/ingest_openfda_zip.py --all")
        print("\nDownload partitions from: https://open.fda.gov/apis/drug/label/download/")
        sys.exit(0)

    df_drugs = process_openfda_zip(target_zip)
    if args.dry_run:
        print(f"[*] Dry run complete. Extracted {len(df_drugs):,} records. Sample:")
        print(df_drugs.head(3))
        sys.exit(0)
        
    run_ingestion(df_drugs, batch_size=args.batch_size, link_ingredients=(not args.no_link), force=args.yes)
