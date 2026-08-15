"""
production_ingest.py  —  MedGraph Nexus  |  Data Pipeline
==========================================================
Reads medicine_dataset.csv and builds a full relational Knowledge Graph
in Neo4j with entity nodes (:Medicine, :Category, :Indication, :Manufacturer, :DosageForm)
and relationships (:TREATS_INDICATION, :BELONGS_TO_CATEGORY, :MANUFACTURED_BY, :AVAILABLE_AS).

Also exports a local graph JSON cache (medgraph_cache.json) for offline embedded GraphRAG fallback.

Usage:
    python production_ingest.py                          # default CSV path
    python production_ingest.py --file /path/to/csv     # custom path
    python production_ingest.py --batch-size 1000        # larger batches
    python production_ingest.py --dry-run               # validate only, no writes
    python production_ingest.py --clear                 # wipe graph first
"""

import os
import sys
import time
import json
import argparse
import logging

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
)
log = logging.getLogger("ingest")

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

DEFAULT_CSV        = "medicine_dataset.csv"
DEFAULT_BATCH_SIZE = 500
CACHE_JSON_PATH    = "medgraph_cache.json"

COLUMN_MAP = {
    "Name":           "name",            # MERGE key
    "Category":       "category",
    "Dosage Form":    "dosage_form",
    "Strength":       "strength",
    "Manufacturer":   "manufacturer",
    "Indication":     "indication",
    "Classification": "classification",
}

# ── Cypher Schema & Ingest Queries ───────────────────────────────────────────

INDEX_CYPHERS = [
    "CREATE INDEX medicine_name IF NOT EXISTS FOR (m:Medicine) ON (m.name)",
    "CREATE INDEX indication_name IF NOT EXISTS FOR (i:Indication) ON (i.name)",
    "CREATE INDEX category_name IF NOT EXISTS FOR (c:Category) ON (c.name)",
    "CREATE INDEX mfg_name IF NOT EXISTS FOR (mf:Manufacturer) ON (mf.name)",
    "CREATE INDEX dosage_name IF NOT EXISTS FOR (df:DosageForm) ON (df.name)",
]

# Entity & Relational Ingest Query
UPSERT_RELATIONAL_BATCH = """
UNWIND $batch AS row
MERGE (m:Medicine {name: row.name})
SET
    m.strength       = row.strength,
    m.classification = row.classification,
    m.category       = row.category,
    m.indication     = row.indication,
    m.dosage_form    = row.dosage_form,
    m.manufacturer   = row.manufacturer

WITH m, row
WHERE row.category IS NOT NULL AND row.category <> ''
MERGE (c:Category {name: row.category})
MERGE (m)-[:BELONGS_TO_CATEGORY]->(c)

WITH m, row
WHERE row.indication IS NOT NULL AND row.indication <> ''
MERGE (i:Indication {name: row.indication})
MERGE (m)-[:TREATS_INDICATION]->(i)

WITH m, row
WHERE row.category IS NOT NULL AND row.category <> '' AND row.indication IS NOT NULL AND row.indication <> ''
MATCH (c:Category {name: row.category})
MATCH (i:Indication {name: row.indication})
MERGE (c)-[:PRIMARY_TREATMENT_FOR]->(i)

WITH m, row
WHERE row.manufacturer IS NOT NULL AND row.manufacturer <> ''
MERGE (mf:Manufacturer {name: row.manufacturer})
MERGE (m)-[:MANUFACTURED_BY]->(mf)

WITH m, row
WHERE row.dosage_form IS NOT NULL AND row.dosage_form <> ''
MERGE (df:DosageForm {name: row.dosage_form})
MERGE (m)-[:AVAILABLE_AS]->(df)
"""

CLEAR_ALL = "MATCH (n) DETACH DELETE n"

COUNT_GRAPH_NODES = """
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS total
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_env() -> bool:
    missing = [k for k in ("NEO4J_URI", "NEO4J_PASSWORD") if not os.getenv(k)]
    if missing:
        log.warning(f"Missing .env variables: {', '.join(missing)}")
        return False
    return True


def load_and_clean(filepath: str) -> list[dict]:
    """Read CSV, rename columns, normalise strings, return list of dicts."""
    log.info(f"Reading CSV dataset: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("UTF-8 failed — retrying with latin-1")
        df = pd.read_csv(filepath, encoding="latin-1")

    log.info(f"Loaded {len(df):,} rows | columns: {df.columns.tolist()}")

    existing_renames = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    target_cols = list(COLUMN_MAP.values())
    for col in target_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[target_cols].copy()

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": "", "None": "", "none": ""})
        )

    df = df[df["name"].str.len() > 0].reset_index(drop=True)
    log.info(f"Clean records ready for graph ingestion: {len(df):,}")
    return df.to_dict("records")


def export_local_json_cache(records: list[dict], output_path: str = CACHE_JSON_PATH):
    """
    Build an embedded JSON Knowledge Graph cache so local RAG can run
    even when Neo4j is offline.
    """
    log.info(f"Building local Knowledge Graph cache at: {output_path}")

    # Build unique nodes and indexed indices
    medicines = {}
    indications = {}
    categories = {}
    manufacturers = {}
    dosage_forms = {}

    for r in records:
        name = r["name"]
        if name not in medicines:
            medicines[name] = {
                "name": name,
                "classification": r.get("classification", ""),
                "category": r.get("category", ""),
                "indication": r.get("indication", ""),
                "dosage_form": r.get("dosage_form", ""),
                "strength": r.get("strength", ""),
                "manufacturer": r.get("manufacturer", "")
            }

        ind = r.get("indication")
        if ind and ind not in indications:
            indications[ind] = {"name": ind, "type": "Indication"}

        cat = r.get("category")
        if cat and cat not in categories:
            categories[cat] = {"name": cat, "type": "Category"}

        mfg = r.get("manufacturer")
        if mfg and mfg not in manufacturers:
            manufacturers[mfg] = {"name": mfg, "type": "Manufacturer"}

        df = r.get("dosage_form")
        if df and df not in dosage_forms:
            dosage_forms[df] = {"name": df, "type": "DosageForm"}

    cache_data = {
        "metadata": {
            "total_medicines": len(medicines),
            "unique_indications": len(indications),
            "unique_categories": len(categories),
            "unique_manufacturers": len(manufacturers),
            "unique_dosage_forms": len(dosage_forms),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "records": list(medicines.values())
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    log.info(f"✅ Embedded Knowledge Graph cache saved ({len(medicines):,} medicines, {len(indications):,} indications/symptoms)")


def run_ingest(driver, records: list[dict], batch_size: int,
               dry_run: bool, clear: bool):
    total   = len(records)
    t_start = time.time()

    # Always generate local JSON cache first
    export_local_json_cache(records)

    if dry_run or not driver:
        log.info(f"DRY-RUN / LOCAL-CACHE ONLY — processed {total:,} records. No changes sent to Neo4j server.")
        return

    with driver.session() as session:
        if clear:
            log.warning("--clear flag set: wiping all Knowledge Graph nodes & relationships...")
            session.run(CLEAR_ALL)
            log.info("Knowledge Graph wiped.")

        log.info("Ensuring indices on Neo4j entity labels...")
        for cypher in INDEX_CYPHERS:
            session.run(cypher)

        log.info(f"Upserting {total:,} records into Neo4j relational graph (batch size {batch_size})...")
        for i in range(0, total, batch_size):
            batch = records[i: i + batch_size]
            session.run(UPSERT_RELATIONAL_BATCH, batch=batch)
            pct = min(100, int((i + len(batch)) / total * 100))
            log.info(f"  Graph progress: {i + len(batch):>7,} / {total:,}  ({pct:3d}%)")

        log.info("Verifying Neo4j node counts by label:")
        results = session.run(COUNT_GRAPH_NODES)
        for r in results:
            log.info(f"   - {r['label']}: {r['total']:,}")

    elapsed = time.time() - t_start
    log.info(f"✅ Neo4j Relational Knowledge Graph Ingest complete in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest medicine dataset into Neo4j Knowledge Graph & generate local cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file",       default=DEFAULT_CSV, help=f"CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Rows per transaction (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--dry-run",    action="store_true", help="Validate CSV and update local JSON cache, do NOT write to Neo4j")
    parser.add_argument("--clear",      action="store_true", help="Wipe Neo4j graph before ingesting")
    args = parser.parse_args()

    # Load & clean CSV
    records = load_and_clean(args.file)
    if not records:
        log.error("No valid records found. Check your CSV file and COLUMN_MAP.")
        sys.exit(1)

    # Connect to Neo4j (skip on dry-run)
    driver = None
    if not args.dry_run:
        log.info(f"Connecting to Neo4j: {NEO4J_URI}")
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            log.info("Neo4j connected ✅")
        except AuthError:
            log.error("Authentication failed. Check NEO4J_USER / NEO4J_PASSWORD.")
            sys.exit(1)
        except ServiceUnavailable:
            log.error("Cannot reach Neo4j. Check NEO4J_URI and network.")
            sys.exit(1)

    try:
        run_ingest(driver, records, args.batch_size, args.dry_run, args.clear)
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()