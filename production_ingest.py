"""
production_ingest.py  —  MedGraph Nexus  |  Data Pipeline
==========================================================
Reads medicine_dataset.csv (from Kaggle) and upserts all
records into Neo4j as :Medicine nodes.

Usage:
    python production_ingest.py                          # default CSV path
    python production_ingest.py --file /path/to/csv     # custom path
    python production_ingest.py --batch-size 1000        # larger batches
    python production_ingest.py --dry-run               # validate only, no writes
    python production_ingest.py --clear                 # wipe all Medicine nodes first

COLUMN_MAP  (left = CSV header, right = Neo4j property):
    "Name"           → name            ← used as the MERGE key
    "Category"       → category
    "Dosage Form"    → dosage_form
    "Strength"       → strength
    "Manufacturer"   → manufacturer
    "Indication"     → indication
    "Classification" → classification
"""

import os
import sys
import time
import argparse
import logging

import pandas as pd
from neo4j import GraphDatabase
from neo4j import ServiceUnavailable, AuthError
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

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAP  — single source of truth for the schema
# Change ONLY the left side (CSV header) if your file uses different names.
# NEVER change the right side — it must match the Cypher in query_agent.py.
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "Name":           "name",            # ← MERGE key — must be unique
    "Category":       "category",
    "Dosage Form":    "dosage_form",
    "Strength":       "strength",
    "Manufacturer":   "manufacturer",
    "Indication":     "indication",
    "Classification": "classification",
}

# ── Cypher statements ─────────────────────────────────────────────────────────

# Create index once (IF NOT EXISTS makes it idempotent)
CREATE_INDEX = """
CREATE INDEX medicine_name IF NOT EXISTS
FOR (m:Medicine) ON (m.name)
"""

# Upsert — MERGE prevents duplicates; SET updates all properties
UPSERT_BATCH = """
UNWIND $batch AS row
MERGE (m:Medicine {name: row.name})
SET
    m.category       = row.category,
    m.dosage_form    = row.dosage_form,
    m.strength       = row.strength,
    m.manufacturer   = row.manufacturer,
    m.indication     = row.indication,
    m.classification = row.classification
"""

# Optional: remove all Medicine nodes before a fresh ingest
CLEAR_ALL = "MATCH (m:Medicine) DETACH DELETE m"

# Count nodes for verification
COUNT_NODES = "MATCH (m:Medicine) RETURN count(m) AS total"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_env():
    missing = [k for k in ("NEO4J_URI", "NEO4J_PASSWORD") if not os.getenv(k)]
    if missing:
        log.error(f"Missing .env variables: {', '.join(missing)}")
        sys.exit(1)


def load_and_clean(filepath: str) -> list[dict]:
    """Read CSV, rename columns, normalise strings, return list of dicts."""
    log.info(f"Reading: {filepath}")

    # Try UTF-8 first, fall back to latin-1 (common for medicine datasets)
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("UTF-8 failed — retrying with latin-1")
        df = pd.read_csv(filepath, encoding="latin-1")

    log.info(f"Loaded {len(df):,} rows | columns: {df.columns.tolist()}")

    # Warn about any expected headers that are missing
    missing_headers = [h for h in COLUMN_MAP if h not in df.columns]
    if missing_headers:
        log.warning(
            f"These CSV headers were NOT found and will be empty: {missing_headers}\n"
            "  → Update COLUMN_MAP in this file to match your CSV exactly."
        )

    # Rename columns that exist
    existing_renames = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Keep only the target columns; add empty string for any that are missing
    target_cols = list(COLUMN_MAP.values())
    for col in target_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[target_cols].copy()

    # Normalise all columns: strip, convert to string, replace nan/"nan"
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": "", "None": "", "none": ""})
        )

    # Drop rows with empty name — they cannot be MERGE'd
    before = len(df)
    df = df[df["name"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} rows with empty 'name'.")

    log.info(f"Clean records ready: {len(df):,}")
    return df.to_dict("records")


def run_ingest(driver, records: list[dict], batch_size: int,
               dry_run: bool, clear: bool):
    total   = len(records)
    t_start = time.time()

    if dry_run:
        log.info(f"DRY RUN — would write {total:,} records in batches of {batch_size}. No changes made.")
        for i in range(0, total, batch_size):
            b = records[i: i + batch_size]
            log.info(f"  [dry-run] rows {i:>6} – {i + len(b) - 1:>6}")
        return

    with driver.session() as session:
        # Optional: wipe existing nodes
        if clear:
            log.warning("--clear flag set: deleting all existing :Medicine nodes...")
            session.run(CLEAR_ALL)
            log.info("All Medicine nodes deleted.")

        # Ensure index exists
        session.run(CREATE_INDEX)
        log.info("Index on :Medicine(name) ensured.")

        # Batch upsert
        log.info(f"Writing {total:,} records in batches of {batch_size}...")
        for i in range(0, total, batch_size):
            batch = records[i: i + batch_size]
            session.run(UPSERT_BATCH, batch=batch)
            pct = min(100, int((i + len(batch)) / total * 100))
            log.info(f"  Progress: {i + len(batch):>7,} / {total:,}  ({pct:3d}%)")

        # Verify
        result = session.run(COUNT_NODES).single()
        total_in_db = result["total"] if result else "unknown"

    elapsed = time.time() - t_start
    log.info(f"✅ Ingest complete — {total:,} rows processed in {elapsed:.1f}s")
    log.info(f"   Total :Medicine nodes now in Neo4j: {total_in_db:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest medicine_dataset.csv into Neo4j for MedGraph Nexus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file",       default=DEFAULT_CSV, help=f"CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Rows per transaction (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--dry-run",    action="store_true", help="Validate CSV, do NOT write to Neo4j")
    parser.add_argument("--clear",      action="store_true", help="Delete all Medicine nodes before ingesting")
    args = parser.parse_args()

    validate_env()

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