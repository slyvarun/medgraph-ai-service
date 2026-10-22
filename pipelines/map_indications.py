"""
MedGraph Nexus - Indication & Symptom Entity Mapping Engine
-----------------------------------------------------------
Zero-cost, local clinical ontology regex parser in RAM.
Extracts standardized clinical symptom and indication entities from unstructured
openFDA label texts ('indications' and 'description') and establishes explicit
(:Medicine)-[:TREATS]->(:Symptom) and (:Ingredient)-[:TREATS]->(:Symptom) edges
using idempotent Cypher UNWIND batches (capped at 1,000).
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Pattern
from neo4j import GraphDatabase
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
load_dotenv(os.path.join(parent_dir, ".env"))
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Standardized Clinical Ontology (SNOMED-CT / MeSH / openFDA Aligned)
CLINICAL_ONTOLOGY: Dict[str, str] = {
    # Cardiovascular & Renal
    "Hypertension": r"\b(hypertension|hypertensive|essential hypertension|arterial hypertension)\b",
    "High Blood Pressure": r"\b(high blood pressure|elevated blood pressure)\b",
    "Angina": r"\b(angina|angina pectoris|ischemic chest pain)\b",
    "Heart Failure": r"\b(heart failure|congestive heart failure|chf|myocardial failure)\b",
    "Edema": r"\b(edema|fluid retention|peripheral edema|pulmonary edema)\b",
    "Arrhythmia": r"\b(arrhythmia|cardiac arrhythmia|atrial fibrillation|tachycardia|ventricular arrhythmia)\b",
    
    # Neurology & Pain
    "Headache": r"\b(headache|headaches|tension headache|cluster headache|cephalea)\b",
    "Migraine": r"\b(migraine|migraines|acute migraine|migraine prophylaxis)\b",
    "Neuropathic Pain": r"\b(neuropathic pain|neuropathy|diabetic neuropathy|postherpetic neuralgia|sciatica)\b",
    "Musculoskeletal Pain": r"\b(musculoskeletal pain|myalgia|muscle spasm|back pain|neck pain)\b",
    "Seizures": r"\b(seizure|seizures|partial onset seizures|tonic clonic|convulsions)\b",
    "Epilepsy": r"\b(epilepsy|epileptic|refractory epilepsy)\b",
    
    # Endocrine & Metabolic
    "Diabetes": r"\b(diabetes mellitus|diabetic patients|type 1 diabetes)\b",
    "Type 2 Diabetes": r"\b(type 2 diabetes|type ii diabetes|t2dm|non-insulin-dependent)\b",
    "Hyperglycemia": r"\b(hyperglycemia|elevated blood glucose|high blood sugar)\b",
    "Hypoglycemia": r"\b(hypoglycemia|low blood sugar)\b",
    "Dyslipidemia": r"\b(dyslipidemia|mixed dyslipidemia|hyperlipidemia|hypertriglyceridemia)\b",
    "Hypercholesterolemia": r"\b(hypercholesterolemia|elevated cholesterol|high cholesterol|elevated ldl)\b",
    "Hypothyroidism": r"\b(hypothyroidism|thyroid deficiency|underactive thyroid)\b",
    
    # Respiratory & Allergy
    "Asthma": r"\b(asthma|bronchial asthma|asthmatic|bronchospasm)\b",
    "COPD": r"\b(copd|chronic obstructive pulmonary disease|emphysema|chronic bronchitis)\b",
    "Bronchitis": r"\b(bronchitis|acute bronchitis|tracheobronchitis)\b",
    "Allergic Rhinitis": r"\b(allergic rhinitis|seasonal allergies|hay fever|perennial rhinitis)\b",
    "Cough": r"\b(cough|coughing|tussive|antitussive)\b",
    "Nasal Congestion": r"\b(nasal congestion|stuffy nose|rhinorrhea|sinus congestion)\b",
    
    # Infectious & Gastrointestinal
    "Bacterial Infection": r"\b(bacterial infection|bacterial infections|susceptible bacterial|streptococcal|staphylococcal|pneumococcal|pneumonia|urinary tract infection|uti)\b",
    "GERD": r"\b(gerd|gastroesophageal reflux|acid reflux|reflux esophagitis|heartburn)\b",
    "Peptic Ulcer": r"\b(peptic ulcer|gastric ulcer|duodenal ulcer|stress ulcer)\b",
    "Nausea": r"\b(nausea|antiemetic|queasiness)\b",
    "Vomiting": r"\b(vomiting|emesis)\b",
    "Diarrhea": r"\b(diarrhea|diarrhoeal|loose stool)\b",
    
    # Psychiatry & Sleep
    "Anxiety": r"\b(anxiety|generalized anxiety disorder|gad|anxiolytic)\b",
    "Depression": r"\b(depression|major depressive disorder|mdd|antidepressant|dysthymia)\b",
    "Insomnia": r"\b(insomnia|sleeplessness|sleep disorder|difficulty falling asleep)\b",
    "Panic Disorder": r"\b(panic disorder|panic attacks|agoraphobia)\b",
    "Bipolar Disorder": r"\b(bipolar disorder|bipolar depression|bipolar mania|manic depressive)\b",
    
    # General Outpatient & Musculoskeletal
    "Arthritis": r"\b(arthritis|osteoarthritis|rheumatoid arthritis|psoriatic arthritis)\b",
    "Inflammation": r"\b(inflammation|inflammatory condition|swelling)\b",
    "Pain": r"\b(acute pain|chronic pain|moderate to severe pain|mild to moderate pain|postoperative pain)\b",
    "Fever": r"\b(fever|pyrexia|antipyretic|febrile)\b"
}

# Pre-compile regexes in RAM for O(1) matching speed
COMPILED_ONTOLOGY: Dict[str, Pattern] = {
    symptom: re.compile(pattern, re.IGNORECASE)
    for symptom, pattern in CLINICAL_ONTOLOGY.items()
}


def extract_symptoms_from_text(text: str) -> List[str]:
    """Matches text against the clinical ontology in RAM using word boundaries."""
    if not text:
        return []
    matches = []
    text_lower = text.lower()
    for symptom, regex in COMPILED_ONTOLOGY.items():
        if regex.search(text_lower):
            matches.append(symptom)
    return matches


def ingest_indication_batch(tx, batch: List[dict]):
    """
    Idempotent Cypher UNWIND batch execution.
    Links (:Medicine) and (:Ingredient) to unified (:Symptom:Indication) nodes.
    """
    query = """
    UNWIND $batch AS record
    MATCH (m) WHERE (m:Medicine OR m:Brand) AND m.rxcui = record.rxcui
    OPTIONAL MATCH (m)-[:CONTAINS]->(i:Ingredient)
    WITH m, collect(DISTINCT i) AS ingredients, record.symptoms AS symptoms
    UNWIND symptoms AS symptom_name
    MERGE (s:Symptom {name: symptom_name})
    ON CREATE SET s:Indication
    MERGE (m)-[:TREATS]->(s)
    FOREACH (ing IN ingredients |
        MERGE (ing)-[:TREATS]->(s)
    )
    """
    tx.run(query, batch=batch)


def run_indication_mapping(batch_size: int = 1000):
    """
    Fetches unmapped medicines with indication texts, extracts clinical entities,
    and writes relationships in batches of 1,000.
    """
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    print("[*] Connecting to Neo4j Aura to query unmapped indications...")
    with driver.session() as session:
        # Fetch only nodes with indication/description texts that haven't been linked to :Symptom
        fetch_query = """
        MATCH (m)
        WHERE (m:Brand OR m:Medicine)
          AND ( (m.indications IS NOT NULL AND m.indications <> '') OR (m.description IS NOT NULL AND m.description <> '') )
          AND NOT (m)-[:TREATS]->(:Symptom)
        RETURN m.rxcui AS rxcui, coalesce(m.indications, '') + ' ' + coalesce(m.description, '') AS text
        """
        results = session.run(fetch_query)
        candidates = [{"rxcui": r["rxcui"], "text": r["text"]} for r in results]
        print(f"[*] Found {len(candidates):,} candidate nodes with unmapped clinical text.")

        if not candidates:
            print("[+] All available medicines are already mapped to :Symptom nodes.")
            run_audit(session)
            driver.close()
            return

        # Perform local regex entity extraction in RAM
        print(f"[*] Performing zero-cost clinical entity extraction in RAM...")
        records_to_ingest = []
        for cand in candidates:
            symptoms = extract_symptoms_from_text(cand["text"])
            if symptoms:
                records_to_ingest.append({
                    "rxcui": cand["rxcui"],
                    "symptoms": symptoms
                })

        print(f"[+] Identified clinical symptoms for {len(records_to_ingest):,} / {len(candidates):,} records.")

        # Batch insert into Neo4j Aura using UNWIND
        total_batches = (len(records_to_ingest) + batch_size - 1) // batch_size
        print(f"[*] Batch writing to AuraDB in chunks of {batch_size} ({total_batches} batches)...")

        for i in range(0, len(records_to_ingest), batch_size):
            batch = records_to_ingest[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            session.execute_write(ingest_indication_batch, batch)
            print(f"  -> Batch {batch_num}/{total_batches} ({len(batch)} records) committed.")

        print("[SUCCESS] All indication mappings committed.")
        run_audit(session)

    driver.close()


def run_audit(session):
    """Executes the verification gate query and metrics summary."""
    print("\n" + "=" * 60)
    print("AUDIT: TOP 10 SYMPTOMS BY BRAND COUNT (MULTI-HOP TRAVERSAL)")
    print("=" * 60)

    audit_query = """
    MATCH (s:Symptom)<-[:TREATS]-(i:Ingredient)<-[:CONTAINS]-(b:Brand)
    RETURN s.name AS Symptom, count(DISTINCT i) AS Ingredients, count(DISTINCT b) AS Brands
    ORDER BY Brands DESC LIMIT 10
    """
    res = session.run(audit_query)
    rows = list(res)
    if not rows:
        print("[!] No multi-hop (:Brand)->(:Ingredient)->(:Symptom) paths found.")
    else:
        print(f"{'Symptom':<26} | {'Ingredients':<12} | {'Brands':<10}")
        print("-" * 54)
        for r in rows:
            print(f"{r['Symptom']:<26} | {r['Ingredients']:<12} | {r['Brands']:<10}")

    print("\n" + "=" * 60)
    print("OVERALL AURADB METRICS AUDIT")
    print("=" * 60)
    tot_n = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
    tot_r = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
    print(f"TOTAL NODES:         {tot_n:,} / 200,000 ({(tot_n/200000)*100:.2f}%)")
    print(f"TOTAL RELATIONSHIPS: {tot_r:,} / 400,000 ({(tot_r/400000)*100:.2f}%)")
    print(f"SAFE HEADROOM:       {180000 - tot_n:,} nodes remaining")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGraph Nexus - Indication & Symptom Entity Mapping Engine")
    parser.add_argument("--batch-size", type=int, default=1000, help="Cypher UNWIND batch size (default: 1000)")
    args = parser.parse_args()

    run_indication_mapping(batch_size=args.batch_size)
