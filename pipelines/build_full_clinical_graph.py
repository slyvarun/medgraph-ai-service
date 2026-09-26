import os
import sys
import zipfile
import json
import time
import re
import requests

API_KEY = os.getenv("OPENFDA_API_KEY", "8IV5urlxZUbtNRQs1lR8tLr1yBodJZOoogO7zphh")

# 14 Official OpenFDA Drug Label Partitions
PARTITIONS = [
    {"num": 1,  "size": 130.61, "url": "https://download.open.fda.gov/drug/label/drug-label-0001-of-0014.json.zip", "alt_names": ["drug-label-0001-of-0014.json (1).zip", "drug-label-0001-of-0014.json.zip"]},
    {"num": 2,  "size": 142.68, "url": "https://download.open.fda.gov/drug/label/drug-label-0002-of-0014.json.zip", "alt_names": ["drug-label-0002-of-0014.json.zip"]},
    {"num": 3,  "size": 137.18, "url": "https://download.open.fda.gov/drug/label/drug-label-0003-of-0014.json.zip", "alt_names": ["drug-label-0003-of-0014.json.zip"]},
    {"num": 4,  "size": 137.41, "url": "https://download.open.fda.gov/drug/label/drug-label-0004-of-0014.json.zip", "alt_names": ["drug-label-0004-of-0014.json.zip"]},
    {"num": 5,  "size": 125.18, "url": "https://download.open.fda.gov/drug/label/drug-label-0005-of-0014.json.zip", "alt_names": ["drug-label-0005-of-0014.json.zip"]},
    {"num": 6,  "size": 135.70, "url": "https://download.open.fda.gov/drug/label/drug-label-0006-of-0014.json.zip", "alt_names": ["drug-label-0006-of-0014.json.zip"]},
    {"num": 7,  "size": 141.78, "url": "https://download.open.fda.gov/drug/label/drug-label-0007-of-0014.json.zip", "alt_names": ["drug-label-0007-of-0014.json.zip"]},
    {"num": 8,  "size": 136.56, "url": "https://download.open.fda.gov/drug/label/drug-label-0008-of-0014.json.zip", "alt_names": ["drug-label-0008-of-0014.json.zip"]},
    {"num": 9,  "size": 126.57, "url": "https://download.open.fda.gov/drug/label/drug-label-0009-of-0014.json.zip", "alt_names": ["drug-label-0009-of-0014.json.zip"]},
    {"num": 10, "size": 130.35, "url": "https://download.open.fda.gov/drug/label/drug-label-0010-of-0014.json.zip", "alt_names": ["drug-label-0010-of-0014.json.zip"]},
    {"num": 11, "size": 143.28, "url": "https://download.open.fda.gov/drug/label/drug-label-0011-of-0014.json.zip", "alt_names": ["drug-label-0011-of-0014.json.zip"]},
    {"num": 12, "size": 134.23, "url": "https://download.open.fda.gov/drug/label/drug-label-0012-of-0014.json.zip", "alt_names": ["drug-label-0012-of-0014.json.zip"]},
    {"num": 13, "size": 128.83, "url": "https://download.open.fda.gov/drug/label/drug-label-0013-of-0014.json.zip", "alt_names": ["drug-label-0013-of-0014.json.zip"]},
    {"num": 14, "size": 23.29,  "url": "https://download.open.fda.gov/drug/label/drug-label-0014-of-0014.json.zip", "alt_names": ["drug-label-0014-of-0014.json.zip"]}
]

SPECIALTY_KEYWORDS = {
    "Cardiology & Vascular": ["hypertension", "cardiovascular", "arrhythmia", "angina", "heart", "statin", "beta blocker", "blood pressure", "vasodilator", "cholesterol", "lipid"],
    "Endocrinology & Diabetes": ["diabetes", "glycemic", "insulin", "metformin", "thyroid", "osteoporosis", "glucose", "hormone", "estrogen", "testosterone"],
    "Oncology & Hematology": ["cancer", "oncology", "neoplasm", "chemotherapy", "lymphoma", "leukemia", "melanoma", "tumor", "carcinoma", "antineoplastic", "hematologic"],
    "Antibiotics & Anti-Infectives": ["antibiotic", "antibacterial", "antiviral", "antifungal", "infection", "pneumonia", "bacterial", "microbial", "penicillin", "cephalosporin"],
    "Neurology & Pain": ["migraine", "epilepsy", "seizure", "parkinson", "neuropathy", "analgesic", "pain", "headache", "neurological", "sedative", "insomnia"],
    "Psychiatry & Mental Health": ["antidepressant", "depression", "anxiety", "bipolar", "schizophrenia", "antipsychotic", "mood", "psychiatric", "ssri", "snri"],
    "Pulmonology & Respiratory": ["asthma", "copd", "bronchodilator", "bronchospasm", "cough", "antihistamine", "respiratory", "inhalation", "bronchial"],
    "Gastroenterology & GI": ["acid reflux", "proton pump", "ulcer", "nausea", "vomiting", "diarrhea", "crohn", "gastrointestinal", "bowel", "antacid", "constipation"],
    "Rheumatology & Autoimmune": ["arthritis", "anti-inflammatory", "nsaid", "immunosuppressive", "lupus", "gout", "joint", "autoimmune"],
    "Dermatology & Allergy": ["dermatitis", "eczema", "psoriasis", "acne", "topical", "allergy", "allergic", "skin", "rash", "corticosteroid"],
    "Nephrology & Urology": ["diuretic", "kidney", "renal", "overactive bladder", "prostate", "urinary", "bladder", "dialysis"],
    "General & Emergency Medicine": ["anesthetic", "antidote", "emergency", "resuscitation", "fever", "fever reducer", "ophthalmic"]
}

def infer_therapeutic_category(indications: str, pharm_classes: list, default_type: str) -> str:
    text_to_check = (indications + " " + " ".join(pharm_classes)).lower()
    for specialty, kws in SPECIALTY_KEYWORDS.items():
        if any(kw in text_to_check for kw in kws):
            return specialty
    if default_type == "CELLULAR THERAPY":
        return "Cellular & Regenerative Medicine"
    return "General Medicine"

def clean_text(text: str, max_length: int = 400) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    if len(cleaned) > max_length:
        return cleaned[:max_length].rstrip() + "..."
    return cleaned

def get_canonical_key(brand_name: str, generic_name: str, substances: list) -> str:
    sub_key = substances[0].lower() if substances else ""
    gen_key = generic_name.lower() if generic_name and generic_name != "not specified" else ""
    brand_key = brand_name.lower()
    target = gen_key or sub_key or brand_key

    cleaned = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ml|g|%|iu|meq)\b', '', target)
    cleaned = re.sub(
        r'\b(tablet|tablets|capsule|capsules|injection|injections|solution|solutions|oral|topical|cream|ointment|suspension|syrup|spray|patch|gel|drop|drops|powder|extended release|delayed release|chewable|film coated|usp|hydrochloride|hcl|sodium|potassium)\b',
        '',
        cleaned
    )
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    cleaned = " ".join(cleaned.split())
    
    if len(cleaned) < 3:
        clean_brand = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ml|g|%)\b', '', brand_key)
        clean_brand = " ".join(re.sub(r'[^a-z0-9\s]', ' ', clean_brand).split())
        return clean_brand or brand_key
        
    return cleaned

def download_file(url, target_path, expected_mb):
    print(f"    Downloading {os.path.basename(target_path)} ({expected_mb} MB)...", flush=True)
    for attempt in range(4):
        t0 = time.time()
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            downloaded = 0
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            elapsed = round(time.time() - t0, 1)
            mb = round(downloaded / (1024 * 1024), 1)
            speed = round(mb / elapsed, 1) if elapsed > 0 else 0
            print(f"    -> Download complete: {mb} MB in {elapsed}s ({speed} MB/s)", flush=True)
            return True
        except Exception as e:
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass
            if attempt == 3:
                print(f"    -> Download failed after 4 attempts: {e}", flush=True)
                return False
            wait_sec = 3 * (attempt + 1)
            print(f"    -> Attempt {attempt + 1} failed ({e}), retrying in {wait_sec}s...", flush=True)
            time.sleep(wait_sec)
    return False

def extract_clinical_record(item, rxnorm_map):
    openfda = item.get("openfda", {})
    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    manufacturers = openfda.get("manufacturer_name", [])
    routes = openfda.get("route", [])
    product_types = openfda.get("product_type", [])
    substances = openfda.get("substance_name", [])
    pharm_classes = openfda.get("pharm_class_epc", [])
    set_id = item.get("set_id", "").lower()

    brand_name = brand_names[0].strip().title() if brand_names else ""
    generic_name = generic_names[0].strip().title() if generic_names else (substances[0].strip().title() if substances else "")

    if not brand_name and not generic_name:
        return None

    if not brand_name:
        brand_name = generic_name
    if not generic_name:
        generic_name = brand_name

    # Clinical Sections
    indications_raw = item.get("indications_and_usage", [""])[0]
    indications = clean_text(indications_raw, max_length=500)
    if not indications:
        indications = clean_text(item.get("purpose", [""])[0] or item.get("description", [""])[0], max_length=500)
    if not indications:
        return None

    clean_substances = [s.strip().title() for s in substances if s.strip()][:5]
    clean_classes = [c.replace("[EPC]", "").strip().title() for c in pharm_classes if c.strip()][:4]

    adverse = clean_text(item.get("adverse_reactions", [""])[0], max_length=350)
    warnings = clean_text(item.get("warnings", [""])[0] or item.get("boxed_warning", [""])[0], max_length=350)
    contra = clean_text(item.get("contraindications", [""])[0], max_length=350)
    interactions = clean_text(item.get("drug_interactions", [""])[0], max_length=350)

    product_type = product_types[0].strip() if product_types else "HUMAN PRESCRIPTION DRUG"
    category = infer_therapeutic_category(indications, clean_classes, product_type)

    rxcui = rxnorm_map.get(set_id, {}).get("rxcui", "")

    return {
        "brand_name": brand_name,
        "generic_name": generic_name,
        "product_type": product_type,
        "category": category,
        "manufacturer": manufacturers[0].strip() if manufacturers else "Unknown",
        "route": routes[0].strip() if routes else "Oral",
        "substances": clean_substances,
        "pharm_classes": clean_classes,
        "indications": indications,
        "warnings": warnings or "None specified in label summary",
        "adverse_reactions": adverse or "None specified in label summary",
        "contraindications": contra or "None specified in label summary",
        "drug_interactions": interactions or "None specified in label summary",
        "rxcui": rxcui
    }

def find_file(filename):
    for d in ["data", "../data", ".", ".."]:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None

def main():
    print("=" * 80, flush=True)
    print("MedGraph Master Pipeline: Harvesting All 14 OpenFDA Partitions & RxNorm", flush=True)
    print("=" * 80, flush=True)

    # 1. Load RxNorm Mappings
    print("\n[Step 1/3] Loading NIH RxNorm Clinical Registry...", flush=True)
    rxnorm_map = {}  # set_id -> {rxcui, name}
    rxnorm_zip = find_file("rxnorm_mappings.zip")
    if rxnorm_zip and os.path.exists(rxnorm_zip):
        with zipfile.ZipFile(rxnorm_zip) as z:
            with z.open("rxnorm_mappings.txt") as f:
                header = f.readline()
                for line in f:
                    parts = line.decode("utf-8", errors="ignore").split("|")
                    if len(parts) >= 4:
                        sid = parts[0].strip().lower()
                        rxcui = parts[2].strip()
                        rxstr = parts[3].strip()
                        if sid and sid not in rxnorm_map:
                            rxnorm_map[sid] = {"rxcui": rxcui, "rxstring": rxstr}
        print(f"Loaded {len(rxnorm_map):,} official RxNorm-mapped drug Set IDs!", flush=True)
    else:
        print("Warning: rxnorm_mappings.zip not found, proceeding with FDA application filter.", flush=True)

    # 2. Iterate All 14 Partitions
    print("\n[Step 2/3] Processing All 14 OpenFDA Partitions (~262,887 raw labels)...", flush=True)
    canonical_store = {}  # canon_key -> medicine_record
    start_time = time.time()

    for p in PARTITIONS:
        num = p["num"]
        print(f"\n--- Processing Partition {num}/14 ---", flush=True)
        
        # Check existing file
        zip_path = None
        for name in p["alt_names"]:
            found = find_file(name)
            if found:
                zip_path = found
                break

        if not zip_path:
            target = f"drug-label-{num:04d}-of-0014.json.zip"
            success = download_file(p["url"], target, p["size"])
            if success:
                zip_path = target
            else:
                print(f"Skipping partition {num} due to download error.", flush=True)
                continue

        # Process Zip Partition
        t_parse = time.time()
        try:
            with zipfile.ZipFile(zip_path) as z:
                inner_name = [n for n in z.namelist() if n.endswith(".json")][0]
                with z.open(inner_name) as f:
                    data = json.load(f)
                    records = data.get("results", [])
                    
                    added_in_part = 0
                    for r in records:
                        set_id = r.get("set_id", "").lower()
                        openfda = r.get("openfda", {})
                        app_nums = openfda.get("application_number", [])
                        prod_types = openfda.get("product_type", [])
                        
                        is_rxnorm = set_id in rxnorm_map
                        is_nda_anda = any(a.startswith(("NDA", "ANDA", "BLA")) for a in app_nums)
                        is_rx = "HUMAN PRESCRIPTION DRUG" in prod_types
                        is_cellular = "CELLULAR THERAPY" in prod_types
                        
                        # Strict Clinical Filter
                        if not (is_rxnorm or is_nda_anda or is_rx or is_cellular):
                            continue
                            
                        # Exclude pure cosmetic / sunscreen / homeopathic without real application
                        brand_lower = " ".join(openfda.get("brand_name", [])).lower()
                        if "sunscreen" in brand_lower or "spf" in brand_lower:
                            if not is_nda_anda and not is_rx:
                                continue

                        rec = extract_clinical_record(r, rxnorm_map)
                        if not rec:
                            continue

                        key = get_canonical_key(rec["brand_name"], rec["generic_name"], rec["substances"])
                        if key not in canonical_store:
                            canonical_store[key] = rec
                            added_in_part += 1
                        else:
                            if len(rec["indications"]) > len(canonical_store[key]["indications"]):
                                canonical_store[key] = rec

                    elapsed_p = round(time.time() - t_parse, 1)
                    print(f"    Parsed {len(records):,} raw labels -> +{added_in_part:,} new unique medicines in {elapsed_p}s (Total Unique: {len(canonical_store):,})", flush=True)

        except Exception as e:
            print(f"Error reading {zip_path}: {e}", flush=True)

    master_dataset = list(canonical_store.values())
    total_time = round(time.time() - start_time, 1)

    print("\n" + "=" * 80, flush=True)
    print(f"Harvest Complete in {total_time}s!", flush=True)
    print(f"Total Unique Clinical Medicines: {len(master_dataset):,}", flush=True)
    print("=" * 80, flush=True)

    # Save to medicine_dataset.json
    out_file = "medicine_dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_dataset, f, indent=2, ensure_ascii=False)
    file_mb = round(os.path.getsize(out_file) / (1024 * 1024), 2)
    print(f"Saved master dataset to {out_file} ({file_mb} MB)", flush=True)

    # Breakdown by Product Type
    pt_counts = {}
    spec_counts = {}
    for d in master_dataset:
        pt = d["product_type"]
        pt_counts[pt] = pt_counts.get(pt, 0) + 1
        spec = d["category"]
        spec_counts[spec] = spec_counts.get(spec, 0) + 1

    print("\nBreakdown by Product Type:")
    for pt, cnt in sorted(pt_counts.items(), key=lambda x: -x[1]):
        print(f"  • {pt}: {cnt:,} medicines")

    print("\nBreakdown by Clinical Specialty:")
    for spec, cnt in sorted(spec_counts.items(), key=lambda x: -x[1]):
        print(f"  • {spec}: {cnt:,} medicines")

    # 3. Batch Ingest into Neo4j
    print("\n[Step 3/3] Ingesting Master Dataset into Neo4j Aura...", flush=True)
    from ingest_graph import KnowledgeGraphIngestor, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    ingestor = KnowledgeGraphIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        ingestor.ingest_data(out_file, batch_size=250, clean_first=True)
        ingestor.verify_graph_counts()
    finally:
        ingestor.close()

if __name__ == "__main__":
    main()
