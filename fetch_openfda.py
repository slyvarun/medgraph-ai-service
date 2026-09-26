import requests
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.getenv("OPENFDA_API_KEY", "8IV5urlxZUbtNRQs1lR8tLr1yBodJZOoogO7zphh")

SPECIALTIES = [
    {
        "name": "Cardiology & Vascular",
        "query": 'indications_and_usage:("hypertension" OR "cardiovascular" OR "arrhythmia" OR "angina" OR "heart failure" OR "statin" OR "beta blocker")'
    },
    {
        "name": "Endocrinology & Diabetes",
        "query": 'indications_and_usage:("diabetes" OR "glycemic" OR "insulin" OR "metformin" OR "thyroid" OR "osteoporosis")'
    },
    {
        "name": "Oncology & Hematology",
        "query": 'indications_and_usage:("cancer" OR "oncology" OR "neoplasm" OR "chemotherapy" OR "lymphoma" OR "leukemia" OR "melanoma")'
    },
    {
        "name": "Antibiotics & Anti-Infectives",
        "query": 'indications_and_usage:("antibiotic" OR "antibacterial" OR "antiviral" OR "antifungal" OR "infection" OR "pneumonia")'
    },
    {
        "name": "Neurology & Pain",
        "query": 'indications_and_usage:("migraine" OR "epilepsy" OR "seizure" OR "parkinson" OR "neuropathy" OR "analgesic" OR "pain")'
    },
    {
        "name": "Psychiatry & Mental Health",
        "query": 'indications_and_usage:("antidepressant" OR "depression" OR "anxiety" OR "bipolar" OR "schizophrenia" OR "antipsychotic")'
    },
    {
        "name": "Pulmonology & Respiratory",
        "query": 'indications_and_usage:("asthma" OR "copd" OR "bronchodilator" OR "bronchospasm" OR "cough" OR "antihistamine")'
    },
    {
        "name": "Gastroenterology & GI",
        "query": 'indications_and_usage:("acid reflux" OR "proton pump" OR "ulcer" OR "nausea" OR "vomiting" OR "diarrhea" OR "crohn")'
    },
    {
        "name": "Rheumatology & Autoimmune",
        "query": 'indications_and_usage:("arthritis" OR "anti-inflammatory" OR "nsaid" OR "immunosuppressive" OR "lupus" OR "gout")'
    },
    {
        "name": "Dermatology & Allergy",
        "query": 'indications_and_usage:("dermatitis" OR "eczema" OR "psoriasis" OR "acne" OR "topical" OR "allergic")'
    },
    {
        "name": "Nephrology & Urology",
        "query": 'indications_and_usage:("diuretic" OR "kidney disease" OR "overactive bladder" OR "prostate" OR "urinary")'
    },
    {
        "name": "General & Emergency Medicine",
        "query": 'indications_and_usage:("anesthetic" OR "antidote" OR "emergency" OR "resuscitation" OR "fever" OR "ophthalmic")'
    }
]

def clean_text(text: str, max_length: int = 400) -> str:
    """Normalize whitespace and truncate text."""
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    if len(cleaned) > max_length:
        return cleaned[:max_length].rstrip() + "..."
    return cleaned

def get_canonical_key(brand_name: str, generic_name: str, substances: list) -> str:
    """
    Generate a canonical key for compound-level deduplication.
    Strips dosage strengths, forms, packaging, and distributor noise.
    """
    sub_key = substances[0].lower() if substances else ""
    gen_key = generic_name.lower() if generic_name and generic_name != "not specified" else ""
    brand_key = brand_name.lower()

    # Prioritize active chemical substance or generic name
    target = gen_key or sub_key or brand_key

    # Strip dosage forms, strengths, numbers
    cleaned = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ml|g|%|iu|meq)\b', '', target)
    cleaned = re.sub(
        r'\b(tablet|tablets|capsule|capsules|injection|injections|solution|solutions|oral|topical|cream|ointment|suspension|syrup|spray|patch|gel|drop|drops|powder|extended release|delayed release|chewable|film coated|usp|hydrochloride|hcl|sodium|potassium)\b',
        '',
        cleaned
    )
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    cleaned = " ".join(cleaned.split())
    
    # If generic key stripped too much, fallback to cleaned brand
    if len(cleaned) < 3:
        clean_brand = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ml|g|%)\b', '', brand_key)
        clean_brand = " ".join(re.sub(r'[^a-z0-9\s]', ' ', clean_brand).split())
        return clean_brand or brand_key
        
    return cleaned

def fetch_category_batch(category_name, search_query, skip, limit=100):
    """Fetch paginated batch from OpenFDA."""
    base_url = "https://api.fda.gov/drug/label.json"
    url = f"{base_url}?api_key={API_KEY}&search=_exists_:openfda.brand_name+AND+({search_query})&limit={limit}&skip={skip}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        return []
    except Exception:
        return []

def extract_record(item, default_category):
    """Extract clean medicine entity."""
    openfda = item.get("openfda", {})
    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    manufacturers = openfda.get("manufacturer_name", [])
    routes = openfda.get("route", [])
    product_types = openfda.get("product_type", [])
    substances = openfda.get("substance_name", [])
    pharm_classes = openfda.get("pharm_class_epc", [])

    brand_name = brand_names[0].strip() if brand_names else ""
    if not brand_name:
        return None

    indications_raw = item.get("indications_and_usage", [""])[0]
    indications = clean_text(indications_raw, max_length=500)
    if not indications:
        return None

    clean_substances = [s.strip().title() for s in substances if s.strip()][:5]
    clean_classes = [c.replace("[EPC]", "").strip().title() for c in pharm_classes if c.strip()][:4]

    adverse = clean_text(item.get("adverse_reactions", [""])[0], max_length=350)
    warnings = clean_text(item.get("warnings", [""])[0] or item.get("boxed_warning", [""])[0], max_length=350)
    contra = clean_text(item.get("contraindications", [""])[0], max_length=350)
    interactions = clean_text(item.get("drug_interactions", [""])[0], max_length=350)

    generic_name = generic_names[0].strip() if generic_names else (clean_substances[0] if clean_substances else "Not Specified")

    return {
        "brand_name": brand_name,
        "generic_name": generic_name,
        "category": default_category,
        "manufacturer": manufacturers[0].strip() if manufacturers else "Unknown",
        "route": routes[0].strip() if routes else "Oral",
        "product_type": product_types[0].strip() if product_types else "Prescription",
        "substances": clean_substances,
        "pharm_classes": clean_classes,
        "indications": indications,
        "warnings": warnings or "None specified in label summary",
        "adverse_reactions": adverse or "None specified in label summary",
        "contraindications": contra or "None specified in label summary",
        "drug_interactions": interactions or "None specified in label summary"
    }

def harvest_variety_dataset(target_total=5000, per_specialty=450, output_file="medicine_dataset.json"):
    """Harvest across 12 specialties with strict canonical zero-duplication."""
    print(f"=== Harvesting 12-Specialty Dataset (Target: ~{target_total} Zero-Duplicate Varieties) ===")
    start_time = time.time()
    
    canonical_store = {}  # canonical_key -> medicine_record
    
    for spec in SPECIALTIES:
        spec_name = spec["name"]
        spec_query = spec["query"]
        print(f"\n--- Harvesting Specialty: {spec_name} ---")
        
        harvested_in_spec = 0
        skip = 0
        consecutive_empty = 0
        
        while harvested_in_spec < per_specialty and skip < 8000:
            offsets = [skip + (i * 100) for i in range(4)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(fetch_category_batch, spec_name, spec_query, off): off for off in offsets}
                batch_items = []
                for f in as_completed(futures):
                    batch_items.extend(f.result())
                    
            if not batch_items:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                skip += 400
                continue
                
            consecutive_empty = 0
            
            for item in batch_items:
                rec = extract_record(item, default_category=spec_name)
                if not rec:
                    continue
                    
                canon_key = get_canonical_key(rec["brand_name"], rec["generic_name"], rec["substances"])
                
                # Check for duplicate compound
                if canon_key not in canonical_store:
                    canonical_store[canon_key] = rec
                    harvested_in_spec += 1
                    if harvested_in_spec >= per_specialty:
                        break
                else:
                    # Update existing record if new label has richer clinical info
                    existing = canonical_store[canon_key]
                    if len(rec["indications"]) > len(existing["indications"]):
                        canonical_store[canon_key] = rec
                        
            skip += 400
            print(f"  [{spec_name}] Harvested {harvested_in_spec}/{per_specialty} (Total unique varieties: {len(canonical_store)})")
            time.sleep(0.2)

    dataset = list(canonical_store.values())
    elapsed = round(time.time() - start_time, 1)
    
    print(f"\n=== Completed 12-Specialty Extraction ===")
    print(f"Total Unique Canonical Varieties: {len(dataset)} in {elapsed} seconds")
    
    # Specialty breakdown
    counts = {}
    for d in dataset:
        c = d["category"]
        counts[c] = counts.get(c, 0) + 1
    for c, cnt in counts.items():
        print(f"  • {c}: {cnt} medicines")
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(dataset)} zero-duplicate records to {output_file}")
    return dataset

if __name__ == "__main__":
    harvest_variety_dataset(target_total=5000, per_specialty=450)
